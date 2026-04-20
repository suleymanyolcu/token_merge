"""Minimal ToMe patch adapted from facebookresearch/ToMe for timm 0.9.x."""

import math
from typing import Callable, List, Tuple, Union

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer


def _identity(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """Balanced bipartite token matching from ToMe."""
    protected = int(class_token) + int(distill_token)
    token_count = metric.shape[1]
    r = min(r, (token_count - protected) // 2)
    if r <= 0:
        return _identity, _identity

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        src_metric, dst_metric = metric[..., ::2, :], metric[..., 1::2, :]
        scores = src_metric @ dst_metric.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        batch_size, src_tokens, channels = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(batch_size, src_tokens - r, channels))
        src = src.gather(dim=-2, index=src_idx.expand(batch_size, r, channels))
        dst = dst.scatter_reduce(-2, dst_idx.expand(batch_size, r, channels), src, reduce=mode)
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        batch_size, _, channels = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(batch_size, r, channels))
        out = torch.zeros(batch_size, metric.shape[1], channels, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(batch_size, unm_len, channels), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(batch_size, r, channels), src=src)
        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weighted token merge that keeps token-size bookkeeping."""
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    return x, size


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    if isinstance(r, tuple):
        r, inflect = r
    else:
        inflect = 0

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / max(num_layers - 1, 1)
    return [int(min_val + step * i) for i in range(num_layers)]


class ToMeAttention(Attention):
    """Modern timm attention patched to return both output and merge metric."""

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_count, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, k.mean(1)


class ToMeBlock(Block):
    """Modern timm block patched to merge tokens between attention and MLP."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        attn_out, metric = self.attn(self.norm1(x), attn_size)
        x = x + self.drop_path1(self.ls1(attn_out))

        r = self._tome_info["r"].pop(0)
        if r > 0:
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                class_token=self._tome_info["class_token"],
                distill_token=self._tome_info["distill_token"],
            )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def apply_patch(model: VisionTransformer, prop_attn: bool = False) -> VisionTransformer:
    """Apply a minimal ToMe patch to a timm VisionTransformer in-place."""
    if model.__class__.__name__ == "ToMeVisionTransformer":
        return model

    def make_tome_class(transformer_class):
        class ToMeVisionTransformer(transformer_class):
            def _reset_tome_state(self):
                self._tome_info["r"] = parse_r(len(self.blocks), self.r)
                self._tome_info["size"] = None

            def forward_features(self, *args, **kwargs):
                self._reset_tome_state()
                return super().forward_features(*args, **kwargs)

            def forward(self, *args, **kwargs):
                self._reset_tome_state()
                return super().forward(*args, **kwargs)

        return ToMeVisionTransformer

    model.__class__ = make_tome_class(model.__class__)
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "prop_attn": prop_attn,
        "class_token": getattr(model, "cls_token", None) is not None,
        "distill_token": getattr(model, "dist_token", None) is not None,
    }

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
            module.fused_attn = False

    return model
