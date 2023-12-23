"""
    Adapted from https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/roformer/modeling_roformer.p
"""

from __future__ import annotations


import torch
from torch import Tensor
from torch.nn import Embedding, Parameter

from .schemes import AtnFn, multihead_atn_fn


class Rotary(Embedding):
    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: Parameter) -> Parameter:
        n_pos, dim = out.shape
        position_enc = torch.tensor(
            [[pos / (10000 ** (2 * (j // 2) / dim)) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.sin(position_enc[:, 0::2])
        out[:, sentinel:] = torch.cos(position_enc[:, 1::2])
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, max_seq_len: int) -> Tensor:
        positions = torch.arange(
            start=0,  end=max_seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)
    
    def adjust_attention(self, sinusoidal_pos: Tensor) -> AtnFn:
        def f(
                queries: Tensor,
                keys: Tensor,
                mask: Tensor) -> Tensor:
            queries, keys = self.apply_rotary_position_embeddings(sinusoidal_pos, queries, keys)
            return multihead_atn_fn(queries, keys, mask, None)
        return f
    
    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer):
        num_qs = query_layer.shape[1]
        num_ks = key_layer.shape[1]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
            query_layer
        )
        query_layer = query_layer * cos_pos[None, :num_qs, :, None] + rotate_half_query_layer * sin_pos[None, :num_qs, :, None]
        rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
        key_layer = key_layer * cos_pos[None, :num_ks, :, None] + rotate_half_key_layer * sin_pos[None, :num_ks, :, None]
        return query_layer, key_layer
