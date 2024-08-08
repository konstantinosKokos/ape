from __future__ import annotations


import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import embedding

from .schemes import AtnFn, multihead_atn_fn


class Rotary(Module):
    weight: Tensor

    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = Parameter(self._init_weight(num_positions, embedding_dim))

    @staticmethod
    def _init_weight(n_pos: int, dim: int) -> Tensor:
        out = torch.zeros(n_pos, dim, dtype=torch.float)
        position_enc = torch.tensor(
            [[pos / (10000 ** (2 * (j // 2) / dim)) for j in range(dim)] for pos in range(n_pos)]
        )
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.sin(position_enc[:, 0::2])
        out[:, sentinel:] = torch.cos(position_enc[:, 1::2])
        return out

    @torch.no_grad()
    def forward(self, max_seq_len: int) -> Tensor:
        positions = torch.arange(
            start=0,  end=max_seq_len, dtype=torch.long, device=self.weight.device
        )
        return embedding(positions, self.weight)
    
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
