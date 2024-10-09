from __future__ import annotations


import torch
from torch import Tensor
from torch.nn import Module, Parameter

from .schemes import AtnFn, multihead_atn_fn


class Rotary(Module):
    thetas: Tensor

    def __init__(self, embedding_dim: int, freq: int = 10000, trainable: bool = True) -> None:
        super().__init__()
        self.thetas = Parameter(Rotary.default_angles(freq, embedding_dim), trainable)

    def forward(self, max_seq_len: int) -> Tensor:
        positions = torch.arange(
            start=0,  end=max_seq_len, dtype=torch.long, device=self.thetas.device
        )
        angles = positions.unsqueeze(1) * self.thetas.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=1)

    @staticmethod
    def default_angles(freq: int, dim: int) -> Tensor:
        return 1. / (freq ** (torch.arange(0, dim, 2)[:(dim//2)].float() / dim))

    @staticmethod
    def adjust_attention(sinusoidal_pos: Tensor) -> AtnFn:
        def f(
                queries: Tensor,
                keys: Tensor,
                mask: Tensor) -> Tensor:
            queries, keys = Rotary.apply_rotary_position_embeddings(sinusoidal_pos, queries, keys)
            return multihead_atn_fn(queries, keys, mask, None)
        return f
    
    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos: Tensor, query_layer: Tensor, key_layer: Tensor) -> Tensor:
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
