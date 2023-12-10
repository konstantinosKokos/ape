from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from math import ceil, log2

from .schemes import applicative, AtnFn


class UnitarySequential(Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super(UnitarySequential, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self._primitives = Parameter(torch.rand(self.num_heads, self.dim, self.dim).softmax(dim=-1).cumsum(-1))
        self.maps = None

    @property
    def hermitian(self) -> Tensor:
        return self._primitives - self._primitives.mH

    @property
    def primitives(self) -> Tensor:
        hermitian = self.hermitian
        return torch.matrix_exp(hermitian)

    def forward(self, position_ids: Tensor) -> Tensor:
        return self.maps[position_ids]

    def adjust_attention(self, q_maps: Tensor, k_maps: Tensor, mediator: tuple[Tensor, bool] | None) -> AtnFn:
        return applicative(q_maps, k_maps, mediator=mediator)

    def _expand_maps(self, history: Tensor) -> Tensor:
        longest = history[-1]
        expanded = history @ longest
        return torch.cat((history, expanded), dim=0)

    def _make_maps(self, size: int) -> Tensor:
        maps = self.primitives.unsqueeze(0)
        for _ in range(ceil(log2(size))):
            maps = self._expand_maps(maps)
        maps = maps[:size]
        eye = torch.eye(self.dim, device=self.primitives.device)[None].repeat(self.num_heads, 1, 1)
        return torch.cat(
            (eye[None],
             maps))

    def precompute(self, size: int) -> None:
        self.maps = self._make_maps(size)
