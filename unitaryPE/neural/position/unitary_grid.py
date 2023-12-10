from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from math import ceil, log2

from .schemes import grid_applicative, AtnFn


class UnitaryGrid(Module):
    def __init__(self, num_axes: int, dim: int, num_heads: int) -> None:
        super(UnitaryGrid, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_axes = num_axes
        self._primitives = Parameter(
            torch.rand(self.num_axes * self.num_heads, self.dim, self.dim).softmax(dim=-1).cumsum(-1))
        self.maps = None

    @property
    def hermitian(self) -> Tensor:
        return self._primitives - self._primitives.mH

    @property
    def primitives(self) -> Tensor:
        hermitian = self.hermitian
        return torch.matrix_exp(hermitian)

    def forward(self, xs: Tensor, ys: Tensor) -> tuple[Tensor, Tensor]:
        maps_x, maps_y = self.maps.chunk(2, dim=1)
        maps_x = maps_x.squeeze(1)
        maps_y = maps_y.squeeze(1)
        return maps_x[xs], maps_y[ys]

    def adjust_attention(
            self,
            q_maps: tuple[Tensor, Tensor],
            k_maps: tuple[Tensor, Tensor],
            mediator: tuple[Tensor, bool] | None) -> AtnFn:
        return grid_applicative(q_maps, k_maps, mediator=mediator)

    def _expand_maps(self, history: Tensor) -> Tensor:
        longest = history[-1]
        expanded = history @ longest
        return torch.cat((history, expanded), dim=0)

    def _make_maps(self, size: int) -> Tensor:
        maps = self.primitives.unsqueeze(0)
        for _ in range(ceil(log2(size))):
            maps = self._expand_maps(maps)
        maps = maps[:size]
        eye = torch.eye(self.dim, device=self.primitives.device)[None].repeat(self.num_axes * self.num_heads, 1, 1)
        maps = torch.cat((eye[None], maps))
        return maps.view(-1, self.num_axes, self.num_heads, self.dim, self.dim)

    def precompute(self, size: int) -> None:
        self.maps = self._make_maps(size)
