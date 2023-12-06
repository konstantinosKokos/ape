from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from math import ceil, log2
from .schemes import applicative, AtnFn


def make_angle_matrix(dim: int, base: int = 10000) -> Tensor:
    angles = torch.arange(start=0, end=dim/2, step=1)/dim
    angles = base ** (-2 * angles)
    cos = torch.cos(angles)
    cos = torch.repeat_interleave(cos, repeats=2)
    sin = torch.sin(angles)
    sin = torch.repeat_interleave(sin, 2)
    sin[torch.arange(len(sin + 1)) % 2 == 1] = 0
    r = torch.zeros(dim, dim)
    r = r.diagonal_scatter(cos)
    r = r.diagonal_scatter(sin[:-1], offset=-1)
    r = r.diagonal_scatter(-sin[:-1], offset=1)
    return r


class Rotary(Module):
    def __init__(self, dim: int) -> None:
        super(Rotary, self).__init__()
        self.dim = dim
        self.primitives = Parameter(make_angle_matrix(dim), requires_grad=False)
        self.maps = None

    def forward(self, position_ids: Tensor) -> Tensor:
        return self.maps[position_ids][None]

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
        eye = torch.eye(self.dim, device=self.primitives.device)
        return torch.cat(
            (eye,
             maps))

    def precompute(self, size: int) -> None:
        self.maps = self._make_maps(size)
