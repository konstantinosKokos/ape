from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from math import ceil, log2
from torch.nn.init import normal_ as normal
from torch.nn.utils.parametrizations import _Orthogonal, parametrize, _OrthMaps

from .schemes import applicative, AtnFn, orthogonal_penalty


class UnitarySequential(Module):
    def __init__(self, dim: int) -> None:
        super(UnitarySequential, self).__init__()
        self.dim = dim
        self.primitives = Parameter(normal(torch.empty(dim, dim)))
        self._orthogonalize()
        self.maps = None

    def forward(self, position_ids: Tensor) -> Tensor:
        return self.maps[position_ids]

    def adjust_attention(self, q_maps: Tensor, k_maps: Tensor, mediator: tuple[Tensor, bool] | None) -> AtnFn:
        return applicative(q_maps, k_maps, mediator=mediator)

    def _orthogonalize(self) -> None:
        # orthogonal_(self.primitives)
        parametrize.register_parametrization(
            module=self,
            tensor_name='primitives',
            parametrization=_Orthogonal(
                weight=self.primitives,
                orthogonal_map=_OrthMaps.matrix_exp,
                use_trivialization=True),
            unsafe=True)

    def penalty(self) -> Tensor:
        return orthogonal_penalty(self.primitives)

    def _expand_maps(self, history: Tensor) -> Tensor:
        longest = history[-1]
        expanded = history @ longest
        return torch.cat((history, expanded), dim=0)

    def _make_maps(self, size: int) -> Tensor:
        maps = self.primitives.unsqueeze(0)
        for _ in range(ceil(log2(size))):
            maps = self._expand_maps(maps)
        maps = maps[:size]
        return torch.cat(
            (torch.eye(self.dim, device=self.primitives.device).unsqueeze(0),
             maps))

    def precompute(self, size: int) -> None:
        self.maps = self._make_maps(size)
