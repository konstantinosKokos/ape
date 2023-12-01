from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.init import normal_ as normal
from torch.nn.utils.parametrizations import _Orthogonal, parametrize, _OrthMaps
from torch.nn.utils.rnn import pad_sequence

from .schemes import applicative, AtnFn


class UnitaryBranching(Module):
    def __init__(self, dim: int, branching_factor: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.branching_factor = branching_factor
        self.primitives: Parameter = Parameter(normal(torch.empty(branching_factor, dim, dim)))
        self._orthogonalize()
        self.maps = None
        self._pos_to_path = {1: [], -1: []}

    def forward(self, mapping: Tensor) -> Tensor:
        indices = torch.ravel(mapping)
        maps = torch.index_select(input=self.maps, dim=0, index=indices)
        return maps.view(*mapping.shape, self.num_heads, self.dim, self.dim)

    def adjust_attention(self, q_maps: Tensor, k_maps: Tensor, mediator: tuple[Tensor, bool] | None) -> AtnFn:
        return applicative(q_maps, k_maps, mediator=mediator)

    def _orthogonalize(self) -> None:
        parametrize.register_parametrization(
            module=self,
            tensor_name='primitives',
            parametrization=_Orthogonal(
                weight=self.primitives,
                orthogonal_map=_OrthMaps.matrix_exp,
                use_trivialization=True),
            unsafe=True)

    def precompute(self, positions: list[int]) -> None:
        path_words = pad_sequence(
            sequences=[
                torch.tensor(self.pos_to_path(pos), device=self.primitives.device, dtype=torch.long)
                if pos > 0 else torch.empty(0, device=self.primitives.device, dtype=torch.long)
                for pos in positions],
            batch_first=True,
            padding_value=self.branching_factor + 1
        )
        maps = torch.eye(n=self.dim, device=self.primitives.device).view(1, 1, self.dim, self.dim)
        maps = maps.repeat(len(path_words), self.num_heads, 1, 1)
        for depth in range(path_words.shape[1]):
            for k in range(self.branching_factor):
                mask = path_words[:, depth] == k
                maps[mask] = maps[mask] @ self.primitives[k]
        self.maps = maps

    def pos_to_path(self, idx: int) -> list[int]:
        if idx in self._pos_to_path:
            return self._pos_to_path[idx]
        parent = (idx - 2) // self.branching_factor + 1 if idx != 1 else -1
        step = (idx - 2) % self.branching_factor + 1 if idx != 1 else 0
        self._pos_to_path[idx] = [*self._pos_to_path[parent], step]
        return self._pos_to_path[idx]
