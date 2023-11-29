from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.init import normal_ as normal
from torch.nn.utils.parametrizations import _Orthogonal, parametrize, _OrthMaps
from torch.nn.utils.rnn import pad_sequence

from .schemes import applicative, AtnFn


class UnitaryBranching(Module):
    def __init__(self, dim: int, branching_factor: int):
        super().__init__()
        self.primitives: Parameter = Parameter(normal(torch.empty(branching_factor, dim, dim)))
        self.identity: Parameter = Parameter(torch.eye(dim).unsqueeze(0))
        self._pos_to_path: dict[int, list[int]] = {1: []}
        self.dim: int = dim
        self.branching_factor: int = branching_factor

    def embed_positions(self, positions: list[int]) -> Tensor:
        word_seq = [torch.tensor(self.pos_to_path(pos), device=self.primitives.device, dtype=torch.long)
                    if pos > 0 else torch.empty(0, device=self.primitives.device, dtype=torch.long)
                    for pos in positions]
        word_ten = pad_sequence(word_seq, padding_value=self.branching_factor + 1, batch_first=True)
        maps = self.identity.repeat(len(positions), 1, 1)
        for depth in range(word_ten.shape[1]):
            for k in range(self.branching_factor):
                mask = word_ten[:, depth] == k
                maps[mask] = maps[mask] @ self.primitives[k]
        return maps

    def forward(self, unique: Tensor) -> Tensor:
        return self.embed_positions(unique.cpu().tolist())

    def adjust_attention(self, q_maps: Tensor, k_maps: Tensor, mediator: tuple[Tensor, bool] | None) -> AtnFn:
        return applicative(q_maps, k_maps, mediator=mediator)

    def revert_mapping(self, embeddings: Tensor, mapping: Tensor) -> Tensor:
        indices = torch.ravel(mapping)
        return torch.index_select(input=embeddings, dim=0, index=indices).view(*mapping.shape, self.dim)

    def pos_to_path(self, idx: int) -> list[int]:
        if idx in self._pos_to_path:
            return self._pos_to_path[idx]
        parent = (idx - 2) // self.branching_factor + 1 if idx != 1 else -1
        step = (idx - 2) % self.branching_factor + 1 if idx != 1 else 0
        self._pos_to_path[idx] = [*self._pos_to_path[parent], step]
        return self._pos_to_path[idx]

    def _orthogonalize(self) -> None:
        parametrize.register_parametrization(
            module=self,
            tensor_name='primitives',
            parametrization=_Orthogonal(
                weight=self.primitives,
                orthogonal_map=_OrthMaps.matrix_exp,
                use_trivialization=True),
            unsafe=True)