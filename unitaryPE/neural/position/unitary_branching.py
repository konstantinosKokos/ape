from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.utils.rnn import pad_sequence

from typing import NoReturn

from .schemes import applicative, AtnFn


class UnitaryBranching(Module):
    def __init__(self, dim: int, branching_factor: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.branching_factor = branching_factor
        self._primitives = Parameter(torch.rand(self.branching_factor * self.num_heads + 1, self.dim, self.dim))
        self.maps = None
        self._pos_to_path = {1: [], 0: [-1], -1: []}

    @property
    def hermitian(self) -> Tensor:
        return self._primitives - self._primitives.mH

    @property
    def primitives(self) -> Tensor:
        hermitian = self.hermitian/self.dim
        return torch.matrix_exp(hermitian)

    def forward(self, mapping: Tensor) -> NoReturn:
        raise NotImplementedError('You have to index the precomputed maps by hand')

    def adjust_attention(self, q_maps: Tensor, k_maps: Tensor, mediator: tuple[Tensor, bool] | None) -> AtnFn:
        return applicative(q_maps, k_maps, mediator=mediator)

    def precompute(self, positions: list[int]) -> None:
        with torch.no_grad():
            path_words = pad_sequence(
                sequences=[
                    torch.tensor(self.pos_to_path(pos), device=self.primitives.device, dtype=torch.short)
                    # if pos > 0 else torch.empty(0, device=self.primitives.device, dtype=torch.long)
                    for pos in positions],
                batch_first=True,
                padding_value=self.branching_factor + 1
            )
            point_mask = path_words.ne(self.branching_factor + 1)
            mask = point_mask[:, None] & point_mask[None]
            pointwise_equal = path_words[:, None].eq(path_words[None])
            common_prefix = pointwise_equal.cumprod(-1).logical_and(mask)
            max_lens = torch.max(point_mask.sum(-1)[:, None], point_mask.sum(-1)[None])
            cpl = common_prefix.sum(-1)
            steps = max_lens - cpl
        primitives = self.primitives
        sos_repr = primitives[-1:]
        primitives = primitives[:-1].unflatten(0, (self.branching_factor, self.num_heads))
        maps = torch.eye(n=self.dim, device=primitives.device).view(1, 1, self.dim, self.dim)
        maps = maps.repeat(len(path_words), self.num_heads, 1, 1)
        sos_ptr = path_words[:, 0] == -1
        maps[sos_ptr] = sos_repr
        for depth in range(path_words.shape[1]):
            for k in range(self.branching_factor):
                mask = path_words[:, depth] == k
                maps[mask] = maps[mask] @ primitives[k]
        self.maps = (maps, steps)

    def pos_to_path(self, idx: int) -> list[int]:
        if idx in self._pos_to_path:
            return self._pos_to_path[idx]
        parent = (idx - 2) // self.branching_factor + 1 if idx != 1 else -1
        step = (idx - 2) % self.branching_factor + 1 if idx != 1 else 0
        self._pos_to_path[idx] = [*self.pos_to_path(parent), step]
        return self._pos_to_path[idx]
