from __future__ import annotations

from torch.nn import Module, Parameter, Embedding, Identity
from torch.nn.functional import linear
from torch import Tensor
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.parametrizations import orthogonal

from operator import add, mul


class PositionalEncoder(Module):
    def __init__(self, core: Module, is_gate: bool = False):
        super(PositionalEncoder, self).__init__()
        self.op = mul if is_gate else add
        self.core = core

    def forward(self, position_ids: Tensor, embeddings: Tensor, compute: bool = True) -> Tensor:
        position_embeddings = self.core(position_ids, compute).to(embeddings.device)
        return self.op(position_embeddings, embeddings)

    @staticmethod
    def sinusoidal_flat(dim: int, is_gate: bool, freq: int = 100) -> PositionalEncoder:
        return PositionalEncoder(SinusoidalFlat(dim, freq), is_gate)

    @staticmethod
    def tree_unitary(dim: int, is_gate: bool) -> PositionalEncoder:
        return PositionalEncoder(TreeUnitary.orthogonal(dim), is_gate)

    @staticmethod
    def fixed(num_positions: int, dim: int, padding_idx: int, is_gate: bool) -> PositionalEncoder:
        return PositionalEncoder(Embedding(num_positions, dim, padding_idx=padding_idx), is_gate)

    @staticmethod
    def identity() -> PositionalEncoder:
        return PositionalEncoder(Identity())


class SinusoidalFlat(Module):
    def __init__(self, dim: int, freq: int = 10000):
        super(SinusoidalFlat, self).__init__()
        self.dim = dim
        self.freq = freq
        self.precomputed = None

    def forward(self, position_ids: Tensor, _: bool):
        (batch_size, max_len) = position_ids.shape[:2]
        if self.precomputed is None or max_len > self.precomputed.shape[1]:
            self.precomputed = self.precompute(max_len)
        return self.precomputed.unsqueeze(0).repeat(batch_size, 1, 1)

    def precompute(self, n: int) -> Tensor:
        pe = torch.empty(n, self.dim, dtype=torch.float)
        positions = torch.arange(0, n).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float) *
                             - (torch.log(torch.tensor(self.freq, dtype=torch.float)) / self.dim))
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe


class TreeUnitary(Module):
    def __init__(self, dim: int):
        super(TreeUnitary, self).__init__()
        self.primitives = Parameter(torch.nn.init.normal_(torch.empty(2, dim, dim)))
        self.init = Parameter(torch.nn.init.normal_(torch.empty(dim)))
        self.dim = dim
        self.precomputed = None

    def forward(self, node_positions: Tensor, compute: bool) -> Tensor:
        (batch_size, max_len) = node_positions.shape[:2]
        if self.precomputed is None or max_len > self.precomputed.shape[1] or compute:
            self.precomputed = self.compute_range(node_positions.max())
        position_embeddings = self.precomputed
        return position_embeddings[(node_positions - 1).flatten()].view(batch_size, max_len, self.dim)

    def compute_range(self, up_to: int) -> Tensor:
        words = [torch.tensor(self.node_pos_to_path(i), device=self.primitives.device, dtype=torch.long)
                 for i in range(1, up_to + 2)]  # right inclusive, offset by 1
        words = pad_sequence(words, padding_value=2, batch_first=True)
        seeds = self.init.data.repeat(words.shape[0], 1)
        for step in range(words.shape[1]):
            seeds[words[:, step] == 0] = linear(seeds[words[:, step] == 0], self.primitives[0])
            seeds[words[:, step] == 1] = linear(seeds[words[:, step] == 1], self.primitives[1])
        return seeds

    @staticmethod
    def node_pos_to_path(idx: int) -> list[int]:
        if idx == 1:
            return []
        return [idx % 2] + TreeUnitary.node_pos_to_path(idx // 2)

    @staticmethod
    def orthogonal(dim: int) -> TreeUnitary:
        return orthogonal(TreeUnitary(dim), name='primitives', orthogonal_map='matrix_exp')  # type: ignore
