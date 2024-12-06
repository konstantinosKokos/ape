import torch
from torch.nn import Module
from torch import Tensor
from torch.nn.functional import embedding


class SinusoidalFlat(Module):
    precomputed: Tensor

    def __init__(self, dim: int, max_seq_len: int, freq: int = 10000):
        super(SinusoidalFlat, self).__init__()
        self.dim = dim
        self.freq = freq
        self.register_buffer('precomputed', self._precompute(max_seq_len), persistent=False)

    def forward(self, position_ids: Tensor):
        return embedding(position_ids, self.precomputed)

    def _precompute(self, n: int) -> Tensor:
        pe = torch.empty(n, self.dim, dtype=torch.float)
        positions = torch.arange(0, n).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float) *
                             - (torch.log(torch.tensor(self.freq, dtype=torch.float)) / self.dim))
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe


class SinusoidalGrid(Module):
    def __init__(self, dim: int, freq: int = 10000):
        super(SinusoidalGrid, self).__init__()
        self.dim = dim
        self.freq = freq
        self.precomputed = None

    def forward(self, position_ids: Tensor):
        (batch_size, max_len) = position_ids.shape[:2]
        if self.precomputed is None or max_len > self.precomputed.shape[1]:
            self.precomputed = self.precompute(max_len)
        return self.precomputed.unsqueeze(0).to(position_ids.device)

    def precompute(self, n: int) -> Tensor:
        pe = torch.tensor([[p / (self.freq ** (2 * (i // 2) / self.dim)) for i in range(self.dim)]
                           for p in range(n)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe
