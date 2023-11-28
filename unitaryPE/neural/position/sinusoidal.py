import torch
from torch.nn import Module
from torch import Tensor


class SinusoidalFlat(Module):
    def __init__(self, dim: int, freq: int = 10000):
        super(SinusoidalFlat, self).__init__()
        self.dim = dim
        self.freq = freq
        self.precomputed = None

    def forward(self, position_ids: Tensor):
        (batch_size, max_len) = position_ids.shape[:2]
        if self.precomputed is None or max_len > self.precomputed.shape[1]:
            self.precomputed = self.precompute(max_len)
        return self.precomputed.unsqueeze(0).to(position_ids.device)

    def precompute(self, n: int) -> Tensor:
        pe = torch.empty(n, self.dim, dtype=torch.float)
        positions = torch.arange(0, n).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float) *
                             - (torch.log(torch.tensor(self.freq, dtype=torch.float)) / self.dim))
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe