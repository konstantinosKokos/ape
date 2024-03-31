from torch.nn import Module, Embedding
from torch import Tensor


class Absolute(Module):
    def __init__(self, dim: int, num_positions: int):
        super(Absolute, self).__init__()
        self.embedding = Embedding(num_positions, dim)
        self.max_size = num_positions

    def forward(self, positions: Tensor) -> Tensor:
        return self.embedding(positions)
