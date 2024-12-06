from torch.nn import Module, Embedding
from torch import Tensor
from .schemes import additive_mediator, AtnFn


class Relative(Module):
    def __init__(self, dim: int, window_size: int):
        super(Relative, self).__init__()
        self.dist_embedding = Embedding(2 * window_size + 1, dim)
        self.window_size = window_size

    def forward(self, distances: Tensor) -> Tensor:
        distances = distances.clamp(min=-self.window_size, max=self.window_size)
        distances = distances + self.window_size
        return self.dist_embedding(distances)

    def adjust_attention(self, qk_pos: Tensor) -> AtnFn:
        return additive_mediator(qk_pos)
