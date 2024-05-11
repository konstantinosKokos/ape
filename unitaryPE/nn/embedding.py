import torch
from torch.nn import Module, Parameter
from torch.nn.functional import embedding, linear
from torch import Tensor
from math import sqrt


class InvertibleEmbedding(Module):
    def __init__(self, num_classes: int, dim: int):
        super(InvertibleEmbedding, self).__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.weight = Parameter(torch.nn.init.normal_(torch.empty(num_classes, dim)), requires_grad=True)

    def embed(self, xs: Tensor) -> Tensor:
        return embedding(xs.clamp(min=0), self.weight, scale_grad_by_freq=True) * sqrt(self.dim)

    def invert(self, xs: Tensor) -> Tensor:
        return linear(xs, self.weight)
