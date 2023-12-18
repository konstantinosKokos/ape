import torch
from torch.nn import Module
from torch import Tensor


def drop_path(x: Tensor, drop_prob: float, training: bool = False):
    if drop_prob == 0. or not training:
        return x
    drop = torch.rand(x.shape[0], device=x.device)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    drop = drop.view(shape)
    mask = drop > drop_prob
    return torch.where(mask, x, 0)


class DropPath(Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)