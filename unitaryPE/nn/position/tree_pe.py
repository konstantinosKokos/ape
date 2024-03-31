"""
    Adapted from https://github.com/AwdHanPeng/TreeTransformer/blob/master/cc/main/src/utils/encodings_utils.py
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.init import uniform_


def create_paths(
        word_length: int,
        max_depth: int,
        branching_factor: int) -> Tensor:
    paths = [torch.zeros(word_length)]
    steps = torch.eye(branching_factor)
    for node_idx in range(1, branching_factor ** max_depth):
        root_idx = (node_idx - 1) // branching_factor
        branch_idx = (node_idx - 1) % branching_factor
        step = steps[branch_idx]
        path = torch.cat((step, paths[root_idx][:-branching_factor]))
        paths.append(path)
    return torch.stack(paths)


class TreePE(Module):
    def __init__(self, dim: int, max_depth: int, branching_factor: int):
        super(TreePE, self).__init__()
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.dim = dim
        self.weight = Parameter(uniform_(torch.empty(dim, dtype=torch.float32), 0.7, 0.999), requires_grad=True)
        self.paths = Parameter(create_paths(
            word_length=self.max_depth * self.branching_factor,
            branching_factor=self.branching_factor,
            max_depth=2 * self.max_depth), requires_grad=False)

    def forward(self, positions: Tensor) -> Tensor:
        positions = (positions - 1).clamp(min=0)
        path_words = self.paths[positions]
        weight = self.weight.tanh()
        depth = torch.arange(self.max_depth, dtype=torch.float, device=weight.device)
        scale = torch.sqrt((1 - weight ** 2) * self.dim / 2)
        weight = (weight[None] ** depth[:, None] * scale)
        weight = weight.repeat(2, 1)
        weight = (path_words[:, :, :, None] * weight[None, None])
        return weight.flatten(-2)
