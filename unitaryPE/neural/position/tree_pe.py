"""
    Adapted from https://github.com/AwdHanPeng/TreeTransformer/blob/master/cc/main/src/utils/encodings_utils.py
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.init import uniform_


def create_paths(max_depth: int, degree: int) -> Tensor:
    paths = [(0, torch.zeros(max_depth * degree))]
    onehots = torch.eye(degree)
    i = 0
    while i < len(paths):
        depth, path = paths[i]
        if depth < max_depth:
            for j in range(degree):
                new_path = (depth + 1, torch.cat([onehots[j], path[:-degree]]))
                paths.append(new_path)
        i += 1
    return torch.stack([p for _, p in paths])


class TreePE(Module):
    def __init__(self, dim: int, max_depth: int, branching_factor: int):
        super(TreePE, self).__init__()
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.dim = dim
        self.weight = Parameter(uniform_(torch.empty(dim, dtype=torch.float32), 0.7, 0.999), requires_grad=False)
        self.paths = create_paths(self.max_depth, self.branching_factor)

    def forward(self, positions: Tensor) -> Tensor:
        positions = (positions - 1).clamp(min=0)
        path_words = self.paths[positions]
        weight = self.weight.tanh()
        depth = torch.arange(self.max_depth, dtype=torch.float, device=weight.device)
        scale = torch.sqrt((1 - weight ** 2) * self.dim / 2)
        weight = (weight[None] ** depth[:, None] * scale)
        weight = weight.repeat(2, 1)
        weight = path_words * weight
        return weight.flatten(-2)
