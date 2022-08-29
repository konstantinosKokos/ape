import pdb

import torch
from torch import Tensor
from .tree import Tree, Node, constant, positionally_encode, descendant_nodes
from typing import Callable


def make_cfn_mtm(mask_on: int, device: torch.device, traversal: Callable[[Tree[Node]], list[Node]]) -> \
        Callable[[list[Tree[tuple[int, int, int]]]], tuple[Tensor, Tensor, Tensor, Tensor]]:
    # collation function for masked tree modeling
    def collate_fn(trees: list[Tree[tuple[int, int, int]]]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        traversed = [traversal(tree) for tree in trees]
        out_ids, in_ids, pos_ids = torch.tensor(traversed).chunk(3, -1)
        out_ids = out_ids.squeeze(-1).to(device)
        in_ids = in_ids.squeeze(-1).to(device)
        pos_ids = pos_ids.squeeze(-1).to(device)
        mask = mask_from_value(out_ids, mask_on)
        return out_ids, in_ids, pos_ids, mask
    return collate_fn


def mask_from_value(node_ids: Tensor, masking_value: int) -> Tensor:
    batch_size, max_len = node_ids.shape
    mask = torch.ones_like(node_ids).unsqueeze(-1).repeat(1, 1, max_len)
    return mask * (node_ids != masking_value).unsqueeze(1)


def mask_from_lens(lens: list[int]) -> Tensor:
    max_len = max(lens)
    ones = torch.ones((len(lens), max_len, max_len), dtype=torch.long)
    for i, l in enumerate(lens):
        ones[i, :, l:] = 0
    return ones


def dfs_mask(max_depth: int, traversal: Callable[[Tree[Node]], list[Node]]) -> Tensor:
    """
    given a tree of n nodes, create a (n x n) tensor M where M[i, j] = 1 if node i is a descendant of node j
    """
    tree_positions = positionally_encode(constant(max_depth, None))
    mask = [[other in descendants for other in traversal(tree_positions)]
            for descendants in traversal(descendant_nodes(tree_positions))]
    return torch.tensor(mask)


def bfs_mask(max_depth: int, traversal: Callable[[Tree[Node]], list[Node]]) -> Tensor:
    """
    given a tree of n nodes, create a (n x n) tensor M where M[i, j] = 1 if node i occurs above/to the right of node j
    """
    tree_positions = positionally_encode(constant(max_depth, None))
    mask = [[tp > other for other in traversal(tree_positions)]
            for tp in traversal(tree_positions)]
    return torch.tensor(mask)


def depth_parallel_mask(max_depth: int, traversal: Callable[[Tree[Node]], list[Node]]) -> Tensor:
    """
    given a tree of n nodes, create a (n x n) tensor M where M[i, j] = 1 if node i occurs above node j
    """
    tree_positions = positionally_encode(constant(max_depth, None))
    mask = [[tp//2 >= other for other in traversal(tree_positions)]
            for tp in traversal(tree_positions)]
    return torch.tensor(mask)


def accuracy(preds: Tensor, truths: Tensor) -> tuple[int, int]:
    return (preds == truths).sum().item(), len(truths)
