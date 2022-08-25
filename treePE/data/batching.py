import pdb

import torch
from torch import Tensor
from .tree import Tree, Node, breadth_first
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


def accuracy(preds: Tensor, truths: Tensor) -> tuple[int, int]:
    return (preds == truths).sum().item(), len(truths)
