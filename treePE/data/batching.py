import pdb

import torch
from torch import Tensor
from .tree import Tree, Node, constant, positionally_encode
from typing import Callable, Literal
from math import log2, floor


def make_cfn_mtm(mask_on: int, device: torch.device, traversal: Callable[[Tree[Node]], list[Node]]) -> \
        Callable[[list[Tree[tuple[int, int, int]]]], tuple[Tensor, Tensor, Tensor, Tensor]]:
    # collation function for masked tree modeling
    def collate_fn(trees: list[Tree[tuple[int, int, int]]]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        traversed = [traversal(tree) for tree in trees]
        out_ids, in_ids, pos_ids = torch.tensor(traversed).chunk(3, -1)
        out_ids = out_ids.squeeze(-1).to(device)
        in_ids = in_ids.squeeze(-1).to(device)
        pos_ids = pos_ids.squeeze(-1).to(device)
        mask = mask_from_value(out_ids, mask_on, out_ids.shape[1])
        return out_ids, in_ids, pos_ids, mask
    return collate_fn


def make_cfn_tree2tree(enc_mask_on: int,
                       dec_mask_on: int,
                       device: torch.device,
                       enc_traversal: Callable[[Tree[Node]], list[Node]],
                       dec_traversal: Callable[[Tree[Node]], list[Node]],
                       regression: Literal['breadth', 'ancestor'],
                       mask_idx: int):
    # collation function for tree2tree decoding
    match regression:
        case 'breadth': cmask_fn = breadth_first_mask
        case 'ancestor': cmask_fn = ancestor_mask
        case _: raise ValueError(f'{regression} not supported')

    def collate_fn(trees: list[tuple[Tree[tuple[int, int]], Tree[tuple[int, int]]]]) -> \
            tuple[Tensor, Tensor, Tensor, list[Tensor], Tensor, list[Tensor], Tensor, Tensor]:
        enc_trees, dec_trees = zip(*trees)
        enc_traversed = [enc_traversal(tree) for tree in enc_trees]

        enc_ids, enc_pos = torch.tensor(enc_traversed).chunk(2, -1)
        enc_ids = enc_ids.squeeze(-1).to(device)
        enc_pos = enc_pos.squeeze(-1).to(device)

        dec_traversed = [dec_traversal(tree) for tree in dec_trees]
        dec_ids, dec_pos = torch.tensor(dec_traversed).chunk(2, -1)
        dec_ids = dec_ids.squeeze(-1).to(device)
        dec_pos = dec_pos.squeeze(-1).to(device)

        enc_mask = mask_from_value(enc_ids, enc_mask_on, enc_ids.shape[1])
        pad_mask = mask_from_value(dec_ids, dec_mask_on, dec_ids.shape[1])
        cross_mask = mask_from_value(enc_ids, enc_mask_on, dec_ids.shape[1])

        causal_masks, pred_masks = cmask_fn(max(tree.depth() for tree in dec_trees), dec_traversal)
        causal_masks = [cm.unsqueeze(0).repeat(dec_ids.shape[0], 1, 1).to(device) for cm in causal_masks]
        pred_masks = [pm.unsqueeze(0).repeat(dec_ids.shape[0], 1).to(device) * pad_mask[:, 0] for pm in pred_masks]
        masked_ids = [torch.where(pm == 1, mask_idx, dec_ids) for pm in pred_masks]

        return enc_ids, enc_pos, enc_mask, masked_ids, dec_pos, causal_masks, cross_mask, dec_ids
    return collate_fn


def mask_from_value(node_ids: Tensor, masking_value: int, max_len: int) -> Tensor:
    mask = torch.ones_like(node_ids).unsqueeze(-2).repeat(1, max_len, 1)
    return mask * (node_ids != masking_value).unsqueeze(1)


def mask_from_lens(lens: list[int]) -> Tensor:
    max_len = max(lens)
    ones = torch.ones((len(lens), max_len, max_len), dtype=torch.long)
    for i, l in enumerate(lens):
        ones[i, :, l:] = 0
    return ones


def offset(tensor: Tensor) -> Tensor:
    ones = torch.ones(tensor.shape[0], tensor.shape[1], 1, dtype=torch.bool, device=tensor.device)
    return torch.cat((ones, tensor), dim=-1)


def _depth(i: int) -> int: return floor(log2(i))
def _ancestors(i: int) -> list[int]: return [] if i == 0 else [f := i // 2] + _ancestors(f)
def _breadth_ancestors(i: int) -> list[int]: return list(range(i))


def ancestor_mask(max_depth: int, traversal: Callable[[Tree[Node]], list[Node]]) \
        -> tuple[list[Tensor], list[Tensor]]:
    nodes = traversal(tree_positions := positionally_encode(constant(max_depth, None)))
    ancestry = traversal(tree_positions.fmap(_ancestors).zip(tree_positions))

    node_depths = torch.tensor([_depth(node) for node in nodes]).unsqueeze(1).repeat(1, len(nodes))
    causal_mask = torch.tensor([[other in ancestors or other == node for other in nodes]
                                for ancestors, node in ancestry])
    false_mask = torch.zeros_like(causal_mask, dtype=torch.bool)

    step_masks = [torch.where(node_depths <= depth, causal_mask, false_mask) for depth in range(max_depth + 1)]
    pred_masks = [node_depths[:, 0] == depth for depth in range(max_depth + 1)]
    return step_masks, pred_masks


def depth_first_mask(max_depth: int, traversal: Callable[[Tree[Node]], list[Node]]) \
        -> tuple[list[Tensor], list[Tensor]]:
    raise NotImplementedError


def breadth_first_mask(max_depth: int, traversal: Callable[[Tree[Node]], list[Node]]) \
        -> tuple[list[Tensor], list[Tensor]]:
    nodes = traversal(tree_positions := positionally_encode(constant(max_depth, None)))
    ancestry = traversal(tree_positions.fmap(_breadth_ancestors).zip(tree_positions))

    node_ids = torch.tensor(nodes).unsqueeze(1).repeat(1, len(nodes))
    causal_mask = torch.tensor([[other in ancestors or other == ancestors for other in nodes]
                                for ancestors, node in ancestry])
    false_mask = torch.zeros_like(causal_mask, dtype=torch.bool)

    step_masks = [torch.where(node_ids <= step, causal_mask, false_mask) for step in range(1, len(nodes) + 1)]
    pred_masks = [node_ids[:, 0] == step + 1 for step in range(len(nodes))]
    return step_masks, pred_masks


def depth_parallel_mask(max_depth: int, traversal: Callable[[Tree[Node]], list[Node]]) \
        -> tuple[list[Tensor], list[Tensor]]:
    raise NotImplementedError



def accuracy(preds: Tensor, truths: Tensor) -> tuple[int, int]:
    return (preds == truths).sum().item(), len(truths)
