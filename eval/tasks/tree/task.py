from __future__ import annotations

from .abstract import Tree, Node, Leaf, Binary, bf_enum, df_enum, breadth_first, depth_first

import random
from dataclasses import dataclass
from abc import ABC, abstractmethod, ABCMeta
from typing import Generic, Callable, Literal


import torch
from torch.distributions import Distribution
from torch import Size


def make_processor(enc_traversal: Literal['breadth', 'depth'],
                   dec_traversal: Literal['breadth', 'depth']):
    match enc_traversal:
        case 'breadth': enc_traversal_fn = breadth_first
        case 'depth': enc_traversal_fn = depth_first
        case _: raise ValueError
    match dec_traversal:
        case 'breadth':
            causal_mask_fn = make_mask_fn(enumeration_fn=bf_enum)
            dec_traversal_fn = breadth_first
        case 'depth':
            causal_mask_fn = make_mask_fn(enumeration_fn=df_enum)
            dec_traversal_fn = depth_first
        case _: raise ValueError

    def process_sample(
            input_tree: Tree[Node], output_tree: Tree[Node]) -> tuple[tuple[list[Node], list[int]],
                                                                      tuple[list[Node], list[int]],
                                                                      list[list[bool]]]:
        input_nodes, input_pos = zip(*enc_traversal_fn(input_tree.zip(bf_enum(input_tree))))
        output_nodes, causal_mask = causal_mask_fn(output_tree.zip(bf_enum(output_tree)), dec_traversal_fn)
        output_nodes, output_pos = zip(*output_nodes)
        return (list(input_nodes), list(input_pos)), (list(output_nodes), list(output_pos)), causal_mask
    return process_sample


def make_mask_fn(
        enumeration_fn: Callable[[Tree[Node]], Tree[int]],
        self_attend: bool = True) \
        -> Callable[[Tree[Node], Callable[[Tree[Node]], list[Node]]], tuple[list[Node], list[list[bool]]]]:
    def mask_fn(tree: Tree[Node], traversal: Callable[[Tree[Node]], list[Node]]) -> tuple[list[Node], list[list[bool]]]:
        nodes, traversed_positions = zip(*traversal(tree.zip(enumeration_fn(tree))))
        node_contexts = [list(range(1, i + self_attend)) for i in traversed_positions]
        mask = [[other in node_ctx for other in traversed_positions] for node_ctx in node_contexts]
        return nodes, mask
    return mask_fn


class TreeGenerator(Generic[Node]):
    def __init__(self, leaves: set[Node], operators: set[Node]):
        self.leaves = set(leaves)
        self.operators = set(operators)

    def random_tree(self, depth: int, unique_nodes: bool = False) -> Tree[Node]:
        if unique_nodes:
            return random_tree_unique(depth, list(self.leaves), list(self.operators))[0]
        return random_tree(depth, list(self.leaves), list(self.operators))


def random_tree(depth: int, leaves: list[Node], operators: list[Node]) -> Tree[Node]:
    if depth == 0:
        return Leaf(random.choice(leaves))

    random_depth = random.randint(0, depth - 1)
    fixed_depth = depth - 1 if random_depth < depth - 1 else random.randint(0, depth - 1)
    ldepth, rdepth = random.choice([(random_depth, fixed_depth), (fixed_depth, random_depth)])
    return Binary(
        random.choice(operators),
        random_tree(ldepth, leaves, operators),
        random_tree(rdepth, leaves, operators))


def random_tree_unique(
        depth: int,
        leaves: list[Node],
        operators: list[Node]) -> tuple[Tree[Node], list[Node], list[Node]]:
    if depth == 0:
        leaves.remove(leaf := random.choice(leaves))
        return Leaf(leaf), leaves, operators

    random_depth = random.randint(0, depth - 1)
    fixed_depth = depth - 1 if random_depth < depth - 1 else random.randint(0, depth - 1)
    ldepth, rdepth = random.choice([(random_depth, fixed_depth), (fixed_depth, random_depth)])
    operators.remove(op := random.choice(operators))
    left, leaves, operators = random_tree_unique(ldepth, leaves, operators)
    right, leaves, operators = random_tree_unique(rdepth, leaves, operators)
    return Binary(op, left, right), leaves, operators


@dataclass
class TreeTask(ABC, metaclass=ABCMeta):
    x_projection: Literal['depth', 'breadth']
    y_projection: Literal['depth', 'breadth']
    sos_token_id: int
    eos_token_id: int

    def __post_init__(self):
        self.preprocess = make_processor(self.x_projection, self.y_projection)

    def preprocess(self, x: Tree[int], y: Tree[int]) -> tuple[tuple[list[int], list[int]],
                                                              tuple[list[int], list[int]],
                                                              list[list[bool]]]:
        # just for type hinting purposes
        ...

    def process(self, x: Tree[int], y: Tree[int]) -> tuple[tuple[list[int], list[int]],
                                                           tuple[list[int], list[int]],
                                                           list[list[bool]]]:
        (input_nodes, input_pos), (output_nodes, output_pos), causal_mask = self.preprocess(x, y)
        input_nodes = [self.sos_token_id] + input_nodes
        input_pos = [self.sos_token_id] + input_pos
        output_nodes = [self.sos_token_id] + output_nodes
        output_pos = [self.sos_token_id] + output_pos
        causal_mask = [[True] + row for row in causal_mask]
        sos_row = [True] + [False] * len(causal_mask)
        return (input_nodes, input_pos), (output_nodes, output_pos), [sos_row] + causal_mask

    @abstractmethod
    def sample(self, depth: int) -> TreeSample:
        ...
    
    def sample_many(self, depths: list[int]) -> list[TreeSample]:
        return [self.sample(length) for length in depths]

    def sample_depth(self, distribution: Distribution, num_samples: int) -> list[TreeSample]:
        return self.sample_many(distribution.sample(Size((num_samples,))).long().tolist())

    def make_sets(
            self,
            distributions: tuple[Distribution, ...],
            num_samples: tuple[int, ...],
            seed: int) -> tuple[list[TreeSample], ...]:
        torch.manual_seed(seed)
        random.seed(seed)
        return tuple(self.sample_depth(distribution, ns) for distribution, ns in zip(distributions, num_samples))


@dataclass(frozen=True)
class TreeSample:
    x: Tree[int]
    y: Tree[int]
    task: TreeTask

    def process(self) -> tuple[tuple[list[Node], list[int]],
                               tuple[list[Node], list[int]],
                               list[list[bool]]]:
        return self.task.process(self.x, self.y)
    
    def __hash__(self) -> int:
        return hash((self.x, self.y, self.task))
