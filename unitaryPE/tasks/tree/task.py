from __future__ import annotations

import random
from dataclasses import dataclass
from abc import ABC, abstractmethod, ABCMeta
from typing import Generic, Iterator
from .abstract import Tree, Node, Leaf, Binary

import torch
from torch.distributions import Distribution
from torch import Size


class TreeGenerator(Generic[Node]):
    def __init__(self, leaves: set[Node], operators: set[Node]):
        self.leaves = list(leaves)
        self.operators = list(operators)

    def random_tree(self, depth: int) -> Tree[Node]:
        if depth == 0:
            return Leaf(random.choice(self.leaves))
        return Binary(random.choice(self.operators),
                      self.random_tree(ldepth := random.randint(0, depth - 1)),
                      self.random_tree(depth - 1 if ldepth < (depth - 1) else random.randint(0, depth - 1)))

    def generate(self, depth: int, num_trees: int) -> Iterator[Tree[Node]]:
        yield from (self.random_tree(depth) for _ in range(num_trees))
        

@dataclass
class TreeTask(ABC, metaclass=ABCMeta):
    @abstractmethod
    def sample(self, depth: int) -> TreeSample:
        ...
    
    def sample_many(self, depths: list[int]) -> list[TreeSample]:
        return [self.sample(length) for length in depths]

    def sample_len(self, distribution: Distribution, num_samples: int) -> list[TreeSample]:
        return self.sample_many(distribution.sample(Size((num_samples,))).long().tolist())

    def make_sets(
            self,
            distributions: tuple[Distribution, ...],
            num_samples: tuple[int, ...],
            seed: int) -> tuple[list[TreeSample], ...]:
        torch.manual_seed(seed)
        random.seed(seed)
        return tuple(self.sample_len(distribution, ns) for distribution, ns in zip(distributions, num_samples))


@dataclass(frozen=True)
class TreeSample:
    x: Tree[int]
    y: Tree[int]
    task: TreeTask
    
    def __hash__(self) -> int:
        return hash((self.x, self.y, self.task))
