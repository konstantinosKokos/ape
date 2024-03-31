from __future__ import annotations

import random
from dataclasses import dataclass
from abc import ABC, abstractmethod, ABCMeta

import torch
from torch.distributions import Distribution
from torch import Size


@dataclass
class SequentialTask(ABC, metaclass=ABCMeta):
    vocab_size: int

    @abstractmethod
    def sample(self, length: int) -> SequentialSample:
        ...

    def sample_many(self, lengths: list[int]) -> list[SequentialSample]:
        return [self.sample(length) for length in lengths]

    def sample_len(self, distribution: Distribution, num_samples: int) -> list[SequentialSample]:
        return self.sample_many(distribution.sample(Size((num_samples,))).long().tolist())

    def make_sets(
            self,
            distributions: tuple[Distribution, ...],
            num_samples: tuple[int, ...],
            seed: int) -> tuple[list[SequentialSample], ...]:
        torch.manual_seed(seed)
        random.seed(seed)
        return tuple(self.sample_len(distribution, ns) for distribution, ns in zip(distributions, num_samples))


@dataclass(frozen=True)
class SequentialSample:
    x: tuple[int, ...]
    y: tuple[int, ...]
    task: SequentialTask

    def __hash__(self) -> int:
        return hash((self.x, self.y))
