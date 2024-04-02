from math import cos, radians

from typing import Callable
from warnings import warn


def make_schedule(warmup_steps: int,
                  total_steps: int,
                  warmdown_steps: int,
                  max_lr: float,
                  min_lr: float,
                  init_lr: float) -> Callable[[int], float]:
    linear_schedule = make_linear_schedule(warmup_steps, max_lr, init_lr)
    cosine_schedule = make_cosine_schedule(warmdown_steps, max_lr, min_lr)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return linear_schedule(step)
        elif step < total_steps - warmdown_steps:
            return max_lr
        elif step > total_steps:
            warn(f"Step is greater than total steps")
            return min_lr
        return cosine_schedule(step - (total_steps - warmdown_steps))
    return schedule


def make_linear_schedule(warmup_steps: int, max_lr: float, min_lr: float) -> Callable[[int], float]:
    slope = (max_lr - min_lr)/warmup_steps

    def linear_schedule(step: int) -> float:
        if step < warmup_steps:
            return min_lr + step * slope
        return max_lr
    return linear_schedule


def make_cosine_schedule(decay_steps: int, max_lr: float, min_lr: float) -> Callable[[int], float]:
    def cosine_schedule(step: int) -> float:
        if step <= decay_steps:
            return min_lr + (max_lr - min_lr) * (cos(radians(step / decay_steps * 180)) + 1) / 2
        return min_lr
    return cosine_schedule


def make_transformer_schedule(dim: int, warmup_steps: int):
    def schedule(step: int) -> float:
        step += 1
        return dim ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)
    return schedule
