import torch
from torch import Tensor
from ..attention import AtnFn, multihead_atn_fn


def applicative(
        q_maps: Tensor | None,
        k_maps: Tensor | None,
        mediator: tuple[Tensor, bool] | None = None) -> AtnFn:
    def wrapped(queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor | None) -> Tensor:
        if q_maps is not None:
            queries = torch.einsum('bqAh,bqhAB->bqBh', queries, q_maps)
        if k_maps is not None:
            keys = torch.einsum('bkBh,bkhAB->bkAh', keys, k_maps)
        return multihead_atn_fn(queries, keys, values, mask, mediator=mediator)
    return wrapped


def additive_mediator(qk: Tensor) -> AtnFn:
    def wrapper(queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor | None) -> Tensor:
        return multihead_atn_fn(queries, keys, values, mask, mediator=(qk, False))
    return wrapper


def multiplicative_mediator(qk: Tensor) -> AtnFn:
    def wrapped(queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor | None) -> Tensor:
        return multihead_atn_fn(queries, keys, values, mask, mediator=(qk, True))
    return wrapped


def orthogonal_penalty(x: Tensor) -> Tensor:
    dim = x.shape[0]
    return (x@x.t() - torch.eye(dim, device=x.device)).abs().sum() / (dim**2 - dim)
