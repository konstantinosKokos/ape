import torch
from torch import Tensor
from ..attention import AtnFn, multihead_atn_fn, Callable

AtnFn2 = Callable[[AtnFn], AtnFn]


def applicative(q_maps: Tensor | None, k_maps: Tensor | None) -> AtnFn2:
    def wrapper(atn_fn: AtnFn = multihead_atn_fn) -> AtnFn:
        def wrapped(queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor | None) -> Tensor:
            if q_maps is not None:
                queries = torch.einsum('bqAh,bqAB->bqBh', queries, q_maps)
            if k_maps is not None:
                keys = torch.einsum('bkBh,bkAB->bkAh', keys, k_maps)
            return atn_fn(queries, keys, values, mask)
        return wrapped
    return wrapper


def intermediating(qk: Tensor) -> AtnFn:
    def wrapped(queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor | None) -> Tensor:
        return multihead_atn_fn(queries, keys, values, mask, qk)
    return wrapped


def orthogonal_penalty(x: Tensor) -> Tensor:
    dim = x.shape[0]
    return (x@x.t() - torch.eye(dim, device=x.device)).abs().sum() / (dim**2 - dim)
