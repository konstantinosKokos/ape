import torch
from torch import Tensor
from ..attention import AtnFn, multihead_atn_fn


def applicative(q_maps: Tensor | None, k_maps: Tensor | None) -> AtnFn:
    def wrapped(queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor | None) -> Tensor:
        if q_maps is not None:
            queries = torch.einsum('bqAh,bqAB->bqBh', queries, q_maps)
        if k_maps is not None:
            keys = torch.einsum('bkBh,bkAB->bkAh', keys, k_maps)
        return multihead_atn_fn(queries, keys, values, mask)
    return wrapped


def intermediating(qk: Tensor):
    def wrapped(queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor | None) -> Tensor:
        return multihead_atn_fn(queries, keys, values, mask, qk)
    return wrapped
