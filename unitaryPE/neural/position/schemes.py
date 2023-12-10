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


def grid_applicative(
        q_maps: tuple[Tensor, Tensor] | None,
        k_maps: tuple[Tensor, Tensor] | None,
        mediator: tuple[Tensor, bool] | None = None) -> AtnFn:
    def wrapped(queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor | None) -> Tensor:
        if q_maps is not None:
            queries_x, queries_y = queries.chunk(2, dim=-2)
            maps_x, maps_y = q_maps
            queries_x = torch.einsum('bqAh,bqhAB->bqBh', queries_x, maps_x)
            queries_y = torch.einsum('bqAh,bqhAB->bqBh', queries_y, maps_y)
            queries = torch.cat((queries_x, queries_y), dim=-2)
        if k_maps is not None:
            keys_x, keys_y = keys.chunk(2, dim=-2)
            maps_x, maps_y = q_maps
            keys_x = torch.einsum('bqAh,bqhAB->bqBh', keys_x, maps_x)
            keys_y = torch.einsum('bqAh,bqhAB->bqBh', keys_y, maps_y)
            keys = torch.cat((keys_x, keys_y), dim=-2)
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
