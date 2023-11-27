import torch
from torch import Tensor
from torch.nn import Linear, Module
from math import sqrt
from typing import Callable

AtnFn = Callable[[Tensor, Tensor, Tensor, Tensor | None], Tensor]


def multihead_atn_fn(
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Tensor | None,
        mediator: tuple[Tensor, bool] | None = None) -> Tensor:
    batch_size, seq_len, dim, num_heads = keys.shape

    match mediator:
        case None:
            weights = torch.einsum('bqdh,bkdh->bqkh', queries, keys) / sqrt(dim)
        case (mediator_weights, True):
            weights = torch.einsum('bqdh,bkdh,bqkdh->bqkh', queries, keys, mediator_weights) / sqrt(dim)
        case (mediator_weights, False):
            weights = torch.einsum('bqdh,bkdh->bqkh', queries, keys) / sqrt(dim)
            additive_weights = torch.einsum('bqdh,bqkdh->bqkh', queries, mediator_weights)
            weights = weights + additive_weights
        case _:
            raise ValueError

    if mask is not None:
        if mask.shape == (batch_size, seq_len):
            mask = mask[:, None, :]
        weights = weights.masked_fill_(~mask[:, :, :, None], value=-1e10)

    weights = weights.softmax(dim=-2)
    return torch.einsum('bqkh,bkdh->bqdh', weights, values).flatten(-2)


class SelfMHA(Module):
    def __init__(self, num_heads: int, dim: int) -> None:
        super(SelfMHA, self).__init__()
        self.linear = Linear(dim, 3 * dim, False)
        self.out = Linear(dim, dim, False)
        self.num_heads = num_heads

    def forward(self, xs: Tensor, mask: Tensor, atn_fn: AtnFn) -> Tensor:
        qs, ks, vs = self.linear(xs).chunk(3, dim=-1)
        qs = qs.view(xs.shape[0], xs.shape[1], -1, self.num_heads)
        ks = ks.view(xs.shape[0], xs.shape[1], -1, self.num_heads)
        vs = vs.view(xs.shape[0], xs.shape[1], -1, self.num_heads)
        atn = atn_fn(qs, ks, vs, mask)
        return self.out(atn)


class CrossMHA(Module):
    def __init__(self, num_heads: int, dim: int) -> None:
        super(CrossMHA, self).__init__()
        self.lin_q = Linear(dim, dim, False)
        self.lin_kv = Linear(dim, 2 * dim, False)
        self.out = Linear(dim, dim, False)
        self.num_heads = num_heads

    def forward(
            self,
            decoder_input: Tensor,
            encoder_input: Tensor,
            cross_mask: Tensor,
            atn_fn: AtnFn) -> Tensor:
        qs = self.lin_q(decoder_input)
        ks, vs = self.lin_kv(encoder_input).chunk(2, dim=-1)
        qs = qs.view(decoder_input.shape[0], decoder_input.shape[1], -1, self.num_heads)
        ks = ks.view(encoder_input.shape[0], encoder_input.shape[1], -1, self.num_heads)
        vs = vs.view(encoder_input.shape[0], encoder_input.shape[1], -1, self.num_heads)
        atn = atn_fn(qs, ks, vs, cross_mask)
        return self.out(atn)
