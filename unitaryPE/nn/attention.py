import torch
from torch import Tensor
from torch.nn import Linear, Module, Dropout
from math import sqrt
from typing import Callable

AtnFn = Callable[[Tensor, Tensor, Tensor], Tensor]


def multihead_atn_fn(
        queries: Tensor,
        keys: Tensor,
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

    return weights.softmax(dim=-2)


class SelfMHA(Module):
    def __init__(self, num_heads: int, dim: int, dropout_rate: float = 0.0) -> None:
        super(SelfMHA, self).__init__()
        self.linear = Linear(dim, 3 * dim, False)
        self.out = Linear(dim, dim, False)
        self.num_heads = num_heads
        self.dropout = Dropout(dropout_rate)

    def forward(self, xs: Tensor, mask: Tensor, atn_fn: AtnFn) -> Tensor:
        qs, ks, vs = self.linear(xs).chunk(3, dim=-1)
        qs = qs.view(xs.shape[0], xs.shape[1], -1, self.num_heads)
        ks = ks.view(xs.shape[0], xs.shape[1], -1, self.num_heads)
        vs = vs.view(xs.shape[0], xs.shape[1], -1, self.num_heads)
        atn_weights = atn_fn(qs, ks, mask)
        atn_weights = self.dropout(atn_weights)
        atn_values = torch.einsum('bqkh,bkdh->bqdh', atn_weights, vs).flatten(-2)
        atn_values = self.out(atn_values)
        return self.dropout(atn_values)


class CrossMHA(Module):
    def __init__(self, num_heads: int, dim: int, dropout_rate: float = 0.0) -> None:
        super(CrossMHA, self).__init__()
        self.lin_q = Linear(dim, dim, False)
        self.lin_kv = Linear(dim, 2 * dim, False)
        self.out = Linear(dim, dim, False)
        self.num_heads = num_heads
        self.dropout = Dropout(dropout_rate)

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
        atn_weights = atn_fn(qs, ks, cross_mask)
        atn_weights = self.dropout(atn_weights)
        atn_values = torch.einsum('bqkh,bkdh->bqdh', atn_weights, vs).flatten(-2)
        atn_values = self.out(atn_values)
        return self.out(atn_values)
