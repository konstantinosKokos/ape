from __future__ import annotations

import pdb

import torch
from torch.nn import Module, ModuleList, Linear, Sequential, ReLU, LayerNorm, Parameter
from torch.nn.functional import dropout, linear, embedding
from torch import Tensor

from math import sqrt
from opt_einsum import contract


class Encoder(Module):
    def __init__(self, vocab_size: int, num_heads: int, num_layers: int, dim: int, positional_encoder: Module) -> None:
        super(Encoder, self).__init__()
        self.encoder_layers = ModuleList([EncoderLayer(num_heads, dim, 0.1) for _ in range(num_layers)])
        self.positional_encoder = positional_encoder
        self.embedding = InvertibleEmbedding(num_classes=vocab_size, dim=dim)

    def forward(self, content_ids: Tensor, position_ids: Tensor, atn_mask: Tensor) -> Tensor:
        embeddings = self.embedding.embed(content_ids)
        embeddings = self.positional_encoder(position_ids, embeddings)
        return self.encode(embeddings, atn_mask)

    def encode(self, weights: Tensor, mask: Tensor) -> Tensor:
        for encoder in self.encoder_layers:
            weights = encoder(weights, mask)
        return weights


class EncoderLayer(Module):
    def __init__(self, num_heads: int, dim: int, dropout_rate: float) -> None:
        super(EncoderLayer, self).__init__()
        self.mha = SelfMHA(num_heads, dim, dropout_rate)
        self.feed_forward = Sequential(Linear(dim, 4 * dim), ReLU(), Linear(4 * dim, dim))
        self.mha_ln = LayerNorm(dim)
        self.ff_ln = LayerNorm(dim)
        self.dropout_rate = dropout_rate

    def forward(self, xs: Tensor, mask: Tensor) -> Tensor:
        xs = dropout(xs, p=self.dropout_rate, training=self.training)
        mha = self.mha(xs, mask)
        mha = self.mha_ln(xs + mha)
        ffn = self.feed_forward(mha)
        ffn = self.ff_ln(mha + ffn)
        return ffn


def multihead_atn_fn(queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor) -> Tensor:
    dk, num_heads = keys.shape[-2:]
    weights = contract('bqdh,bkdh->bqkh', queries, keys) / sqrt(dk)     # type: ignore
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_heads)
    weights = weights.masked_fill_(mask == 0, value=-1e10)
    weights = weights.softmax(dim=-2)
    return contract('bqkh,bkdh->bqdh', weights, values).flatten(-2)     # type: ignore


class SelfMHA(Module):
    def __init__(self, num_heads: int, dim: int, dropout_rate: float) -> None:
        super(SelfMHA, self).__init__()
        self.linear = Linear(dim, 3 * dim, False)
        self.out = Linear(dim, dim, False)
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def forward(self, xs: Tensor, mask: Tensor) -> Tensor:
        qs, ks, vs = dropout(self.linear(xs), p=self.dropout_rate, training=self.training).chunk(3, dim=-1)
        qs = qs.view(xs.shape[0], xs.shape[1], -1, self.num_heads)
        ks = ks.view(xs.shape[0], xs.shape[1], -1, self.num_heads)
        vs = vs.view(xs.shape[0], xs.shape[1], -1, self.num_heads)
        return self.out(multihead_atn_fn(qs, ks, vs, mask))


class InvertibleEmbedding(Module):
    def __init__(self, num_classes: int, dim: int):
        super(InvertibleEmbedding, self).__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.weight = Parameter(torch.nn.init.normal_(torch.empty(num_classes, dim)), requires_grad=True)

    def embed(self, xs: Tensor) -> Tensor:
        return embedding(xs, self.weight)  # * sqrt(self.dim)

    def invert(self, xs: Tensor) -> Tensor:
        return linear(xs, self.weight)
