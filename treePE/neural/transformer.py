from __future__ import annotations

import pdb

import torch
from torch.nn import Module, ModuleList, Linear, Sequential, ReLU, LayerNorm, Parameter
from torch.nn.functional import dropout, linear, embedding
from torch import Tensor

from math import sqrt
from opt_einsum import contract


class Transformer(Module):
    def __init__(self, dim: int, num_heads: int,
                 enc_vocab_size: int, enc_num_layers: int, enc_positional_encoder: Module,
                 dec_vocab_size: int, dec_num_layers: int, dec_positional_encoder: Module) -> None:
        super(Transformer, self).__init__()
        self.encoder = Encoder(enc_vocab_size, num_heads, enc_num_layers, dim, enc_positional_encoder)
        self.decoder = Decoder(dec_vocab_size, num_heads, dec_num_layers, dim, dec_positional_encoder)

    def encode(self, encoder_ids: Tensor, encoder_positions: Tensor, encoder_mask: Tensor) -> Tensor:
        return self.encoder(encoder_ids, encoder_positions, encoder_mask)

    def decode(self, enc_ctx: Tensor,
               decoder_ids: Tensor, decoder_positions: Tensor, decoder_mask: Tensor,
               cross_mask: Tensor) -> Tensor:
        return self.decoder(enc_ctx, cross_mask, decoder_ids, decoder_positions, decoder_mask)

    def forward(self,
                encoder_ids: Tensor, encoder_positions: Tensor, encoder_mask: Tensor,
                decoder_ids: Tensor, decoder_positions: Tensor, decoder_mask: Tensor,
                cross_mask: Tensor):
        enc_ctx = self.encode(encoder_ids, encoder_positions, encoder_mask)
        return self.decode(enc_ctx, decoder_ids, decoder_positions, decoder_mask, cross_mask)


class Encoder(Module):
    def __init__(self, vocab_size: int, num_heads: int, num_layers: int, dim: int, positional_encoder: Module) -> None:
        super(Encoder, self).__init__()
        self.encoder_layers = ModuleList([EncoderLayer(num_heads, dim, 0.05) for _ in range(num_layers)])
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


class Decoder(Module):
    def __init__(self, vocab_size: int, num_heads: int, num_layers: int, dim: int, positional_encoder: Module) -> None:
        super(Decoder, self).__init__()
        self.decoder_layers = ModuleList([DecoderLayer(num_heads, dim, 0.1) for _ in range(num_layers)])
        self.positional_encoder = positional_encoder
        self.embedding = InvertibleEmbedding(num_classes=vocab_size, dim=dim)

    def forward(self, enc_ctx: Tensor, enc_mask: Tensor,
                content_ids: Tensor, position_ids: Tensor, dec_mask: Tensor) -> Tensor:
        embeddings = self.embedding.embed(content_ids)
        embeddings = self.positional_encoder(position_ids, embeddings)
        return self.decode_step(enc_ctx, enc_mask, embeddings, dec_mask)

    def decode_step(self, enc_ctx: Tensor, enc_mask: Tensor, dec_ctx: Tensor, dec_mask: Tensor) -> Tensor:
        for decoder in self.decoder_layers:
            dec_ctx = decoder(enc_ctx, enc_mask, dec_ctx, dec_mask)
        return dec_ctx


class DecoderLayer(Module):
    def __init__(self, num_heads: int, dim: int, dropout_rate: float) -> None:
        super(DecoderLayer, self).__init__()
        self.self_mha = SelfMHA(num_heads, dim, dropout_rate)
        self.self_mha_ln = LayerNorm(dim)
        self.cross_mha = CrossMHA(num_heads, dim, dropout_rate)
        self.cross_mha_ln = LayerNorm(dim)
        self.ffn = Sequential(Linear(dim, 4 * dim), ReLU(), Linear(4 * dim, dim))
        self.ffn_ln = LayerNorm(dim)
        self.dropout_rate = dropout_rate

    def forward(self, enc_ctx: Tensor, cross_mask: Tensor, dec_ctx: Tensor, dec_mask: Tensor) -> Tensor:
        dec_ctx = dropout(dec_ctx, p=self.dropout_rate, training=self.training)
        dec_mha = self.self_mha(dec_ctx, dec_mask)
        dec_mha = self.self_mha_ln(dec_mha + dec_ctx)

        enc_dec_ctx = self.cross_mha(dec_ctx, enc_ctx, cross_mask)
        enc_dec_ctx = self.cross_mha_ln(enc_dec_ctx + dec_mha)

        out = self.ffn(enc_dec_ctx)
        out = dropout(out, p=self.dropout_rate, training=self.training)
        return self.ffn_ln(out + enc_dec_ctx)


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


class CrossMHA(Module):
    def __init__(self, num_heads: int, dim: int, dropout_rate: float) -> None:
        super(CrossMHA, self).__init__()
        self.lin_q = Linear(dim, dim, False)
        self.lin_kv = Linear(dim, 2 * dim, False)
        self.out = Linear(dim, dim, False)
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def forward(self, dec_xs: Tensor, enc_xs: Tensor, mask: Tensor) -> Tensor:
        qs = dropout(self.lin_q(dec_xs), p=self.dropout_rate, training=self.training)
        ks, vs = dropout(self.lin_kv(enc_xs), p=self.dropout_rate, training=self.training).chunk(2, dim=-1)
        qs = qs.view(enc_xs.shape[0], enc_xs.shape[1], -1, self.num_heads)
        ks = ks.view(enc_xs.shape[0], enc_xs.shape[1], -1, self.num_heads)
        vs = vs.view(enc_xs.shape[0], enc_xs.shape[1], -1, self.num_heads)
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
