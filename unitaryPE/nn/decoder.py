from torch.nn import Module, ModuleList, Sequential, Linear, ReLU, GELU, LayerNorm, Dropout
from torch import Tensor
from .attention import SelfMHA, CrossMHA, AtnFn
from typing import Literal


def _get_activation(name: Literal['ReLU', 'GELU']) -> Module:
    match name:
        case 'ReLU':
            return ReLU()
        case 'GELU':
            return GELU()
        case _:
            raise ValueError


class Decoder(Module):
    def __init__(
            self,
            num_heads: int,
            num_layers: int,
            dim: int,
            dropout_rate: float = 0.1,
            weight_dropout: float = 0.1,
            mlp_ratio: int = 4,
            activation: Literal['ReLU', 'GELU'] = 'ReLU') -> None:
        super(Decoder, self).__init__()
        self.dropout = Dropout(dropout_rate)
        self.decoder_layers = ModuleList(
            [DecoderLayer(
                num_heads=num_heads,
                dim=dim,
                dropout_rate=dropout_rate,
                weight_dropout=weight_dropout,
                mlp_ratio=mlp_ratio,
                activation=activation)
             for _ in range(num_layers)])

    def forward(
            self,
            encoder_input: Tensor,
            cross_mask: Tensor,
            decoder_input: Tensor,
            decoder_mask: Tensor,
            self_atn_fn: AtnFn,
            cross_atn_fn: AtnFn) -> Tensor:
        decoder_input = self.dropout(decoder_input)
        for decoder in self.decoder_layers:
            decoder_input = decoder.forward(
                encoder_input=encoder_input,
                cross_mask=cross_mask,
                decoder_input=decoder_input,
                decoder_mask=decoder_mask,
                self_atn_fn=self_atn_fn,
                cross_atn_fn=cross_atn_fn)
        return decoder_input


class DecoderLayer(Module):
    def __init__(
            self,
            num_heads: int,
            dim: int,
            dropout_rate: float,
            weight_dropout: float,
            mlp_ratio: int,
            activation: Literal['ReLU', 'GELU']) -> None:
        super(DecoderLayer, self).__init__()
        self.self_mha = SelfMHA(num_heads, dim, dropout_rate=weight_dropout)
        self.self_mha_ln = LayerNorm(dim)
        self.cross_mha = CrossMHA(num_heads, dim, dropout_rate=weight_dropout)
        self.cross_mha_ln = LayerNorm(dim)
        self.ffn = Sequential(Linear(dim, mlp_ratio * dim), _get_activation(activation), Linear(mlp_ratio * dim, dim))
        self.ffn_ln = LayerNorm(dim)
        self.dropout = Dropout(dropout_rate)

    def forward(
            self,
            encoder_input: Tensor,
            cross_mask: Tensor,
            decoder_input: Tensor,
            decoder_mask: Tensor,
            self_atn_fn: AtnFn,
            cross_atn_fn: AtnFn) -> Tensor:
        dec_mha = self.self_mha.forward(
            xs=decoder_input,
            mask=decoder_mask,
            atn_fn=self_atn_fn
        )
        dec_mha = decoder_input + self.dropout(dec_mha)
        dec_mha = self.self_mha_ln(dec_mha)
        cross_mha = self.cross_mha.forward(
            encoder_input=encoder_input,
            decoder_input=dec_mha,
            cross_mask=cross_mask,
            atn_fn=cross_atn_fn
        )
        cross_mha = dec_mha + self.dropout(cross_mha)
        cross_mha = self.cross_mha_ln(cross_mha)

        ffn = self.ffn(cross_mha)
        ffn = cross_mha + self.dropout(ffn)
        return self.ffn_ln(ffn)
