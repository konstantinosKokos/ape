from torch.nn import Module, ModuleList, Sequential, Linear, ReLU, LayerNorm, Dropout
from torch import Tensor
from .attention import SelfMHA, CrossMHA, AtnFn


class Decoder(Module):
    def __init__(
            self,
            num_heads: int,
            num_layers: int,
            dim: int) -> None:
        super(Decoder, self).__init__()
        self.decoder_layers = ModuleList([DecoderLayer(num_heads, dim, 0.15) for _ in range(num_layers)])

    def forward(
            self,
            encoder_input: Tensor,
            cross_mask: Tensor,
            decoder_input: Tensor,
            decoder_mask: Tensor,
            self_atn_fn: AtnFn,
            cross_atn_fn: AtnFn) -> Tensor:
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
            dropout_rate: float) -> None:
        super(DecoderLayer, self).__init__()
        self.self_mha = SelfMHA(num_heads, dim)
        self.self_mha_ln = LayerNorm(dim)
        self.cross_mha = CrossMHA(num_heads, dim)
        self.cross_mha_ln = LayerNorm(dim)
        self.ffn = Sequential(Linear(dim, 2 * dim), ReLU(), Linear(2 * dim, dim))
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
        dec_mha = self.self_mha_ln.forward(decoder_input)
        dec_mha = self.self_mha.forward(
            xs=dec_mha,
            mask=decoder_mask,
            atn_fn=self_atn_fn)
        dec_mha = self.dropout(dec_mha)
        dec_mha = dec_mha + decoder_input

        cross_mha = self.cross_mha_ln(dec_mha)
        cross_mha = self.cross_mha.forward(
            decoder_input=cross_mha,
            encoder_input=encoder_input,
            cross_mask=cross_mask,
            atn_fn=cross_atn_fn)
        cross_mha = self.dropout(cross_mha)
        cross_mha = dec_mha + cross_mha

        ffn = self.ffn_ln(cross_mha)
        ffn = self.ffn(ffn)
        return ffn + cross_mha
