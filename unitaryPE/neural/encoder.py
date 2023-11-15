from torch.nn import Module, ModuleList, Sequential, Linear, ReLU, LayerNorm, Dropout
from torch import Tensor
from .attention import SelfMHA, AtnFn


class EncoderLayer(Module):
    def __init__(
            self,
            num_heads: int,
            dim: int,
            dropout_rate: float) -> None:
        super(EncoderLayer, self).__init__()
        self.mha = SelfMHA(num_heads, dim)
        self.ffn = Sequential(Linear(dim, 4 * dim), ReLU(), Linear(4 * dim, dim))
        self.mha_ln = LayerNorm(dim)
        self.ffn_ln = LayerNorm(dim)
        self.dropout = Dropout(dropout_rate)

    def forward(
            self,
            encoder_input: Tensor,
            encoder_mask: Tensor,
            atn_fn: AtnFn) -> Tensor:
        mha = self.mha_ln(encoder_input)
        mha = self.mha.forward(mha, encoder_mask, atn_fn)
        mha = self.dropout(mha)
        mha = mha + encoder_input
        ffn = self.ffn_ln.forward(mha)
        ffn = self.ffn(ffn)
        return ffn + mha


class Encoder(Module):
    def __init__(
            self,
            num_heads: int,
            num_layers: int,
            dim: int) -> None:
        super(Encoder, self).__init__()
        self.encoder_layers = ModuleList([EncoderLayer(num_heads, dim, 0.15) for _ in range(num_layers)])

    def forward(
            self,
            encoder_input: Tensor,
            encoder_mask: Tensor,
            atn_fn: AtnFn) -> Tensor:
        for layer in self.encoder_layers:
            encoder_input = layer.forward(
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                atn_fn=atn_fn)
        return encoder_input
