from torch.nn import Module
from torch import Tensor

from .base import Base
from ...neural.encoder import Encoder
from ...neural.decoder import Decoder
from ...neural.position import TreePE
from ...neural.embedding import InvertibleEmbedding
from ...neural.attention import multihead_atn_fn


class ShivQuirk(Module, Base):
    def __init__(
            self,
            vocab_size: int,
            dim: int,
            num_heads: int,
            num_layers: tuple[int, int],
            branching_factor: int):
        super(ShivQuirk, self).__init__()
        self.encoder = Encoder(num_heads=num_heads, num_layers=num_layers[0], dim=dim)
        self.decoder = Decoder(num_heads=num_heads, num_layers=num_layers[1], dim=dim)
        self.positional_encoder = TreePE(
            dim=dim//7//branching_factor,
            branching_factor=branching_factor,
            max_depth=7)
        self.embedding = InvertibleEmbedding(num_classes=vocab_size, dim=dim)

    def forward(
            self,
            encoder_ids: Tensor,
            encoder_pos: Tensor,
            encoder_mask: Tensor,
            decoder_ids: Tensor,
            decoder_pos: Tensor,
            decoder_mask: Tensor,
            cross_mask: Tensor) -> Tensor:
        encoder_input = self.embedding.embed(encoder_ids)
        encoder_input = encoder_input + self.positional_encoder.forward(encoder_pos)
        decoder_input = self.embedding.embed(decoder_ids)
        decoder_input = decoder_input + self.positional_encoder.forward(decoder_pos)

        encoder_input = self.encoder.forward(
            encoder_input=encoder_input,
            encoder_mask=encoder_mask,
            atn_fn=multihead_atn_fn)
        decoder_output = self.decoder.forward(
            encoder_input=encoder_input,
            decoder_input=decoder_input,
            decoder_mask=decoder_mask,
            cross_mask=cross_mask,
            self_atn_fn=multihead_atn_fn,
            cross_atn_fn=multihead_atn_fn)
        return self.embedding.invert(decoder_output)
