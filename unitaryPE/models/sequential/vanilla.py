from torch.nn import Module
from torch import Tensor
import torch

from .base import Base
from ...neural.encoder import Encoder
from ...neural.decoder import Decoder
from ...neural.position import SinusoidalFlat
from ...neural.embedding import InvertibleEmbedding
from ...neural.attention import multihead_atn_fn


class SequentialVanilla(Module, Base):
    def __init__(
            self,
            vocab_size: int,
            dim: int,
            num_heads: int,
            num_layers: tuple[int, int]):
        super(SequentialVanilla, self).__init__()
        self.encoder = Encoder(num_heads=num_heads, num_layers=num_layers[0], dim=dim)
        self.decoder = Decoder(num_heads=num_heads, num_layers=num_layers[1], dim=dim)
        self.positional_encoder = SinusoidalFlat(dim=dim)
        self.embedding = InvertibleEmbedding(num_classes=vocab_size, dim=dim)

    def forward(
            self,
            encoder_ids: Tensor,
            encoder_mask: Tensor,
            decoder_ids: Tensor,
            decoder_mask: Tensor,
            cross_mask: Tensor) -> Tensor:
        encoder_input = self.embedding.embed(encoder_ids)
        decoder_input = self.embedding.embed(decoder_ids)
        max_seq_len = max(encoder_ids.shape[1], decoder_ids.shape[1])
        self.positional_encoder.precompute(max_seq_len)
        positions = torch.arange(max_seq_len, device=decoder_input.device)[None, :]

        encoder_input = encoder_input + self.positional_encoder.forward(positions[None, :encoder_ids.shape[1]])
        decoder_input = decoder_input + self.positional_encoder.forward(positions[None, :decoder_ids.shape[1]])
        enc_atn_fn = multihead_atn_fn
        dec_atn_fn = multihead_atn_fn
        cross_atn_fn = multihead_atn_fn

        encoder_input = self.encoder.forward(
            encoder_input=encoder_input,
            encoder_mask=encoder_mask,
            atn_fn=enc_atn_fn)
        decoder_output = self.decoder.forward(
            encoder_input=encoder_input,
            decoder_input=decoder_input,
            decoder_mask=decoder_mask,
            cross_mask=cross_mask,
            self_atn_fn=dec_atn_fn,
            cross_atn_fn=cross_atn_fn)
        return self.embedding.invert(decoder_output)
