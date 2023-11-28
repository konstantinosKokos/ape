from torch.nn import Module
from torch import Tensor
import torch

from .base import Base
from ...neural.encoder import Encoder
from ...neural.decoder import Decoder
from ...neural.position import Relative
from ...neural.embedding import InvertibleEmbedding


class SequentialRelative(Module, Base):
    def __init__(
            self,
            vocab_size: int,
            dim: int,
            num_heads: int,
            num_layers: tuple[int, int],
            window_size: int):
        super(SequentialRelative, self).__init__()
        self.encoder = Encoder(num_heads=num_heads, num_layers=num_layers[0], dim=dim)
        self.decoder = Decoder(num_heads=num_heads, num_layers=num_layers[1], dim=dim)
        self.positional_encoder = Relative(dim=dim // num_heads, window_size=window_size)
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
        enc_len = encoder_ids.shape[1]
        dec_len = decoder_ids.shape[1]
        max_seq_len = max(enc_len, dec_len)
        distances = torch.arange(max_seq_len, device=decoder_input.device)[None, :]
        distances = distances[:, :, None] - distances[:, None, :]
        mediator = self.positional_encoder.forward(distances).unsqueeze(-1)

        enc_atn_fn = self.positional_encoder.adjust_attention(mediator[:, :enc_len, :enc_len])
        dec_atn_fn = self.positional_encoder.adjust_attention(
            decoder_mask[:, :, :, None, None] * mediator[:, :dec_len, :dec_len])
        cross_atn_fn = self.positional_encoder.adjust_attention(mediator[:, :dec_len, :enc_len])

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