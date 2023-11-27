from torch.nn import Module
from torch import Tensor
import torch

from ...neural.encoder import Encoder
from ...neural.decoder import Decoder
from ...neural.position import UnitarySequential
from ...neural.embedding import InvertibleEmbedding


class SequentialUnitary(Module):
    def __init__(
            self,
            vocab_size: int,
            dim: int,
            num_heads: int,
            num_layers: tuple[int, int]):
        super(SequentialUnitary, self).__init__()
        self.encoder = Encoder(num_heads=num_heads, num_layers=num_layers[0], dim=dim)
        self.decoder = Decoder(num_heads=num_heads, num_layers=num_layers[1], dim=dim)
        self.positional_encoder = UnitarySequential(dim=dim//num_heads)
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
        self.positional_encoder.precompute(max_seq_len)
        positions = torch.arange(max_seq_len, device=decoder_input.device)[None, :]
        distances = (positions[:, :, None] - positions[:, None]).unsqueeze(-1).unsqueeze(-1)
        mediator = (0.98 ** distances.abs())

        enc_maps = self.positional_encoder.forward(positions[:1, :encoder_ids.shape[1]])
        dec_maps = self.positional_encoder.forward(positions[:1, :decoder_ids.shape[1]])
        enc_atn_fn = self.positional_encoder.adjust_attention(
            q_maps=enc_maps,
            k_maps=enc_maps,
            mediator=(mediator[:, :enc_len, :enc_len], True))
        dec_atn_fn = self.positional_encoder.adjust_attention(
            q_maps=dec_maps,
            k_maps=dec_maps,
            mediator=(mediator[:, :dec_len, :dec_len], True))
        cross_atn_fn = self.positional_encoder.adjust_attention(
            q_maps=dec_maps,
            k_maps=enc_maps,
            mediator=(mediator[:, :dec_len, :enc_len], True))

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