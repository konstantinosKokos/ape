from torch.nn import Module
from torch import Tensor
import torch

from .base import Base
from ...neural.encoder import Encoder
from ...neural.decoder import Decoder
from ...neural.position import UnitaryBranching
from ...neural.embedding import InvertibleEmbedding


class TreeUnitary(Module, Base):
    def __init__(
            self,
            vocab_size: int,
            dim: int,
            num_heads: int,
            num_layers: tuple[int, int],
            branching_factor: int):
        super(TreeUnitary, self).__init__()
        self.encoder = Encoder(num_heads=num_heads, num_layers=num_layers[0], dim=dim)
        self.decoder = Decoder(num_heads=num_heads, num_layers=num_layers[1], dim=dim)
        self.positional_encoder = UnitaryBranching(
            dim=dim//num_heads,
            branching_factor=branching_factor,
            num_heads=num_heads)
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
        decoder_input = self.embedding.embed(decoder_ids)

        unique_enc_pos, inverse_x = encoder_pos.unique(return_inverse=True)
        unique_dec_pos, inverse_y = decoder_pos.unique(return_inverse=True)
        unique_pos, inverse_xy = torch.cat((unique_enc_pos, unique_dec_pos)).unique(return_inverse=True)
        self.positional_encoder.precompute(positions=unique_pos.cpu().tolist())
        unique_enc_maps = self.positional_encoder.forward(inverse_xy[:len(unique_enc_pos)])
        unique_dec_maps = self.positional_encoder.forward(inverse_xy[len(unique_enc_maps):])
        enc_maps = unique_enc_maps[inverse_x]
        dec_maps = unique_dec_maps[inverse_y]

        encoder_depths = encoder_pos // 2
        decoder_depths = decoder_pos // 2

        enc_atn_fn = self.positional_encoder.adjust_attention(
            q_maps=enc_maps,
            k_maps=enc_maps,
            mediator=(0.98 ** (encoder_depths[:, :, None] - encoder_depths[:, None]), True))
        dec_atn_fn = self.positional_encoder.adjust_attention(
            q_maps=dec_maps,
            k_maps=dec_maps,
            mediator=(0.98 ** (decoder_depths[:, :, None] - decoder_depths[:, None]), True))
        cross_atn_fn = self.positional_encoder.adjust_attention(
            q_maps=dec_maps,
            k_maps=enc_maps,
            mediator=(0.98 ** (decoder_depths[:, :, None] - encoder_depths[:, None]), True))

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
