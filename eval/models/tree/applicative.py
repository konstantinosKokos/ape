from torch.nn import Module
from torch import Tensor
import torch

from .base import Base
from unitaryPE.nn.encoder import Encoder
from unitaryPE.nn.decoder import Decoder
from unitaryPE.nn.position import UnitaryBranching
from unitaryPE.nn.embedding import InvertibleEmbedding


def index_steps(steps: Tensor, idx_x: Tensor, idx_y: Tensor) -> Tensor:
    batch_size, num_rows = idx_x.shape
    _, num_cols = idx_y.shape
    row_index = idx_x.unsqueeze(-1).expand(batch_size, num_rows, num_cols)
    col_index = idx_y.unsqueeze(-2).expand(batch_size, num_rows, num_cols)
    return steps[row_index, col_index]


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
        unique_maps, unique_steps = self.positional_encoder.maps
        steps = unique_steps[inverse_xy.unsqueeze(-1), inverse_xy.unsqueeze(0)]
        enc_maps = unique_maps[inverse_xy[:len(unique_enc_pos)]][inverse_x]
        dec_maps = unique_maps[inverse_xy[len(unique_enc_pos):]][inverse_y]
        inverse_y = inverse_y + inverse_x.max() + 1

        enc_mediator = make_mediator(steps, inverse_x, inverse_x)
        dec_mediator = make_mediator(steps, inverse_y, inverse_y)
        x_mediator = make_mediator(steps, inverse_y, inverse_x)

        enc_atn_fn = self.positional_encoder.adjust_attention(
            q_maps=enc_maps,
            k_maps=enc_maps,
            mediator=(enc_mediator, True))
        dec_atn_fn = self.positional_encoder.adjust_attention(
            q_maps=dec_maps,
            k_maps=dec_maps,
            mediator=(dec_mediator, True))
        cross_atn_fn = self.positional_encoder.adjust_attention(
            q_maps=dec_maps,
            k_maps=enc_maps,
            mediator=(x_mediator, True))

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


def make_mediator(steps: Tensor, idx_x: Tensor, idx_y: Tensor) -> Tensor:
    return 0.98 ** steps[idx_x.unsqueeze(-1), idx_y.unsqueeze(-2)].unflatten(-1, (-1, 1, 1))
