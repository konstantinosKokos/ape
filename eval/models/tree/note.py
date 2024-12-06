from torch.nn import Module
from torch import Tensor

from .base import Base
from ape.nn.encoder import Encoder
from ape.nn.decoder import Decoder
from ape.nn.position import Branching
from ape.nn.embedding import InvertibleEmbedding
from ape.nn.attention import multihead_atn_fn


class NoTE(Module, Base):
    def __init__(
            self,
            vocab_size: int,
            dim: int,
            num_heads: int,
            num_layers: tuple[int, int],
            branching_factor: int):
        super(NoTE, self).__init__()
        self.encoder = Encoder(num_heads=num_heads, num_layers=num_layers[0], dim=dim)
        self.decoder = Decoder(num_heads=num_heads, num_layers=num_layers[1], dim=dim)
        self.positional_encoder = Branching(
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
        self.positional_encoder.precompute(positions=unique_enc_pos.cpu().tolist())
        unique_maps, unique_steps = self.positional_encoder.maps
        enc_maps = unique_maps[inverse_x]
        steps = unique_steps[inverse_x]
        enc_mediator = make_mediator(steps, inverse_x, inverse_x)

        enc_atn_fn = self.positional_encoder.adjust_attention(
            q_maps=enc_maps,
            k_maps=enc_maps,
            mediator=(enc_mediator, True))

        encoder_input = self.encoder.forward(
            encoder_input=encoder_input,
            encoder_mask=encoder_mask,
            atn_fn=enc_atn_fn)
        decoder_output = self.decoder.forward(
            encoder_input=encoder_input,
            decoder_input=decoder_input,
            decoder_mask=decoder_mask,
            cross_mask=cross_mask,
            self_atn_fn=multihead_atn_fn,
            cross_atn_fn=multihead_atn_fn)
        return self.embedding.invert(decoder_output)


def make_mediator(steps: Tensor, idx_x: Tensor, idx_y: Tensor) -> Tensor:
    return 0.98 ** steps[idx_x.unsqueeze(-1), idx_y.unsqueeze(-2)].unflatten(-1, (-1, 1, 1))
