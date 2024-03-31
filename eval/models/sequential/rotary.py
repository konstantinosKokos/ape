from torch.nn import Module
from torch import Tensor

from .base import Base
from unitaryPE.nn.encoder import Encoder
from unitaryPE.nn.decoder import Decoder
from unitaryPE.nn.position import Rotary
from unitaryPE.nn.embedding import InvertibleEmbedding


class SequentialRotary(Module, Base):
    def __init__(
            self,
            vocab_size: int,
            dim: int,
            num_heads: int,
            num_layers: tuple[int, int]):
        super(SequentialRotary, self).__init__()
        self.encoder = Encoder(num_heads=num_heads, num_layers=num_layers[0], dim=dim)
        self.decoder = Decoder(num_heads=num_heads, num_layers=num_layers[1], dim=dim)
        self.positional_encoder = Rotary(num_positions=300, embedding_dim=dim//num_heads)
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
        spos = self.positional_encoder.forward(max_seq_len)
        enc_atn_fn = self.positional_encoder.adjust_attention(sinusoidal_pos=spos)
        dec_atn_fn = self.positional_encoder.adjust_attention(sinusoidal_pos=spos)
        cross_atn_fn = self.positional_encoder.adjust_attention(sinusoidal_pos=spos)
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
