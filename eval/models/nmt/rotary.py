from .base import Base, make_decoder_mask, beam_active

from torch.nn import Module
from torch import Tensor
import torch

from ape.nn.encoder import Encoder
from ape.nn.decoder import Decoder
from ape.nn.position import Rotary
from ape.nn.embedding import InvertibleEmbedding


class MTRotary(Module, Base):
    def __init__(
            self,
            vocab_size: int,
            dim: int,
            num_heads: int,
            num_layers: tuple[int, int],
            sos_token_id: int,
            eos_token_id: int):
        super(MTRotary, self).__init__()
        self.encoder = Encoder(num_heads=num_heads, num_layers=num_layers[0], dim=dim)
        self.decoder = Decoder(num_heads=num_heads, num_layers=num_layers[1], dim=dim)
        self.positional_encoder = Rotary(embedding_dim=dim//num_heads)
        self.embedding = InvertibleEmbedding(num_classes=vocab_size, dim=dim)
        self.vocab_size = vocab_size
        self.dim = dim
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id

    def forward(
            self,
            source_ids: Tensor,
            source_mask: Tensor,
            target_ids: Tensor,
            causal_mask: Tensor,
            reduction: str = 'mean',
            label_smoothing: float = 0.1,
    ) -> tuple[Tensor, Tensor]:
        return super().get_loss(source_ids, target_ids, source_mask, causal_mask, reduction, label_smoothing)

    def forward_train(
            self,
            source_ids: Tensor,
            source_mask: Tensor,
            target_ids: Tensor,
            target_mask: Tensor) -> Tensor:
        source_embeddings = self.embedding.embed(source_ids)
        target_embeddings = self.embedding.embed(target_ids)
        spos = self.positional_encoder.forward(max_seq_len=max(source_ids.size(1), target_ids.size(1)))

        enc_atn_fn = self.positional_encoder.adjust_attention(sinusoidal_pos=spos)
        dec_atn_fn = self.positional_encoder.adjust_attention(sinusoidal_pos=spos)
        cross_atn_fn = self.positional_encoder.adjust_attention(sinusoidal_pos=spos)

        encoder_output = self.encoder.forward(
            encoder_input=source_embeddings,
            encoder_mask=source_mask,
            atn_fn=enc_atn_fn)
        decoder_output = self.decoder.forward(
            encoder_input=encoder_output,
            decoder_input=target_embeddings,
            decoder_mask=target_mask,
            cross_mask=source_mask,
            self_atn_fn=dec_atn_fn,
            cross_atn_fn=cross_atn_fn)
        return self.embedding.invert(decoder_output)

    def forward_dev(
            self,
            source_ids: Tensor,
            source_mask: Tensor,
            causal_mask: Tensor,
            max_decode_length: int,
            beam_width: int,
            alpha: float = 0.6
    ) -> tuple[Tensor, Tensor]:
        source_embeddings = self.embedding.embed(source_ids)
        spos = self.positional_encoder.forward(max_seq_len=max(max_decode_length, source_ids.size(1)))

        enc_atn_fn = self.positional_encoder.adjust_attention(sinusoidal_pos=spos)
        dec_atn_fn = self.positional_encoder.adjust_attention(sinusoidal_pos=spos)
        cross_atn_fn = self.positional_encoder.adjust_attention(sinusoidal_pos=spos)

        encoder_output = self.encoder.forward(
            encoder_input=source_embeddings,
            encoder_mask=source_mask,
            atn_fn=enc_atn_fn)
        encoder_output = encoder_output.repeat_interleave(beam_width, dim=0)
        source_mask = source_mask.repeat_interleave(beam_width, dim=0)

        decoding: bool = True
        beam_paths = torch.ones(source_embeddings.size(0), beam_width, 1, dtype=torch.long, device=source_ids.device)
        beam_paths *= self.sos_token_id
        beam_scores = torch.zeros(source_embeddings.size(0), beam_width, device=source_ids.device, dtype=torch.float)
        current_step: int = 0

        while decoding:
            current_step += 1
            beam_paths, beam_scores = self.step(
                encoder_output=encoder_output,
                decoder_input=self.embedding.embed(beam_paths).flatten(0, 1),
                dec_atn_fn=dec_atn_fn,
                cross_atn_fn=cross_atn_fn,
                source_mask=source_mask,
                decoder_mask=causal_mask,
                beam_paths=beam_paths,
                beam_scores=beam_scores,
                beam_width=beam_width,
                current_step=current_step,
                alpha=alpha
            )
            decoding = (beam_active(self.eos_token_id, beam_paths).any().item() and current_step < max_decode_length)
        return beam_paths, beam_scores
