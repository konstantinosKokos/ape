from .base import Base, make_decoder_mask, beam_active

from torch.nn import Module
from torch import Tensor
import torch

from unitaryPE.nn.encoder import Encoder
from unitaryPE.nn.decoder import Decoder
from unitaryPE.nn.position import Relative
from unitaryPE.nn.embedding import InvertibleEmbedding


class MTRelative(Module, Base):
    def __init__(
            self,
            vocab_size: int,
            dim: int,
            num_heads: int,
            num_layers: tuple[int, int],
            window_size: int,
            sos_token_id: int,
            eos_token_id: int):
        super(MTRelative, self).__init__()
        self.encoder = Encoder(num_heads=num_heads, num_layers=num_layers[0], dim=dim)
        self.decoder = Decoder(num_heads=num_heads, num_layers=num_layers[1], dim=dim)
        self.positional_encoder = Relative(dim=dim // num_heads, window_size=window_size)
        self.embedding = InvertibleEmbedding(num_classes=vocab_size, dim=dim)
        self.vocab_size = vocab_size
        self.dim = dim
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id

    def forward_train(
            self,
            source_ids: Tensor,
            source_mask: Tensor,
            target_ids: Tensor,
            target_mask: Tensor) -> Tensor:
        source_embeddings = self.embedding.embed(source_ids)
        target_embeddings = self.embedding.embed(target_ids)

        distances = torch.arange(max(source_ids.size(1), target_ids.size(1)), device=source_ids.device)
        distances = distances[:, None] - distances[None, :]
        mediator = self.positional_encoder.forward(distances)

        enc_atn_fn = self.positional_encoder.adjust_attention(
            qk_pos=mediator[None, :source_ids.size(1), :source_ids.size(1), :, None])
        dec_atn_fn = self.positional_encoder.adjust_attention(
            mediator[None, :target_ids.size(1), :target_ids.size(1), :, None])
        cross_atn_fn = self.positional_encoder.adjust_attention(
            qk_pos=mediator[None, :target_ids.size(1), :source_ids.size(1), :, None])

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
        distances = torch.arange(max(max_decode_length, source_ids.size(1)), device=source_ids.device)
        distances = distances[:, None] - distances[None, :]
        mediator = self.positional_encoder.forward(distances)

        enc_atn_fn = self.positional_encoder.adjust_attention(
            qk_pos=mediator[None, :source_ids.size(1), :source_ids.size(1), :, None])
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
                dec_atn_fn=self.positional_encoder.adjust_attention(
                    qk_pos=mediator[None, :current_step, :current_step, :, None]),
                cross_atn_fn=self.positional_encoder.adjust_attention(
                    qk_pos=mediator[None, :current_step, :source_ids.size(1), :, None]),
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
