from .base import Base, beam_active

from torch.nn import Module
from torch import Tensor
import torch

from unitaryPE.nn.encoder import Encoder
from unitaryPE.nn.decoder import Decoder
from unitaryPE.nn.position import SinusoidalFlat
from unitaryPE.nn.embedding import InvertibleEmbedding
from unitaryPE.nn.attention import multihead_atn_fn


class MTVanilla(Module, Base):
    def __init__(
            self,
            vocab_size: int,
            dim: int,
            num_heads: int,
            num_layers: tuple[int, int],
            sos_token_id: int,
            eos_token_id: int):
        super(MTVanilla, self).__init__()
        self.encoder = Encoder(num_heads=num_heads, num_layers=num_layers[0], dim=dim)
        self.decoder = Decoder(num_heads=num_heads, num_layers=num_layers[1], dim=dim)
        self.positional_encoder = SinusoidalFlat(dim=dim, freq=10000, max_seq_len=500)
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
        encoder_input = self.embedding.embed(source_ids)
        decoder_input = self.embedding.embed(target_ids)
        max_seq_len = max(source_ids.shape[1], target_ids.shape[1])
        positions = torch.arange(max_seq_len, device=decoder_input.device)[None]

        encoder_input = encoder_input + self.positional_encoder.forward(positions[:, :source_ids.shape[1]])
        decoder_input = decoder_input + self.positional_encoder.forward(positions[:, :target_ids.shape[1]])
        enc_atn_fn = multihead_atn_fn
        dec_atn_fn = multihead_atn_fn
        cross_atn_fn = multihead_atn_fn

        encoder_input = self.encoder.forward(
            encoder_input=encoder_input,
            encoder_mask=source_mask,
            atn_fn=enc_atn_fn)
        decoder_output = self.decoder.forward(
            encoder_input=encoder_input,
            decoder_input=decoder_input,
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
        max_seq_len = max(source_ids.shape[1], max_decode_length)
        positions = torch.arange(max_seq_len, device=source_ids.device)[None]
        source_pe = self.positional_encoder.forward(positions[:, :source_ids.size(1)])
        target_pe = self.positional_encoder.forward(positions[:, :max_decode_length])

        encoder_output = self.encoder.forward(
            encoder_input=source_embeddings + source_pe,
            encoder_mask=source_mask,
            atn_fn=multihead_atn_fn)
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
                decoder_input=(self.embedding.embed(beam_paths) + target_pe[:, :current_step]).flatten(0, 1),
                dec_atn_fn=multihead_atn_fn,
                cross_atn_fn=multihead_atn_fn,
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
