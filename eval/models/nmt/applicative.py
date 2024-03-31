from .base import Base

from torch.nn import Module
from torch.nn.functional import log_softmax
from torch import Tensor
import torch

from unitaryPE.nn.encoder import Encoder
from unitaryPE.nn.decoder import Decoder
from unitaryPE.nn.position import UnitarySequential
from unitaryPE.nn.embedding import InvertibleEmbedding


class MTUnitary(Module, Base):
    def __init__(
            self,
            vocab_size: int,
            dim: int,
            num_heads: int,
            num_layers: tuple[int, int],
            sos_token_id: int,
            eos_token_id: int):
        super(MTUnitary, self).__init__()
        self.encoder = Encoder(num_heads=num_heads, num_layers=num_layers[0], dim=dim)
        self.decoder = Decoder(num_heads=num_heads, num_layers=num_layers[1], dim=dim)
        self.source_pe = UnitarySequential(dim=dim // num_heads, num_heads=num_heads)
        self.target_pe = UnitarySequential(dim=dim // num_heads, num_heads=num_heads)
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

        self.source_pe.precompute(source_ids.size(1))
        self.target_pe.precompute(target_ids.size(1))
        source_positions = torch.arange(source_ids.size(1), device=source_ids.device)
        target_positions = torch.arange(target_ids.size(1), device=target_ids.device)
        source_maps = self.source_pe.forward(source_positions[None])
        target_maps = self.target_pe.forward(target_positions[None])

        mediator = make_mediator(max(source_maps.size(1), target_maps.size(1)), device=source_ids.device)
        enc_atn_fn = self.source_pe.adjust_attention(
            q_maps=source_maps,
            k_maps=source_maps,
            mediator=(mediator[:, :source_ids.size(1), :source_ids.size(1)], True))
        dec_atn_fn = self.source_pe.adjust_attention(
            q_maps=target_maps,
            k_maps=target_maps,
            mediator=(mediator[:, :target_ids.size(1), :target_ids.size(1)], True))
        cross_atn_fn = self.source_pe.adjust_attention(
            q_maps=target_maps,
            k_maps=source_maps,
            mediator=(mediator[:, :target_ids.size(1), :source_ids.size(1)], True))

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
            max_decode_length: int,
            beam_width: int) -> tuple[Tensor, Tensor]:
        source_embeddings = self.embedding.embed(source_ids)
        source_positions = torch.arange(source_ids.size(1), device=source_ids.device)
        target_positions = torch.arange(max_decode_length, device=source_ids.device)
        source_maps = self.source_pe.forward(source_positions)
        target_maps = self.target_pe.forward(target_positions)

        mediator = make_mediator(max(source_ids.size(1), max_decode_length), device=source_ids.device)

        enc_atn_fn = self.source_pe.adjust_attention(
            q_maps=source_maps[None],
            k_maps=source_maps[None],
            mediator=(mediator[:, :source_ids.size(1), :source_ids.size(1)], True))
        encoder_output = self.encoder.forward(
            encoder_input=source_embeddings,
            encoder_mask=source_mask,
            atn_fn=enc_atn_fn)

        decoding: bool = True
        beam_paths = torch.ones(source_embeddings.size(0), beam_width, 1, dtype=torch.long, device=source_ids.device)
        beam_paths *= self.sos_token_id
        beam_scores = torch.zeros(source_embeddings.size(0), beam_width, device=source_ids.device, dtype=torch.float)
        decoder_mask = make_decoder_mask(max_decode_length, source_ids.device)
        current_step: int = 0

        while decoding:
            beam_paths, beam_scores = self.step(
                encoder_output=encoder_output,
                source_maps=source_maps,
                source_mask=source_mask,
                target_maps=target_maps,
                decoder_mask=decoder_mask,
                beam_paths=beam_paths,
                beam_scores=beam_scores,
                beam_width=beam_width,
                current_step=(current_step := current_step + 1),
                mediator=mediator)
            decoding = beam_active(self.eos_token_id, beam_paths) and current_step < max_decode_length
        return beam_paths, beam_scores

    def step(
            self,
            encoder_output: Tensor,
            source_maps: Tensor,
            source_mask: Tensor,
            beam_paths: Tensor,
            target_maps: Tensor,
            decoder_mask: Tensor,
            beam_scores: Tensor,
            beam_width: int,
            current_step: int,
            mediator: Tensor) -> tuple[Tensor, Tensor]:
        dec_atn_fn = self.source_pe.adjust_attention(
            q_maps=target_maps[None, :current_step],
            k_maps=target_maps[None, :current_step],
            mediator=(mediator[:, :current_step, :current_step], True))
        cross_atn_fn = self.source_pe.adjust_attention(
            q_maps=target_maps[None, :current_step],
            k_maps=source_maps[None],
            mediator=(mediator[:, :current_step, :encoder_output.size(1)], True))
        decoder_step = self.decoder.forward(
            encoder_input=encoder_output.repeat(beam_width, 1, 1),
            cross_mask=source_mask.repeat(beam_width, 1),
            decoder_input=self.embedding.embed(beam_paths).flatten(0, 1),
            decoder_mask=decoder_mask[None, :current_step, :current_step],
            self_atn_fn=dec_atn_fn,
            cross_atn_fn=cross_atn_fn)[:, -1]

        decoder_preds = self.embedding.invert(decoder_step)
        decoder_preds = log_softmax(decoder_preds, dim=-1).view(-1, beam_width, self.vocab_size)
        paths, scores = beam_search(
            predictions=decoder_preds,
            beam_paths=beam_paths,
            beam_scores=beam_scores,
            beam_width=beam_width)
        return paths, scores


def beam_active(eos_token_id: int, beam_paths: Tensor) -> bool:
    return beam_paths.ne(eos_token_id).any().cpu().tolist()  # noqa


def beam_search(
        predictions: Tensor,    # B x K x V
        beam_paths: Tensor,     # B x K x T
        beam_scores: Tensor,    # B x K
        beam_width: int
):
    # Get best k predictions for each batch/beam combination
    per_beam_values, per_beam_indices = torch.topk(predictions, k=beam_width, dim=-1)
    # Calculate accumulated scores for each beam path
    accumulated_scores = per_beam_values + beam_scores.unsqueeze(-1)
    # Flatten beam dimension
    accumulated_scores = accumulated_scores.view(-1, beam_width ** 2)
    # Get topk indices
    topk_values, topk_indices = torch.topk(accumulated_scores, k=beam_width, dim=-1)
    # Revert indexing
    origins = topk_indices // beam_width
    choices = topk_indices % beam_width
    # Construct new paths
    paths = torch.gather(beam_paths, dim=1, index=origins.unsqueeze(-1).expand(-1, -1, beam_paths.size(-1)))
    steps = torch.gather(per_beam_indices, dim=2, index=choices.unsqueeze(-1))
    paths = torch.cat((paths, steps), dim=-1)
    beam_scores = topk_values
    return paths, beam_scores


def make_decoder_mask(length: int, device: torch.device) -> Tensor:
    return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=0)


def make_mediator(size: int, device: torch.device) -> Tensor:
    positions = torch.arange(size, device=device)[None, :]
    distances = (positions[:, :, None] - positions[:, None]).unsqueeze(-1).unsqueeze(-1)
    mediator = (0.98 ** distances.abs())
    return mediator
