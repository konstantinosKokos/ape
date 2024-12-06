import torch
from torch.nn.functional import cross_entropy, log_softmax
from torch import Tensor
from abc import abstractmethod, ABC

from ape.nn.attention import AtnFn
from ape.nn.decoder import Decoder
from ape.nn.embedding import InvertibleEmbedding


class Base(ABC):
    decoder: Decoder
    embedding: InvertibleEmbedding
    vocab_size: int
    eos_token_id: int

    @abstractmethod
    def forward_train(
            self,
            source_ids: Tensor,
            source_mask: Tensor,
            target_ids: Tensor,
            target_mask: Tensor) -> Tensor:
        ...

    @abstractmethod
    def forward_dev(
            self,
            source_ids: Tensor,
            source_mask: Tensor,
            causal_mask: Tensor,
            max_decode_length: int,
            beam_width: int,
            alpha: float
    ) -> tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def forward(
            self,
            source_ids: Tensor,
            target_ids: Tensor,
            source_mask: Tensor,
            causal_mask: Tensor,
            reduction: str = 'mean',
            label_smoothing: float = 0.1
    ) -> tuple[Tensor, Tensor]:
        ...

    def get_loss(
            self,
            source_ids: Tensor,
            target_ids: Tensor,
            source_mask: Tensor,
            causal_mask: Tensor,
            reduction: str = 'mean',
            label_smoothing: float = 0.1
    ) -> tuple[Tensor, Tensor]:
        preds = self.forward_train(
            source_ids=source_ids,
            source_mask=source_mask,
            target_ids=target_ids,
            target_mask=causal_mask)

        loss = cross_entropy(
            ignore_index=-1,
            input=preds[:, :-1].flatten(0, -2),
            target=target_ids[:, 1:].flatten(),
            reduction=reduction,
            label_smoothing=label_smoothing)
        numels = target_ids.ne(-1).sum()
        return loss, numels

    def get_acc(
            self,
            source_ids: Tensor,
            target_ids: Tensor,
            source_mask: Tensor,
            causal_mask: Tensor,
            beam_width: int = 1,
            alpha: float = 1.
    ) -> tuple[int, int]:
        preds, _ = self.forward_dev(
            source_ids=source_ids,
            source_mask=source_mask,
            causal_mask=causal_mask,
            max_decode_length=target_ids.size(1) - 1,
            beam_width=beam_width,
            alpha=alpha
        )
        preds = preds[:, 0]
        target_mask = target_ids.ne(-1)
        corr = preds.eq(target_ids).bitwise_and(target_mask).sum().item()
        return corr, target_mask.sum().item()

    def step(
            self,
            encoder_output: Tensor,
            decoder_input: Tensor,
            dec_atn_fn: AtnFn,
            cross_atn_fn: AtnFn,
            source_mask: Tensor,
            decoder_mask: Tensor,
            beam_paths: Tensor,
            beam_scores: Tensor,
            beam_width: int,
            current_step: int,
            alpha: float
    ) -> tuple[Tensor, Tensor]:
        decoder_step = self.decoder.forward(
            encoder_input=encoder_output,
            cross_mask=source_mask,
            decoder_input=decoder_input,
            decoder_mask=decoder_mask[None, :current_step, :current_step],
            self_atn_fn=dec_atn_fn,
            cross_atn_fn=cross_atn_fn)[:, -1]

        decoder_preds = self.embedding.invert(decoder_step)
        decoder_preds = log_softmax(decoder_preds, dim=-1).unflatten(0, (-1, beam_width))

        if current_step == 1:
            decoder_preds[:, 1:] = -1e08

        paths, scores = beam_search(
            predictions=decoder_preds,
            beam_paths=beam_paths,
            beam_scores=beam_scores,
            beam_width=beam_width,
            eos_token_id=self.eos_token_id,
            alpha=alpha)
        return paths, scores


def beam_active(eos_token_id: int, beam_paths: Tensor) -> Tensor:
    return beam_paths.eq(eos_token_id).any(dim=-1).logical_not()


def beam_search(
        predictions: Tensor,    # B x K x V
        beam_paths: Tensor,     # B x K x T
        beam_scores: Tensor,    # B x K
        beam_width: int,
        eos_token_id: int,
        alpha: float = 0.6
) -> tuple[Tensor, Tensor]:
    # Currently active beams
    active_mask = beam_active(eos_token_id, beam_paths)
    # Mask out inactive beams, except for the EOS token
    predictions[active_mask.logical_not()] = -1e08
    predictions = torch.masked_fill(
        input=predictions,
        mask=active_mask.logical_not().unsqueeze(-1) &
             (torch.arange(predictions.size(-1), device=active_mask.device).view(1, 1, -1) == eos_token_id),
        value=0.
    )
    # Get best k predictions for each batch/beam combination
    per_beam_values, per_beam_indices = torch.topk(predictions, k=beam_width, dim=-1)
    # Calculate accumulated scores for each beam path
    accumulated_scores = per_beam_values + beam_scores.unsqueeze(-1)
    # Apply length normalization
    accumulated_scores[active_mask] /= norm_weight(alpha, predictions.size(-1) + 1)
    # Flatten beam dimension
    accumulated_scores = accumulated_scores.flatten(1, -1)
    # Get topk indices
    beam_scores, topk_indices = torch.topk(accumulated_scores, k=beam_width, dim=-1)
    # Revert indexing
    origins = topk_indices // beam_width
    choices = topk_indices % beam_width
    # Construct new paths
    paths = torch.gather(beam_paths, dim=1, index=origins.unsqueeze(-1).expand(-1, -1, beam_paths.size(-1)))
    steps = torch.gather(per_beam_indices, dim=1, index=origins.unsqueeze(-1).expand(-1, -1, beam_width))
    steps = torch.gather(steps, dim=2, index=choices.unsqueeze(-1))
    paths = torch.cat((paths, steps), dim=-1)

    return paths, beam_scores


def make_decoder_mask(length: int, device: torch.device) -> Tensor:
    return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=0)


def norm_weight(alpha: float, y: int) -> float:
    return ((y+5)/(y+4))**alpha
