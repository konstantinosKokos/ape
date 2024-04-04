import torch
from torch.nn.functional import cross_entropy
from torch import Tensor
from abc import abstractmethod, ABC


class Base(ABC):
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
            max_decode_length: int,
            beam_width: int) -> tuple[Tensor, Tensor]:
        ...

    def go_batch(
            self,
            source_ids: Tensor,
            target_ids: Tensor,
            source_mask: Tensor,
            causal_mask: Tensor,
            reduction: str = 'mean'
    ) -> Tensor:
        preds = self.forward_train(
            source_ids=source_ids,
            source_mask=source_mask,
            target_ids=target_ids,
            target_mask=causal_mask)

        return cross_entropy(
            ignore_index=-1,
            input=preds[:, :-1].flatten(0, -2),
            target=target_ids[:, 1:].flatten(),
            reduction=reduction)


def beam_active(eos_token_id: int, beam_paths: Tensor) -> Tensor:
    return beam_paths.eq(eos_token_id).any(dim=-1).logical_not()


def beam_search(
        predictions: Tensor,    # B x K x V
        beam_paths: Tensor,     # B x K x T
        beam_scores: Tensor,    # B x K
        beam_width: int,
        eos_token_id: int,
):
    # Currently active beams
    active_mask = beam_active(eos_token_id, beam_paths)
    # Mask out inactive beams, except for the EOS token
    predictions[active_mask.logical_not()] = -1e08
    predictions[active_mask.logical_not().unsqueeze(-1) & (torch.arange(32000).view(1, -1) == eos_token_id)] = 0.
    # Get best k predictions for each batch/beam combination
    per_beam_values, per_beam_indices = torch.topk(predictions, k=beam_width, dim=-1)
    # Calculate accumulated scores for each beam path
    accumulated_scores = per_beam_values + beam_scores.unsqueeze(-1)
    # Flatten beam dimension
    accumulated_scores = accumulated_scores.flatten(1, -1)
    # Get topk indices
    topk_values, topk_indices = torch.topk(accumulated_scores, k=beam_width, dim=-1)
    # Revert indexing
    origins = topk_indices // beam_width
    choices = topk_indices % beam_width

    # # Construct new paths
    paths = torch.gather(beam_paths, dim=1, index=origins.unsqueeze(-1).expand(-1, -1, beam_paths.size(-1)))
    steps = torch.gather(per_beam_indices, dim=1, index=origins.unsqueeze(-1).expand(-1, -1, beam_width))
    steps = torch.gather(steps, dim=2, index=choices.unsqueeze(-1))
    paths = torch.cat((paths, steps), dim=-1)
    beam_scores = topk_values

    return paths, beam_scores


def make_decoder_mask(length: int, device: torch.device) -> Tensor:
    return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=0)
