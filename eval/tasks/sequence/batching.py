from torch.nn.utils.rnn import pad_sequence
import torch
from torch import Tensor
from .task import SequentialSample


def make_collator(device: str = 'cpu'):
    def collate_fn(
            samples: list[SequentialSample]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        input_ids = pad_sequence([torch.tensor(sample.x, dtype=torch.long) for sample in samples], batch_first=True, padding_value=-1)
        output_ids = pad_sequence([torch.tensor(sample.y, dtype=torch.long) for sample in samples], batch_first=True, padding_value=-1)
        input_mask = input_ids.ne(-1)
        causal_mask = torch.tril(torch.ones(output_ids.shape[1], output_ids.shape[1], dtype=torch.bool), diagonal=0)
        return (input_ids.to(device),
                output_ids.to(device),
                input_mask.to(device),
                causal_mask[None, :].to(device))
    return collate_fn
