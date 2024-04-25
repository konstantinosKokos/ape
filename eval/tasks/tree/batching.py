import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad


def make_collator(device: str = 'cpu'):
    def collate_fn(
            samples: list[tuple[tuple[list[int], list[int]],
                                tuple[list[int], list[int]],
                                list[list[bool]]]]
    ) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor], Tensor]:
        inputs, outputs, masks = zip(*samples)
        input_ids, input_pos = zip(*inputs)
        output_ids, output_pos = zip(*outputs)
        max_len = max(map(len, output_ids))

        def go(xs: list[list[int]]) -> Tensor:
            return pad_sequence([torch.tensor(x, dtype=torch.long) for x in xs], padding_value=-1, batch_first=True)

        input_ids = go(input_ids).to(device)
        output_ids = go(output_ids).to(device)
        input_pos = go(input_pos).to(device)
        output_pos = go(output_pos).to(device)
        input_mask = input_ids.ne(-1)
        output_mask = output_ids.ne(-1)
        causal_mask = torch.stack([
            pad(input=torch.tensor(mask, dtype=torch.bool),
                pad=(0, max_len - len(mask), 0, max_len - len(mask)),
                value=False)
            for mask in masks]).to(device)
        return (input_ids, input_pos, input_mask), (output_ids, output_pos, output_mask), causal_mask
    return collate_fn


def make_flat_collator(device: str = 'cpu'):
    def collate_fn(
            samples: list[tuple[tuple[list[int], list[int]],
                                tuple[list[int], list[int]],
                                list[list[bool]]]]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        inputs, outputs, masks = zip(*samples)
        input_ids, _ = zip(*inputs)
        output_ids, _ = zip(*outputs)
        max_len = max(map(len, output_ids))

        def go(xs: list[list[int]]) -> Tensor:
            return pad_sequence([torch.tensor(x, dtype=torch.long) for x in xs], padding_value=-1, batch_first=True)

        input_ids = go(input_ids).to(device)
        output_ids = go(output_ids).to(device)
        input_mask = input_ids.ne(-1)
        causal_mask = torch.stack([
            pad(input=torch.tensor(mask, dtype=torch.bool),
                pad=(0, max_len - len(mask), 0, max_len - len(mask)),
                value=False)
            for mask in masks]).to(device)
        return input_ids, output_ids, input_mask, causal_mask
    return collate_fn
