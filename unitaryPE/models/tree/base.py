from torch.nn.functional import cross_entropy
from torch import Tensor
from abc import abstractmethod, ABC


class Base(ABC):
    @abstractmethod
    def forward(self,
                encoder_ids: Tensor,
                encoder_pos: Tensor,
                encoder_mask: Tensor,
                decoder_ids: Tensor,
                decoder_pos: Tensor,
                decoder_mask: Tensor,
                cross_mask: Tensor) -> Tensor:
        ...

    def go_batch(self,
                 input_ids: Tensor,
                 input_pos: Tensor,
                 input_mask: Tensor,
                 output_ids: Tensor,
                 output_pos: Tensor,
                 output_mask: Tensor,
                 causal_mask: Tensor) -> tuple[Tensor, int, int, int]:
        preds = self.forward(
            encoder_ids=input_ids,
            encoder_pos=input_pos,
            encoder_mask=input_mask,
            decoder_ids=output_ids,
            decoder_pos=output_pos,
            decoder_mask=causal_mask,
            cross_mask=input_mask)

        preds = preds[:, :-1]
        output_ids = output_ids[:, 1:]

        loss = cross_entropy(
            ignore_index=-1,
            input=preds.flatten(0, -2),
            target=output_ids.flatten(),
            reduction='mean')

        output_mask = ~output_mask[:, 1:]
        num_masked_tokens = output_mask.sum()

        sharp = preds.argmax(dim=-1).masked_fill_(output_mask, -1)

        correct_tokens = (sharp.eq(output_ids))
        correct_samples = correct_tokens.all(-1).sum()

        return (loss,
                (correct_tokens.sum() - num_masked_tokens).item(),
                (output_ids.numel() - num_masked_tokens).item(),
                correct_samples.item())
