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

    def get_loss(self,
                 input_ids: Tensor,
                 input_pos: Tensor,
                 input_mask: Tensor,
                 output_ids: Tensor,
                 output_pos: Tensor,
                 causal_mask: Tensor,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.1
                 ) -> Tensor:
        preds = self.forward(
            encoder_ids=input_ids,
            encoder_pos=input_pos,
            encoder_mask=input_mask,
            decoder_ids=output_ids,
            decoder_pos=output_pos,
            decoder_mask=causal_mask,
            cross_mask=input_mask)

        return cross_entropy(
            ignore_index=-1,
            input=preds[:, :-1].flatten(0, -2),
            target=output_ids[:, 1:].flatten(),
            reduction=reduction,
            label_smoothing=label_smoothing
        )
