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
