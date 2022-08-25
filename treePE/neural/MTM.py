import pdb

from ..data.batching import accuracy
from .transformer import Encoder

from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer

from typing import Iterable


class MTM(Encoder):
    def predict_mask(self, content_ids: Tensor, position_ids: Tensor, atn_mask: Tensor, masking_value: int) -> Tensor:
        ctx = self.forward(content_ids, position_ids, atn_mask)
        return self.embedding.invert(ctx[content_ids == masking_value])

    def go_batch(self, batch: tuple[Tensor, Tensor, Tensor, Tensor],
                 masking_value: int,
                 optimizer: Optimizer | None) -> tuple[float, tuple[int, int]]:
        out_ids, in_ids, pos_ids, atn_mask = batch
        preds = self.predict_mask(in_ids, pos_ids, atn_mask, masking_value)
        truths = out_ids[in_ids == masking_value]
        loss = cross_entropy(preds, truths, reduction='mean')

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item(), accuracy(preds.argmax(dim=-1), truths)

    def go_epoch(self, data: Iterable[tuple[Tensor, Tensor, Tensor, Tensor]],
                 masking_value: int,
                 optimizer: Optimizer | None) -> tuple[float, int, int]:
        loss_avg, e_correct, e_total = 0., 0, 0
        for n, batch in enumerate(data):
            b_loss, (b_correct, b_total) = self.go_batch(batch, masking_value, optimizer)
            loss_avg = (loss_avg * n + b_loss) / (n + 1)
            e_correct += b_correct
            e_total += b_total
        return loss_avg, e_correct, e_total
