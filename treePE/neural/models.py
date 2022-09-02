import pdb

from ..data.batching import accuracy
from .transformer import Transformer, Encoder
from torch import Tensor, cat
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from typing import Iterable


class Tree2Tree(Transformer):
    def predict(self,
                encoder_ids: Tensor, encoder_positions: Tensor, encoder_mask: Tensor,
                decoder_ids: list[Tensor], decoder_positions: Tensor, decoder_mask: list[Tensor],
                cross_mask: Tensor, mask_idx: int) -> Tensor:
        enc_ctx = self.encode(encoder_ids, encoder_positions, encoder_mask)
        preds = cat(
                [self.decode(enc_ctx, step_ids, decoder_positions, step_mask, cross_mask)[step_ids == mask_idx]
                 for step_ids, step_mask in zip(decoder_ids, decoder_mask)])
        return self.decoder.embedding.invert(preds)

    def go_batch(self,
                 batch: tuple[Tensor, Tensor, Tensor, list[Tensor], Tensor, list[Tensor], Tensor, Tensor],
                 mask_idx: int,
                 pad_idx: int,
                 opt_schedule: tuple[Optimizer, LambdaLR] | None) -> tuple[float, tuple[int, int]]:
        encoder_ids, encoder_pos, encoder_mask, decoder_ids, decoder_pos, decoder_mask, cross_mask, true_ids = batch
        preds = self.predict(encoder_ids, encoder_pos, encoder_mask,
                             decoder_ids, decoder_pos, decoder_mask,
                             cross_mask, mask_idx)
        truths = true_ids[true_ids != pad_idx]
        loss = cross_entropy(preds, truths, reduction='mean')

        if opt_schedule is not None:
            optimizer, scheduler = opt_schedule
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        return loss.item(), accuracy(preds.argmax(dim=-1), truths)

    def go_epoch(self,
                 data: Iterable[tuple[Tensor, Tensor, Tensor, list[Tensor], Tensor, list[Tensor], Tensor, Tensor]],
                 mask_idx: int,
                 pad_idx: int,
                 opt_schedule: tuple[Optimizer, LambdaLR] | None, ) -> tuple[float, int, int]:
        loss_avg, e_correct, e_total = 0., 0, 0
        for n, batch in enumerate(data):
            b_loss, (b_correct, b_total) = self.go_batch(batch, mask_idx, pad_idx, opt_schedule)
            loss_avg = (loss_avg * n + b_loss) / (n + 1)
            e_correct += b_correct
            e_total += b_total
        return loss_avg, e_correct, e_total


class MaskedTreeModeling(Encoder):
    def predict_mask(self, content_ids: Tensor, position_ids: Tensor, atn_mask: Tensor, masking_value: int) -> Tensor:
        ctx = self.forward(content_ids, position_ids, atn_mask)
        return self.embedding.invert(ctx[content_ids == masking_value])

    def go_batch(self, batch: tuple[Tensor, Tensor, Tensor, Tensor],
                 masking_value: int,
                 opt_schedule: tuple[Optimizer, LambdaLR] | None) -> tuple[float, tuple[int, int]]:
        out_ids, in_ids, pos_ids, atn_mask = batch
        preds = self.predict_mask(in_ids, pos_ids, atn_mask, masking_value)
        truths = out_ids[in_ids == masking_value]
        loss = cross_entropy(preds, truths, reduction='mean')

        if opt_schedule is not None:
            optimizer, scheduler = opt_schedule
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        return loss.item(), accuracy(preds.argmax(dim=-1), truths)

    def go_epoch(self, data: Iterable[tuple[Tensor, Tensor, Tensor, Tensor]],
                 masking_value: int,
                 opt_schedule: tuple[Optimizer, LambdaLR] | None, ) -> tuple[float, int, int]:
        loss_avg, e_correct, e_total = 0., 0, 0
        for n, batch in enumerate(data):
            b_loss, (b_correct, b_total) = self.go_batch(batch, masking_value, opt_schedule)
            loss_avg = (loss_avg * n + b_loss) / (n + 1)
            e_correct += b_correct
            e_total += b_total
        return loss_avg, e_correct, e_total
