from torch.nn.functional import cross_entropy


class Base:
    def go_batch(self, input_ids, output_ids, input_mask, output_mask, causal_mask):
        preds = self.forward(
            encoder_ids=input_ids,
            encoder_mask=input_mask,
            decoder_ids=output_ids,
            decoder_mask=causal_mask,
            cross_mask=input_mask)
        preds = preds[:, :-1][output_mask[:, 1:]]
        output_ids = output_ids[:, 1:][output_mask[:, 1:]]
        loss = cross_entropy(
            input=preds,
            target=output_ids,
            reduction='mean')
        correct = sum(preds.argmax(dim=-1).eq(output_ids)).item()
        total = output_ids.numel()
        return loss, correct, total
