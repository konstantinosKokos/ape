import torch

from unitaryPE.tasks.sequence import SequenceRepeat, SequenceCopy, SequenceReverse
from unitaryPE.tasks.sequence.batching import make_collator
from unitaryPE.models.sequential import (Model, SequentialUnitary, SequentialRelative, SequentialVanilla)
from unitaryPE.neural.schedule import make_schedule
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def run(
        model: Model,
        reverse: bool,
        num_repeats: int,
        vocab_size: int,
        seq_len_mu: int,
        seq_len_var: int,
        num_epochs: int,
        num_layers: tuple[int, int],
        num_heads: int,
        dim: int,
        num_positions: int | None):
    if num_repeats > 1 and reverse:
        raise ValueError('No repeat-reverse')

    if num_repeats > 1:
        task = SequenceRepeat(vocab_size=vocab_size, num_repeats=num_repeats)
    elif reverse:
        task = SequenceReverse(vocab_size=vocab_size)
    else:
        task = SequenceCopy(vocab_size=vocab_size)

    train_len_dist = Normal(seq_len_mu, seq_len_var)
    test_len_dist = Normal(seq_len_mu, seq_len_var)

    train_set, dev_set, test_set = task.make_sets(
        distributions=(train_len_dist, train_len_dist, test_len_dist),
        num_samples=(10000, 1000, 1000),
        seed=42)

    train_dl = DataLoader(train_set, batch_size=64, collate_fn=make_collator('cuda'), shuffle=True)
    dev_dl = DataLoader(dev_set, batch_size=32, collate_fn=make_collator('cuda'), shuffle=False)
    test_dl = DataLoader(test_set, batch_size=32, collate_fn=make_collator('cuda'), shuffle=False)

    match model:
        case Model.Relative:
            model = SequentialRelative(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                window_size=num_positions).to('cuda')
        case Model.Unitary:
            model = SequentialUnitary(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers).to('cuda')
        case Model.Sinusoidal:
            model = SequentialVanilla(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers).to('cuda')
        case _:
            raise ValueError

    steps_per_epoch = len(train_dl)
    optim = AdamW(model.parameters(), lr=5e-4)
    scheduler = LambdaLR(
        optimizer=optim,
        lr_lambda=make_schedule(
            warmup_steps=steps_per_epoch * 5,
            warmdown_steps=steps_per_epoch * (num_epochs - 5),
            total_steps=steps_per_epoch * num_epochs,
            min_lr=1e-3, max_lr=1))

    for epoch in range(num_epochs):
        print(scheduler.get_last_lr())
        correct, total, epoch_loss = 0, 0, 0
        model.train()
        print(f'{epoch}')
        for (input_ids, output_ids, input_mask, output_mask, causal_mask) in train_dl:
            preds = model.forward(
                encoder_ids=input_ids,
                encoder_mask=input_mask,
                decoder_ids=output_ids,
                decoder_mask=causal_mask,
                cross_mask=input_mask)
            preds = preds[:, :-1].flatten(0, -2)
            output_ids = output_ids[:, 1:].flatten()
            pad_mask = output_ids != -1
            preds = preds[pad_mask]
            output_ids = output_ids[pad_mask]
            loss = cross_entropy(
                input=preds,
                target=output_ids,
                reduction='mean')
            if isinstance(model, SequentialUnitary):
                loss += model.positional_encoder.penalty()
            correct += sum(preds.argmax(dim=-1).eq(output_ids)).item()
            total += output_ids.numel()
            epoch_loss += loss.item()
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
        print(f'train: {correct/total} -- ({epoch_loss})')
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for (input_ids, output_ids, input_mask, output_mask, causal_mask) in dev_dl:
                preds = model.forward(
                    encoder_ids=input_ids,
                    encoder_mask=input_mask,
                    decoder_ids=output_ids,
                    decoder_mask=causal_mask,
                    cross_mask=input_mask)
                preds = preds[:, :-1].flatten(0, -2)
                output_ids = output_ids[:, 1:].flatten()
                pad_mask = output_ids != -1
                preds = preds[pad_mask]
                output_ids = output_ids[pad_mask]
                correct += sum(preds.argmax(dim=-1).eq(output_ids)).item()
                total += output_ids.numel()
            print(f'dev: {correct/total}')
            for (input_ids, output_ids, input_mask, output_mask, causal_mask) in test_dl:
                preds = model.forward(
                    encoder_ids=input_ids,
                    encoder_mask=input_mask,
                    decoder_ids=output_ids,
                    decoder_mask=causal_mask,
                    cross_mask=input_mask)
                preds = preds[:, :-1].flatten(0, -2)
                output_ids = output_ids[:, 1:].flatten()
                pad_mask = output_ids != -1
                preds = preds[pad_mask]
                output_ids = output_ids[pad_mask]
                correct += sum(preds.argmax(dim=-1).eq(output_ids)).item()
                total += output_ids.numel()
            print(f'test: {correct / total}')


if __name__ == '__main__':
    run(model=Model.Relative,
        num_repeats=1,
        reverse=True,
        vocab_size=20,
        seq_len_mu=100,
        seq_len_var=10,
        num_epochs=200,
        num_layers=(2, 2),
        dim=512,
        num_heads=8,
        num_positions=50)
