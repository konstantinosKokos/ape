import os
import sys
import time

if (slurm_submit_dir := os.environ.get('SLURM_SUBMIT_DIR', default=None)) is not None:
    sys.path.append(os.environ['SLURM_SUBMIT_DIR'])


import argparse
import torch

from unitaryPE.tasks.sequence import SequenceRepeat, SequenceCopy, SequenceReverse
from unitaryPE.tasks.sequence.batching import make_collator
from unitaryPE.models.sequential import (Model, SequentialUnitary, SequentialRelative, SequentialVanilla)
from unitaryPE.neural.schedule import make_schedule
from torch.distributions import Normal
from torch.utils.data import DataLoader
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
        num_positions: int | None,
        store_path: str | None,
        seed: int = 42):
    start_time = time.time()

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
        seed=42)  # keep this fixed for data consistency

    train_dl = DataLoader(train_set, batch_size=64, collate_fn=make_collator('cuda'), shuffle=True)
    dev_dl = DataLoader(dev_set, batch_size=32, collate_fn=make_collator('cuda'), shuffle=False)
    test_dl = DataLoader(test_set, batch_size=32, collate_fn=make_collator('cuda'), shuffle=False)

    torch.manual_seed(seed)

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
    optim = AdamW(model.parameters(), lr=1)
    scheduler = LambdaLR(
        optimizer=optim,
        lr_lambda=make_schedule(
            warmup_steps=steps_per_epoch * 5,
            warmdown_steps=steps_per_epoch * (num_epochs - 5),
            total_steps=steps_per_epoch * num_epochs,
            min_lr=1e-9,
            max_lr=5e-4,
            init_lr=1e-7))

    best_dev_acc = -1e10
    for epoch in range(num_epochs):
        correct_tokens, total_tokens, correct_samples, epoch_loss = 0, 0, 0, 0
        model.train()
        print(f'{epoch}')
        for (input_ids, output_ids, input_mask, output_mask, causal_mask) in train_dl:
            loss, batch_correct_tokens, batch_total_tokens, batch_correct_samples = model.go_batch(
                input_ids=input_ids,
                output_ids=output_ids,
                input_mask=input_mask,
                output_mask=output_mask,
                causal_mask=causal_mask)
            correct_tokens += batch_correct_tokens
            total_tokens += batch_total_tokens
            correct_samples += batch_correct_samples
            epoch_loss += loss.item()
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
        print(f'Train loss {epoch_loss}')
        print(f'Train acc (token) {correct_tokens/total_tokens}')
        print(f'Train acc (sample) {correct_samples/len(train_set)}')
        model.eval()
        correct_tokens, total_tokens, correct_samples, epoch_loss = 0, 0, 0, 0
        if (epoch > 0 and epoch % 5 == 0) or epoch > num_epochs // 2:
            with torch.no_grad():
                for (input_ids, output_ids, input_mask, output_mask, causal_mask) in dev_dl:
                    loss, batch_correct_tokens, batch_total_tokens, batch_correct_samples = model.go_batch(
                        input_ids=input_ids,
                        output_ids=output_ids,
                        input_mask=input_mask,
                        output_mask=output_mask,
                        causal_mask=causal_mask)
                    correct_tokens += batch_correct_tokens
                    total_tokens += batch_total_tokens
                    correct_samples += batch_correct_samples
                    epoch_loss += loss.item()
                print(f'Dev loss {epoch_loss}')
                print(f'Dev acc (token) {(dev_acc := correct_tokens / total_tokens)}')
                print(f'Dev acc (sample) {correct_samples / len(dev_set)}')
                if dev_acc > best_dev_acc and store_path is not None:
                    best_dev_acc = dev_acc
                    torch.save(model.state_dict(), store_path)
                for (input_ids, output_ids, input_mask, output_mask, causal_mask) in test_dl:
                    loss, batch_correct_tokens, batch_total_tokens, batch_correct_samples = model.go_batch(
                        input_ids=input_ids,
                        output_ids=output_ids,
                        input_mask=input_mask,
                        output_mask=output_mask,
                        causal_mask=causal_mask)
                    correct_tokens += batch_correct_tokens
                    total_tokens += batch_total_tokens
                    correct_samples += batch_correct_samples
                    epoch_loss += loss.item()
                print(f'Test loss {epoch_loss}')
                print(f'Test acc (token) {correct_tokens / total_tokens}')
                print(f'Test acc (sample) {correct_samples / len(test_set)}')
        print('-' * 64)
        sys.stdout.flush()
    duration = time.time() - start_time
    print(f'Training took {duration} seconds.')
    sys.stdout.flush()


def parse_args():
    parser = argparse.ArgumentParser(description='Run a single training iteration')
    parser.add_argument('--model', type=str, choices=['Relative', 'Unitary', 'Sinusoidal'], help='Type of model to use')
    parser.add_argument('--num_repeats', type=int, default=1, help='Number of repeats for SequenceRepeat task')
    parser.add_argument('--reverse', action='store_true', help='Use reverse for SequenceReverse task')
    parser.add_argument('--vocab_size', type=int, default=20, help='Size of vocabulary')
    parser.add_argument('--seq_len_mu', type=int, default=100, help='Mean sequence length')
    parser.add_argument('--seq_len_var', type=int, default=10, help='Sequence length variance')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--num_layers', type=int, nargs=2, default=(2, 2), help='Number of layers for the model')
    parser.add_argument('--dim', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_positions', type=int, default=55, help='Number of positions for the window size')
    parser.add_argument('--store_path', type=str, default=None, help='If/where to store the trained model')
    parser.add_argument('--seed', type=int, default=42, help='The id of the current repetition')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(model=Model[args.model],
        num_heads=args.num_heads,
        num_epochs=args.num_epochs,
        num_positions=args.num_positions,
        num_repeats=args.num_repeats,
        reverse=args.reverse,
        vocab_size=args.vocab_size,
        seq_len_mu=args.seq_len_mu,
        seq_len_var=args.seq_len_var,
        dim=args.dim,
        num_layers=args.num_layers,
        store_path=args.store_path,
        seed=args.seed)
