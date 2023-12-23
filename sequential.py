import os
import sys
import time

if (slurm_submit_dir := os.environ.get('SLURM_SUBMIT_DIR', default=None)) is not None:
    sys.path.append(os.environ['SLURM_SUBMIT_DIR'])


import argparse
import torch

from eval.tasks.sequence import SequenceRepeat, SequenceCopy, SequenceReverse
from eval.tasks.tree import TreeCopy, TreeReorder, C3, TreeApply
from eval.tasks.tree.batching import make_flat_collator
from eval.tasks.sequence.batching import make_collator
from eval.models.sequential import (Model, SequentialUnitary, SequentialRelative,
                                    SequentialVanilla, SequentialRotary)
from unitaryPE.nn.schedule import make_schedule
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from typing import Literal


def run(
        model: Model,
        task: Literal['copy', 'reverse', 'repeat', 'tree-copy', 'tree-reorder', 'c3', 'apply'],
        vocab_size: int,
        seq_len_mu: int,
        seq_len_var: int,
        num_epochs: int,
        num_layers: tuple[int, int],
        num_heads: int,
        dim: int,
        store_path: str | None,
        regression: Literal['breadth', 'depth'] | None = None,
        seed: int = 42):
    start_time = time.time()
    train_len_dist = Normal(seq_len_mu, seq_len_var)
    test_len_dist = Normal(seq_len_mu, seq_len_var)
    match task:
        case 'copy':
            task = SequenceCopy(vocab_size=vocab_size)
            post_proc = lambda x: x
            collator = make_collator('cuda')
        case 'repeat':
            task = SequenceRepeat(vocab_size=vocab_size, num_repeats=2)
            post_proc = lambda x: x
            collator = make_collator('cuda')
        case 'reverse':
            task = SequenceReverse(vocab_size=vocab_size)
            post_proc = lambda x: x
            collator = make_collator('cuda')
        case 'tree-copy':
            task = TreeCopy(vocab_size=vocab_size, x_projection='breadth', y_projection=regression)
            post_proc = lambda x: x.process()
            collator = make_flat_collator('cuda')
        case 'tree-reorder':
            task = TreeReorder(vocab_size=vocab_size, x_projection='breadth', y_projection=regression)
            post_proc = lambda x: x.process()
            collator = make_flat_collator('cuda')
        case 'c3':
            task = C3(x_projection='breadth', y_projection=regression)
            post_proc = lambda x: x.process()
            collator = make_flat_collator('cuda')
        case 'apply':
            task = TreeApply(x_projection='breadth', y_projection=regression, vocab_size=vocab_size)
            post_proc = lambda x: x.process()
            collator = make_flat_collator('cuda')
        case _:
            raise ValueError

    train_set, dev_set, test_set = task.make_sets(
        distributions=(train_len_dist, train_len_dist, test_len_dist),
        num_samples=(10000, 1000, 1000),
        seed=42)  # keep this fixed for data consistency

    train_dl = DataLoader(list(map(post_proc, train_set)), batch_size=64, collate_fn=collator, shuffle=True)
    dev_dl = DataLoader(list(map(post_proc, dev_set)), batch_size=32, collate_fn=collator, shuffle=False)
    test_dl = DataLoader(list(map(post_proc, test_set)), batch_size=32, collate_fn=collator, shuffle=False)

    torch.manual_seed(seed)

    match model:
        case Model.Relative:
            model = SequentialRelative(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                window_size=seq_len_mu).to('cuda')
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
        case Model.Rotary:
            model = SequentialRotary(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers).to('cuda')

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

    best_epoch, best_dev_acc = None, -1e10
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
        if (epoch > 0 and epoch % 5 == 0) or epoch > num_epochs // 2:
            with torch.no_grad():
                correct_tokens, total_tokens, correct_samples, epoch_loss = 0, 0, 0, 0
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
                    best_epoch, best_dev_acc = epoch, dev_acc
                    torch.save(model.state_dict(), store_path)
                correct_tokens, total_tokens, correct_samples, epoch_loss = 0, 0, 0, 0
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
    print(f'Training took {duration} seconds. Best epoch was {best_epoch}')
    sys.stdout.flush()


def parse_args():
    parser = argparse.ArgumentParser(description='Run a single training iteration')
    parser.add_argument('--model', type=str, choices=['Relative', 'Unitary', 'Sinusoidal', 'Rotary'], help='Type of model to use')
    parser.add_argument('--task', type=str, default='copy', choices=['copy', 'reverse', 'repeat', 'tree-copy', 'tree-reorder', 'c3', 'apply'], help='Which task to train on')
    parser.add_argument('--vocab_size', type=int, default=20, help='Size of vocabulary')
    parser.add_argument('--seq_len_mu', type=int, default=100, help='Mean sequence length')
    parser.add_argument('--seq_len_var', type=int, default=10, help='Sequence length variance')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--num_layers', type=int, nargs=2, default=(2, 2), help='Number of layers for the model')
    parser.add_argument('--dim', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--store_path', type=str, default=None, help='If/where to store the trained model')
    parser.add_argument('--regression', type=str, choices=['breadth', 'depth'], default=None, help='Regression order for tree decoding')
    parser.add_argument('--seed', type=int, default=42, help='The id of the current repetition')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(model=Model[args.model],
        task=args.task,
        num_heads=args.num_heads,
        num_epochs=args.num_epochs,
        vocab_size=args.vocab_size,
        seq_len_mu=args.seq_len_mu,
        seq_len_var=args.seq_len_var,
        dim=args.dim,
        num_layers=args.num_layers,
        store_path=args.store_path,
        regression=args.regression,
        seed=args.seed)
