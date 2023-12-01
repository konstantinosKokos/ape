import os
import sys
import time

if (slurm_submit_dir := os.environ.get('SLURM_SUBMIT_DIR', default=None)) is not None:
    sys.path.append(os.environ['SLURM_SUBMIT_DIR'])


import argparse
import torch

from unitaryPE.tasks.tree import TreeCopy
from unitaryPE.tasks.tree.batching import make_collator
from unitaryPE.models.tree import TreeUnitary, Model
from unitaryPE.neural.schedule import make_schedule
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from typing import Literal


def run(
        model: Model,
        vocab_size: int,
        tree_depth_mu: int,
        tree_depth_var: int,
        num_epochs: int,
        num_layers: tuple[int, int],
        num_heads: int,
        dim: int,
        regression: Literal['breadth', 'depth'],
        store_path: str | None,
        seed: int = 42):
    start_time = time.time()

    task = TreeCopy(vocab_size=vocab_size, x_projection='breadth', y_projection=regression)

    train_depth_dist = Normal(tree_depth_mu, tree_depth_var)
    test_depth_dist = Normal(tree_depth_mu, tree_depth_var)

    train_set, dev_set, test_set = task.make_sets(
        distributions=(train_depth_dist, train_depth_dist, test_depth_dist),
        num_samples=(10000, 1000, 1000),
        seed=42)  # keep this fixed for data consistency

    train_dl = DataLoader([sample.process() for sample in train_set],
                          batch_size=64, collate_fn=make_collator('cuda'), shuffle=True)
    dev_dl = DataLoader([sample.process() for sample in dev_set],
                        batch_size=32, collate_fn=make_collator('cuda'), shuffle=False)
    test_dl = DataLoader([sample.process() for sample in test_set],
                         batch_size=32, collate_fn=make_collator('cuda'), shuffle=False)

    torch.manual_seed(seed)

    match model:
        case Model.Unitary:
            model = TreeUnitary(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                branching_factor=2).to('cuda')
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

    best_epoch, best_dev_acc = None, -1e10
    for epoch in range(num_epochs):
        correct_tokens, total_tokens, correct_samples, epoch_loss = 0, 0, 0, 0
        model.train()
        print(f'{epoch}')
        for ((input_ids, input_pos, input_mask), (output_ids, output_pos, output_mask), causal_mask) in train_dl:
            loss, batch_correct_tokens, batch_total_tokens, batch_correct_samples = model.go_batch(
                input_ids=input_ids,
                input_pos=input_pos,
                input_mask=input_mask,
                output_ids=output_ids,
                output_pos=output_pos,
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
                for ((input_ids, input_pos, input_mask), (output_ids, output_pos, output_mask), causal_mask) in dev_dl:
                    loss, batch_correct_tokens, batch_total_tokens, batch_correct_samples = model.go_batch(
                        input_ids=input_ids,
                        input_pos=input_pos,
                        input_mask=input_mask,
                        output_ids=output_ids,
                        output_pos=output_pos,
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
                for ((input_ids, input_pos, input_mask), (output_ids, output_pos, output_mask), causal_mask) in test_dl:
                    loss, batch_correct_tokens, batch_total_tokens, batch_correct_samples = model.go_batch(
                        input_ids=input_ids,
                        input_pos=input_pos,
                        input_mask=input_mask,
                        output_ids=output_ids,
                        output_pos=output_pos,
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
    parser.add_argument('--model', type=str, choices=['Unitary'], help='Type of model to use')
    parser.add_argument('--regression', type=str, default='breadth', choices=['breadth', 'depth'], help='Decoding order')
    parser.add_argument('--vocab_size', type=int, default=20, help='Size of vocabulary')
    parser.add_argument('--tree_depth_mu', type=int, default=7, help='Mean tree depth')
    parser.add_argument('--tree_depth_var', type=int, default=1, help='Tree depth variance')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--num_layers', type=int, nargs=2, default=(2, 2), help='Number of layers for the model')
    parser.add_argument('--dim', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--store_path', type=str, default=None, help='If/where to store the trained model')
    parser.add_argument('--seed', type=int, default=42, help='The id of the current repetition')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(model=Model[args.model],
        num_heads=args.num_heads,
        num_epochs=args.num_epochs,
        vocab_size=args.vocab_size,
        tree_depth_mu=args.tree_depth_mu,
        tree_depth_var=args.tree_depth_var,
        dim=args.dim,
        num_layers=args.num_layers,
        store_path=args.store_path,
        regression=args.regression,
        seed=args.seed)
