import os
import sys
import time

if (slurm_submit_dir := os.environ.get('SLURM_SUBMIT_DIR', default=None)) is not None:
    sys.path.append(os.environ['SLURM_SUBMIT_DIR'])


import argparse
import torch

from eval.tasks.tree import TreeCopy, TreeReorder, C3, TreeApply
from eval.tasks.tree.batching import make_collator
from eval.models.tree import TreeUnitary, ShivQuirk, Model
from unitaryPE.nn.schedule import make_schedule
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
        task: Literal['copy', 'reorder', 'c3', 'apply'],
        num_epochs: int,
        num_layers: tuple[int, int],
        num_heads: int,
        dim: int,
        regression: Literal['breadth', 'depth'],
        store_path: str | None,
        seed: int = 42):
    start_time = time.time()

    match task:
        case 'copy':
            task = TreeCopy(vocab_size=vocab_size, x_projection='breadth', y_projection=regression, sos_token_id=0, eos_token_id=-1)
        case 'reorder':
            task = TreeReorder(vocab_size=vocab_size, x_projection='breadth', y_projection=regression, sos_token_id=0, eos_token_id=-1)
        case 'c3':
            task = C3(x_projection='breadth', y_projection=regression, sos_token_id=0, eos_token_id=-1)
        case 'apply':
            task = TreeApply(x_projection='breadth', y_projection=regression, vocab_size=vocab_size, sos_token_id=0, eos_token_id=-1)
        case _:
            raise ValueError

    train_depth_dist = Normal(tree_depth_mu, tree_depth_var)
    test_depth_dist = Normal(tree_depth_mu, tree_depth_var)

    train_set, dev_set, _ = task.make_sets(
        distributions=(train_depth_dist, train_depth_dist, test_depth_dist),
        num_samples=(6000, 2000, 2000),
        seed=42)  # keep this fixed for data consistency
    print(sum(t.x.numel() for t in train_set)/len(train_set))
    print(sum(t.y.numel() for t in train_set) / len(train_set))

    train_dl = DataLoader([sample.process() for sample in train_set], # noqa
                          batch_size=64, collate_fn=make_collator('cuda'), shuffle=True)
    dev_dl = DataLoader([sample.process() for sample in dev_set],  # noqa
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
        case Model.ShivQuirk:
            model = ShivQuirk(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                branching_factor=2,
                max_depth=tree_depth_mu + tree_depth_var).to('cuda')
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

    best_epoch, best_dev_loss = None, 1e10
    for epoch in range(num_epochs):
        epoch_loss = 0.
        model.train()
        print(f'{epoch}')
        for ((input_ids, input_pos, input_mask), (output_ids, output_pos, output_mask), causal_mask) in train_dl:
            loss = model.get_loss(
                input_ids=input_ids,
                input_pos=input_pos,
                input_mask=input_mask,
                output_ids=output_ids,
                output_pos=output_pos,
                causal_mask=causal_mask)
            epoch_loss += loss.item()
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
        print(f'Train loss {epoch_loss}')
        model.eval()
        epoch_loss = 0.
        if (epoch > 0 and epoch % 5 == 0) or epoch > num_epochs // 2:
            with torch.no_grad():
                for ((input_ids, input_pos, input_mask), (output_ids, output_pos, _), causal_mask) in dev_dl:
                    loss = model.get_loss(
                        input_ids=input_ids,
                        input_pos=input_pos,
                        input_mask=input_mask,
                        output_ids=output_ids,
                        output_pos=output_pos,
                        causal_mask=causal_mask)
                    epoch_loss += loss.item()
                print(f'Dev loss {epoch_loss}')
                if epoch_loss < best_dev_loss and store_path is not None:
                    best_epoch, best_dev_loss = epoch, epoch_loss
                    torch.save(model.state_dict(), store_path)
    duration = time.time() - start_time
    print(f'Training took {duration} seconds. Best epoch was {best_epoch}')
    sys.stdout.flush()


def evaluate(
        model: Model,
        vocab_size: int,
        tree_depth_mu: int,
        tree_depth_var: int,
        task: Literal['copy', 'reorder', 'c3', 'apply'],
        num_layers: tuple[int, int],
        num_heads: int,
        dim: int,
        regression: Literal['breadth', 'depth'],
        store_path: str | None,
        seed: int = 42):

    match task:
        case 'copy':
            task = TreeCopy(vocab_size=vocab_size, x_projection='breadth', y_projection=regression, sos_token_id=0, eos_token_id=-1)
        case 'reorder':
            task = TreeReorder(vocab_size=vocab_size, x_projection='breadth', y_projection=regression, sos_token_id=0, eos_token_id=-1)
        case 'c3':
            task = C3(x_projection='breadth', y_projection=regression, sos_token_id=0, eos_token_id=-1)
        case 'apply':
            task = TreeApply(x_projection='breadth', y_projection=regression, vocab_size=vocab_size, sos_token_id=0, eos_token_id=-1)
        case _:
            raise ValueError

    train_depth_dist = Normal(tree_depth_mu, tree_depth_var)
    test_depth_dist = Normal(tree_depth_mu, tree_depth_var)

    _, _, test_set = task.make_sets(
        distributions=(train_depth_dist, train_depth_dist, test_depth_dist),
        num_samples=(6000, 2000, 2000),
        seed=42)  # keep this fixed for data consistency

    test_dl = DataLoader([sample.process() for sample in test_set],  # noqa
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
        case Model.ShivQuirk:
            model = ShivQuirk(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                branching_factor=2,
                max_depth=tree_depth_mu + tree_depth_var).to('cuda')
        case _:
            raise ValueError

    model.load_state_dict(torch.load(store_path, map_location='cuda'), strict=True)
    model.eval()
    loss = torch.tensor([], device='cuda', dtype=torch.float)
    with torch.no_grad():
        for ((input_ids, input_pos, input_mask), (output_ids, output_pos, _), causal_mask) in test_dl:
            pad_mask = output_ids[:, :-1].flatten().ne(-1)
            xe = model.get_loss(
                input_ids=input_ids,
                input_pos=input_pos,
                input_mask=input_mask,
                output_ids=output_ids,
                output_pos=output_pos,
                causal_mask=causal_mask,
                reduction='none',
                label_smoothing=0.
            )[pad_mask]
            loss = torch.cat((loss, xe), dim=-1)
        ppl = torch.exp(torch.mean(loss)).item()
    print(f'{ppl=}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run a single training iteration')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--model', type=str, required=True, choices=['Unitary', 'ShivQuirk'], help='Type of model to use')
    parser.add_argument('--regression', type=str, required=True, choices=['breadth', 'depth'], help='Decoding order')
    parser.add_argument('--vocab_size', type=int, default=20, help='Size of vocabulary')
    parser.add_argument('--tree_depth_mu', type=int, default=7, help='Mean tree depth')
    parser.add_argument('--tree_depth_var', type=int, default=1, help='Tree depth variance')
    parser.add_argument('--task', type=str, required=True, choices=['copy', 'reorder', 'c3', 'apply'], help='Which task to train on')
    parser.add_argument('--num_epochs', type=int, default=400, help='Number of training epochs')
    parser.add_argument('--num_layers', type=int, nargs=2, default=(2, 2), help='Number of layers for the model')
    parser.add_argument('--dim', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--store_path', type=str, default=None, help='If/where to store the trained model')
    parser.add_argument('--seed', type=int, default=42, help='The id of the current repetition')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.eval:
        evaluate(
            model=Model[args.model],
            task=args.task,
            num_heads=args.num_heads,
            vocab_size=args.vocab_size,
            tree_depth_mu=args.tree_depth_mu,
            tree_depth_var=args.tree_depth_var,
            dim=args.dim,
            num_layers=args.num_layers,
            store_path=args.store_path,
            regression=args.regression,
            seed=args.seed)
    else:
        run(
            model=Model[args.model],
            task=args.task,
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
