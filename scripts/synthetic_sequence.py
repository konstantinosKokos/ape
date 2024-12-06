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
from eval.models.nmt import Model, MTAlgebraic, MTRelative, MTVanilla, MTRotary, MTAbsolute
from ape.nn.schedule import make_schedule
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from typing import Literal


def train(
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
            task = TreeCopy(vocab_size=vocab_size, x_projection='breadth', y_projection=regression, sos_token_id=0, eos_token_id=-1)  # noqa
            post_proc = lambda x: x.process()
            collator = make_flat_collator('cuda')
        case 'tree-reorder':
            task = TreeReorder(vocab_size=vocab_size, x_projection='breadth', y_projection=regression, sos_token_id=0, eos_token_id=-1)  # noqa
            post_proc = lambda x: x.process()
            collator = make_flat_collator('cuda')
        case 'c3':
            task = C3(x_projection='breadth', y_projection=regression, sos_token_id=0, eos_token_id=-1)  # noqa
            post_proc = lambda x: x.process()
            collator = make_flat_collator('cuda')
        case 'apply':
            task = TreeApply(x_projection='breadth', y_projection=regression, vocab_size=vocab_size, sos_token_id=0, eos_token_id=-1)  # noqa
            post_proc = lambda x: x.process()
            collator = make_flat_collator('cuda')
        case _:
            raise ValueError

    train_set, dev_set, test_set = task.make_sets(
        distributions=(train_len_dist, train_len_dist, test_len_dist),
        num_samples=(6000, 2000, 2000),
        seed=42)  # keep this fixed for data consistency

    train_dl = DataLoader(list(map(post_proc, train_set)), batch_size=64, collate_fn=collator, shuffle=True)  # noqa
    dev_dl = DataLoader(list(map(post_proc, dev_set)), batch_size=32, collate_fn=collator, shuffle=False)  # noqa
    test_dl = DataLoader(list(map(post_proc, test_set)), batch_size=32, collate_fn=collator, shuffle=False)  # noqa

    torch.manual_seed(seed)

    match model:
        case Model.Relative:
            model = MTRelative(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                window_size=seq_len_mu,
                sos_token_id=task.sos_token_id,
                eos_token_id=task.eos_token_id
            ).to('cuda')
        case Model.Unitary:
            model = MTAlgebraic(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                sos_token_id=task.sos_token_id,
                eos_token_id=task.eos_token_id
            ).to('cuda')
        case Model.Sinusoidal:
            model = MTVanilla(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                sos_token_id=task.sos_token_id,
                eos_token_id=task.eos_token_id
            ).to('cuda')
        case Model.Rotary:
            model = MTRotary(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                sos_token_id=task.sos_token_id,
                eos_token_id=task.eos_token_id
            ).to('cuda')
        case Model.Absolute:
            model = MTAbsolute(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                sos_token_id=task.sos_token_id,
                eos_token_id=task.eos_token_id,
                num_positions=seq_len_mu
            ).to('cuda')

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
        for (source_ids, target_ids, source_mask, causal_mask) in train_dl:
            loss, _ = model.get_loss(
                source_ids=source_ids,
                source_mask=source_mask,
                target_ids=target_ids,
                causal_mask=causal_mask,
                label_smoothing=0.)
            epoch_loss += loss.item()
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
        print(f'Train loss {epoch_loss}')
        model.eval()
        if (epoch > 0 and epoch % 5 == 0) or epoch > num_epochs // 2:
            with torch.no_grad():
                epoch_loss = 0.
                for (source_ids, target_ids, source_mask, causal_mask) in dev_dl:
                    loss, _ = model.get_loss(
                        source_ids=source_ids,
                        source_mask=source_mask,
                        target_ids=target_ids,
                        causal_mask=causal_mask,
                        label_smoothing=0.
                    )
                    epoch_loss += loss.item()
                print(f'Dev loss {epoch_loss}')
                if epoch_loss < best_dev_loss and store_path is not None:
                    best_epoch, best_dev_loss = epoch, epoch_loss
                    torch.save(model.state_dict(), store_path)
        print('-' * 64)
        sys.stdout.flush()
    duration = time.time() - start_time
    print(f'Training took {duration} seconds. Best epoch was {best_epoch}')
    sys.stdout.flush()


def evaluate(
        model: Model,
        task: Literal['copy', 'reverse', 'repeat', 'tree-copy', 'tree-reorder', 'c3', 'apply'],
        vocab_size: int,
        seq_len_mu: int,
        seq_len_var: int,
        num_layers: tuple[int, int],
        num_heads: int,
        dim: int,
        store_path: str | None,
        regression: Literal['breadth', 'depth'] | None = None):
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
            task = TreeCopy(vocab_size=vocab_size, x_projection='breadth', y_projection=regression, sos_token_id=0, eos_token_id=-1)  # noqa
            post_proc = lambda x: x.process()
            collator = make_flat_collator('cuda')
        case 'tree-reorder':
            task = TreeReorder(vocab_size=vocab_size, x_projection='breadth', y_projection=regression, sos_token_id=0, eos_token_id=-1)  # noqa
            post_proc = lambda x: x.process()
            collator = make_flat_collator('cuda')
        case 'c3':
            task = C3(x_projection='breadth', y_projection=regression, sos_token_id=0, eos_token_id=-1)  # noqa
            post_proc = lambda x: x.process()
            collator = make_flat_collator('cuda')
        case 'apply':
            task = TreeApply(x_projection='breadth', y_projection=regression, vocab_size=vocab_size, sos_token_id=0, eos_token_id=-1)  # noqa
            post_proc = lambda x: x.process()
            collator = make_flat_collator('cuda')
        case _:
            raise ValueError

    _, _, test_set = task.make_sets(
        distributions=(train_len_dist, train_len_dist, test_len_dist),
        num_samples=(6000, 2000, 2000),
        seed=42)  # keep this fixed for data consistency

    test_dl = DataLoader(list(map(post_proc, test_set)), batch_size=32, collate_fn=collator, shuffle=False)  # noqa

    match model:
        case Model.Relative:
            model = MTRelative(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                window_size=seq_len_mu,
                sos_token_id=task.sos_token_id,
                eos_token_id=task.eos_token_id
            ).to('cuda')
        case Model.Unitary:
            model = MTAlgebraic(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                sos_token_id=task.sos_token_id,
                eos_token_id=task.eos_token_id
            ).to('cuda')
        case Model.Sinusoidal:
            model = MTVanilla(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                sos_token_id=task.sos_token_id,
                eos_token_id=task.eos_token_id
            ).to('cuda')
        case Model.Rotary:
            model = MTRotary(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                sos_token_id=task.sos_token_id,
                eos_token_id=task.eos_token_id
            ).to('cuda')
        case Model.Absolute:
            model = MTAbsolute(
                vocab_size=vocab_size + 2,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                sos_token_id=task.sos_token_id,
                eos_token_id=task.eos_token_id,
                num_positions=seq_len_mu
            ).to('cuda')

    model.load_state_dict(torch.load(store_path, map_location='cuda'), strict=True)
    model.eval()

    loss = torch.tensor([], device='cuda', dtype=torch.float)
    with torch.no_grad():
        for (source_ids, target_ids, source_mask, causal_mask) in test_dl:
            pad_mask = target_ids[:, :-1].flatten().ne(-1)
            batch_xe, _ = model.get_loss(
                source_ids=source_ids,
                source_mask=source_mask,
                target_ids=target_ids,
                causal_mask=causal_mask,
                reduction='none',
                label_smoothing=0.
            )
            loss = torch.cat((loss, batch_xe[pad_mask]), dim=-1)
        ppl = torch.exp(torch.mean(loss)).item()
    print(f'{ppl=}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run a single training iteration')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--model', type=str, required=True, choices=['Relative', 'Unitary', 'Sinusoidal', 'Rotary', 'Absolute'], help='Type of model to use')
    parser.add_argument('--task', type=str, required=True, choices=['copy', 'reverse', 'repeat', 'tree-copy', 'tree-reorder', 'c3', 'apply'], help='Which task to train on')
    parser.add_argument('--vocab_size', type=int, default=20, help='Size of vocabulary')
    parser.add_argument('--seq_len_mu', type=int, default=100, help='Mean sequence length')
    parser.add_argument('--seq_len_var', type=int, default=10, help='Sequence length variance')
    parser.add_argument('--num_epochs', type=int, default=400, help='Number of training epochs')
    parser.add_argument('--num_layers', type=int, nargs=2, default=(2, 2), help='Number of layers for the model')
    parser.add_argument('--dim', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--store_path', type=str, required=True, help='If/where to store the trained model')
    parser.add_argument('--regression', type=str, choices=['breadth', 'depth'], default=None, help='Regression order for tree decoding')
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
            seq_len_mu=args.seq_len_mu,
            seq_len_var=args.seq_len_var,
            dim=args.dim,
            num_layers=args.num_layers,
            store_path=args.store_path,
            regression=args.regression)
    else:
        train(
            model=Model[args.model],
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
