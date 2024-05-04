import os
import sys
import time

if (slurm_submit_dir := os.environ.get('SLURM_SUBMIT_DIR', default=None)) is not None:
    sys.path.append(os.environ['SLURM_SUBMIT_DIR'])

import argparse
from random import randint

from eval.models.nmt import Model, MTUnitary, MTVanilla, MTRotary, MTRelative, MTAbsolute
from eval.tasks.nmt import make_collator, load_datasets, split_ds, filter_ds, Dataloader

from unitaryPE.nn.schedule import make_transformer_schedule

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from datetime import timedelta


def ddp_setup(rank: int, world_size: int, master_port: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, timeout=timedelta(minutes=4))


def argmin(xs: list[float]) -> int:
    return min(list(range(len(xs))), key=lambda i: xs[i])


def run(
        rank: int,
        world_size: int,
        master_port: int,
        flip: bool,
        model: Model,
        vocab_size: int,
        dim: int,
        num_layers: tuple[int, int],
        num_heads: int,
        data_path: str,
        store_path: str,
        num_updates: int,
        batch_size: int,
        update_every: int,
        num_checkpoints: int,
        sos_token_id: int = 0,
        eos_token_id: int = 1,
        seed: int = 42
):
    start_time = time.time()
    train_set, dev_set = load_datasets(data_path, subsets=('train', 'dev'), flip=flip)
    train_set, dev_set = tuple(map(lambda ds: filter_ds(ds, max_seq_len=95), (train_set, dev_set)))
    train_set = split_ds(train_set, world_size, rank)
    dev_set = split_ds(dev_set, world_size, rank)
    print(f'{start_time} -- {rank} -- {len(train_set)}')
    train_dl = Dataloader(train_set)
    dev_dl = Dataloader(dev_set)
    collator = make_collator(rank)
    sys.stdout.flush()

    ddp_setup(rank, world_size, master_port)

    smoke_test = torch.tensor(rank, device=rank)
    print(f'{smoke_test} @ {rank}')
    dist.all_reduce(smoke_test)
    print(f'{smoke_test} @ {rank}')
    sys.stdout.flush()

    torch.manual_seed(seed)

    match model:
        case Model.Unitary:
            model = MTUnitary(
                vocab_size=vocab_size,
                num_layers=num_layers,
                dim=dim,
                num_heads=num_heads,
                sos_token_id=sos_token_id,
                eos_token_id=eos_token_id
            )
        case Model.Sinusoidal:
            model = MTVanilla(
                vocab_size=vocab_size,
                num_layers=num_layers,
                dim=dim,
                num_heads=num_heads,
                sos_token_id=sos_token_id,
                eos_token_id=eos_token_id
            )
        case Model.Rotary:
            model = MTRotary(
                vocab_size=vocab_size,
                num_layers=num_layers,
                dim=dim,
                num_heads=num_heads,
                sos_token_id=sos_token_id,
                eos_token_id=eos_token_id
            )
        case Model.Relative:
            model = MTRelative(
                vocab_size=vocab_size,
                num_layers=num_layers,
                dim=dim,
                num_heads=num_heads,
                window_size=300,
                sos_token_id=sos_token_id,
                eos_token_id=eos_token_id
            )
        case Model.Absolute:
            model = MTAbsolute(
                vocab_size=vocab_size,
                num_layers=num_layers,
                dim=dim,
                num_heads=num_heads,
                num_positions=300,
                sos_token_id=sos_token_id,
                eos_token_id=eos_token_id
            )
        case _:
            raise ValueError

    model = DistributedDataParallel(model.to(rank), device_ids=[rank])

    optim = AdamW(model.parameters(), lr=1, betas=(0.9, 0.98), weight_decay=0.)
    scheduler = LambdaLR(
        optimizer=optim,
        lr_lambda=make_transformer_schedule(
            dim=model.module.dim,
            warmup_steps=4000)
    )

    dev_losses, checkpoint, steps, updates, train_rml = [], 0, 0, 0, None
    while True:
        model.train()
        train_iterator = map(collator, train_dl.get_batches(batch_size=batch_size))
        for (input_ids, output_ids, input_mask, causal_mask) in train_iterator:
            loss = model.module.get_loss(
                source_ids=input_ids,
                source_mask=input_mask,
                target_ids=output_ids,
                causal_mask=causal_mask,
            )
            loss.backward()
            steps += 1
            train_rml = loss.detach() if train_rml is None else (0.98 * train_rml + 0.02 * loss.detach())

            if steps % update_every == 0:
                updates += 1
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

                if updates > 0 and updates % 500 == 0:
                    dist.all_reduce(train_rml)
                    train_rml = train_rml.item() / world_size

                    model.eval()
                    numels, dev_loss = 0, None
                    with torch.no_grad():
                        dev_iterator = map(collator, dev_dl.get_batches(batch_size=batch_size))
                        for (input_ids, output_ids, input_mask, causal_mask) in dev_iterator:
                            loss = model.module.get_loss(
                                source_ids=input_ids,
                                source_mask=input_mask,
                                target_ids=output_ids,
                                causal_mask=causal_mask,
                                reduction='sum'
                            )
                            dev_loss = loss if dev_loss is None else dev_loss + loss
                            numels += output_ids.ne(-1).sum()
                        dev_loss /= numels
                        dist.all_reduce(dev_loss)
                        dev_loss /= world_size
                        dev_losses.append(dev_loss.item())
                    model.train()

                    if rank == 0:
                        print(f'{steps}:{updates}:{scheduler.get_last_lr()[0]:.5f}')
                        print(f'{train_rml:.3f}:{dev_loss.item():.3f}')
                        sys.stdout.flush()

                        if dev_loss < max(sorted(dev_losses)[:num_checkpoints]):
                            print(f'Saving {checkpoint} at {updates}.')
                            sys.stdout.flush()
                            torch.save(model.module.state_dict(), f'{store_path}/{checkpoint}.chk')
                            checkpoint = 0 if checkpoint == (num_checkpoints - 1) else checkpoint + 1

        if updates == num_updates:
            break

    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description='Run a single training iteration')
    parser.add_argument('--model', type=str, required=True, choices=['Unitary', 'Sinusoidal', 'Rotary', 'Relative', 'Absolute'], help='Type of model to use')
    parser.add_argument('--flip', action="store_true", help='Flip translation direction.')
    parser.add_argument('--vocab_size', type=int, required=True, help='Size of vocabulary')
    parser.add_argument('--dim', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--num_layers', type=int, nargs=2, default=(6, 6), help='Number of layers for the model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--data_path', type=str, required=True, help='Where to load the vectorized data from')
    parser.add_argument('--store_path', type=str, required=True, help='If/where to store the trained model')
    parser.add_argument('--num_updates', type=int, required=True, help='Total number of parameter updates')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size (forward)')
    parser.add_argument('--update_every', type=int, required=True, help='Frequency of backward steps')
    parser.add_argument('--num_checkpoints', type=int, default=10, help='How many checkpoints to store')
    parser.add_argument('--seed', type=int, default=42, help='The id of the current repetition')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    world_size = torch.cuda.device_count()
    print(f'Effective batch size: {args.batch_size * args.update_every * world_size}')
    sys.stdout.flush()

    mp.spawn(
        run,
        nprocs=world_size,
        args=(
            world_size,
            randint(0, 100) + 12355,
            args.flip,
            Model[args.model],
            args.vocab_size,
            args.dim,
            args.num_layers,
            args.num_heads,
            args.data_path,
            args.store_path,
            args.num_updates,
            args.batch_size,
            args.update_every,
            args.num_checkpoints,
            0,
            1,
            args.seed
        )
    )
