import os
import sys
import time

if (slurm_submit_dir := os.environ.get('SLURM_SUBMIT_DIR', default=None)) is not None:
    sys.path.append(os.environ['SLURM_SUBMIT_DIR'])

import argparse
import torch

from eval.models.nmt import Model, MTUnitary, MTVanilla
from eval.tasks.nmt import make_collator, load_datasets, split_ds, Dataloader

from unitaryPE.nn.schedule import make_transformer_schedule

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from datetime import timedelta


def ddp_setup(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, timeout=timedelta(minutes=4))


def run(
        rank: int,
        world_size: int,
        model: Model,
        store_path: str,
        data_path: str,
        num_layers: tuple[int, int],
        dim: int,
        num_heads: int,
        max_updates: int,
        batch_size: int,
        update_every: int,
        save_every: int,
        flip: bool = True,
        sos_token_id: int = 0,
        eos_token_id: int = 1,
        seed: int = 42
):
    start_time = time.time()
    train_set, = load_datasets(data_path, subsets=('train',))
    train_set = split_ds(train_set, world_size, rank)
    print(f'{start_time} -- {rank} -- {len(train_set)}')
    train_dl = Dataloader(train_set)
    collator = make_collator(rank)
    sys.stdout.flush()

    ddp_setup(rank, world_size)

    smoke_test = torch.tensor(rank, device=rank)
    print(f'{smoke_test} @ {rank}')
    dist.all_reduce(smoke_test)
    print(f'{smoke_test} @ {rank}')
    sys.stdout.flush()

    torch.manual_seed(seed)

    match model:
        case Model.Unitary:
            model = MTUnitary(
                vocab_size=32000,
                num_layers=num_layers,
                dim=dim,
                num_heads=num_heads,
                sos_token_id=sos_token_id,
                eos_token_id=eos_token_id
            )
        case Model.Sinusoidal:
            model = MTVanilla(
                vocab_size=32000,
                num_layers=num_layers,
                dim=dim,
                num_heads=num_heads,
                sos_token_id=sos_token_id,
                eos_token_id=eos_token_id
            )
        case _:
            raise ValueError

    model = DistributedDataParallel(model.to(rank), device_ids=[rank])

    optim = AdamW(model.parameters(), lr=1)
    scheduler = LambdaLR(
        optimizer=optim,
        lr_lambda=make_transformer_schedule(
            dim=model.dim,
            warmup_steps=4000)
    )

    steps, updates, train_rml = 0, 0, None
    while updates < max_updates:
        model.train()
        for (input_ids, output_ids, input_mask, causal_mask) \
                in map(collator, train_dl.get_batches(batch_size=batch_size, flip=flip)):
            steps += 1
            loss = model.module.go_batch(
                source_ids=input_ids,
                source_mask=input_mask,
                target_ids=output_ids,
                causal_mask=causal_mask,
            )
            loss.backward()

            train_rml = loss.detach() if train_rml is None else (0.98 * train_rml + 0.02 * loss.detach())

            if steps % update_every == 0:
                updates += 1
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

            if updates > 0 and updates % 50 == 0:
                dist.all_reduce(train_rml)
                train_rml = train_rml.item() / world_size

                if rank == 0:
                    print(f'{updates}:{train_rml}')
                    sys.stdout.flush()

            if rank == 0 and updates > 0 and updates % save_every == 0:
                torch.save(model.module.state_dict(), f'{store_path}/{updates//save_every}.chk')

    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description='Run a single training iteration')
    parser.add_argument('--model', type=str, choices=['Unitary', 'Sinusoidal'], help='Type of model to use')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Size of vocabulary')
    parser.add_argument('--num_updates', type=int, default=15000, help='Total number of parameter updates')
    parser.add_argument('--batch_size', type=int, default=8000, help='Batch size (forward)')
    parser.add_argument('--update_every', type=int, default=40, help='Frequency of backward steps')
    parser.add_argument('--save_every', type=int, default=500, help='Frequency of model checkpointing')
    parser.add_argument('--num_layers', type=int, nargs=2, default=(6, 6), help='Number of layers for the model')
    parser.add_argument('--dim', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--data_path', type=str, help='Where to load the vectorized data from')
    parser.add_argument('--store_path', type=str, default=None, help='If/where to store the trained model')
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
            Model[args.model],
            args.store_path,
            args.data_path,
            args.num_layers,
            args.dim,
            args.num_heads,
            args.num_updates,
            args.batch_size,
            args.update_every,
            args.save_every,
        )
    )
