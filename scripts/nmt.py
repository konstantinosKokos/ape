import os
import sys

if (slurm_submit_dir := os.environ.get('SLURM_SUBMIT_DIR', default=None)) is not None:
    sys.path.append(os.environ['SLURM_SUBMIT_DIR'])

import argparse

from eval.models.nmt import Model, MTUnitary, MTVanilla, MTRotary, MTRelative, MTAbsolute
from eval.tasks.nmt import make_collator, load_datasets, clean_dataset, Dataloader

from unitaryPE.nn.schedule import make_transformer_schedule

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DataParallel

from time import time

def argmin(xs: list[float]) -> int:
    return min(list(range(len(xs))), key=lambda i: xs[i])


def run(
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
        accum_steps: int,
        num_checkpoints: int,
        sos_token_id: int = 0,
        eos_token_id: int = 1,
        seed: int = 42
):
    train_set, dev_set = load_datasets(data_path, subsets=('train', 'dev'), flip=flip)
    train_set, dev_set = tuple(map(clean_dataset, (train_set, dev_set)))
    device_ids = list(range(torch.cuda.device_count()))
    update_every = accum_steps // len(device_ids)
    effective_batch_size = len(device_ids) * batch_size * update_every
    print(f'{len(device_ids)} * {batch_size} * {update_every} = {effective_batch_size}')
    train_dl = Dataloader(train_set)
    dev_dl = Dataloader(dev_set)
    collator = make_collator(f'cuda:{next(iter(device_ids))}')
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

    model = DataParallel(model, device_ids=device_ids)
    model = model.to(f'cuda:{next(iter(device_ids))}')

    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98), weight_decay=0.)
    scheduler = LambdaLR(
        optimizer=optim,
        lr_lambda=make_transformer_schedule(
            dim=model.module.dim,
            warmup_steps=4000)
    )

    dev_losses = []
    checkpoint, total_steps, updates = 0, 0, 0
    train_rml, batch_loss, batch_numels = None, None, None
    epoch = -1

    start = time()

    while True:
        epoch += 1
        model.train()
        train_iterator = map(collator, train_dl.get_batches(batch_size=batch_size * len(device_ids)))

        for input_ids, output_ids, input_mask, causal_mask in train_iterator:
            total_steps += 1
            loss, numels = model.forward(
                source_ids=input_ids,
                source_mask=input_mask,
                target_ids=output_ids,
                causal_mask=causal_mask,
                reduction='sum'
            )
            loss, numels = map(torch.sum, (loss, numels))
            loss.backward()
            batch_numels = numels.item() if batch_numels is None else batch_numels + numels.item()
            batch_loss = loss.item() if batch_loss is None else batch_loss + loss.item()

            if total_steps % update_every == 0:
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad /= numels
                optim.step()
                scheduler.step()
                optim.zero_grad()

                updates += 1
                batch_loss /= batch_numels
                train_rml = batch_loss if train_rml is None else (0.98 * train_rml + 0.02 * batch_loss)
                batch_loss, batch_numels = None, None

                if updates > 0 and updates % 500 == 0:

                    model.eval()
                    dev_numels, dev_loss = 0, 0
                    with torch.no_grad():
                        dev_iterator = map(collator, dev_dl.get_batches(batch_size=batch_size * len(device_ids)))
                        for input_ids, output_ids, input_mask, causal_mask in dev_iterator:
                            loss, numels = model.forward(
                                source_ids=input_ids,
                                source_mask=input_mask,
                                target_ids=output_ids,
                                causal_mask=causal_mask,
                                reduction='sum'
                            )
                            loss, numels = map(torch.sum, (loss, numels))
                            dev_numels += numels.item()
                            dev_loss += loss.item()
                        dev_loss /= dev_numels

                        dev_losses.append(dev_loss)
                    model.train()

                    print(f'{epoch}:{total_steps}:{updates}:{scheduler.get_last_lr()[0]:.5f}')
                    print(f'{train_rml:.3f}:{dev_loss:.3f}')

                    if dev_loss < max(sorted(dev_losses)[:num_checkpoints]):
                        print(f'Saving {checkpoint} at {updates}.')
                        torch.save(model.module.state_dict(), f'{store_path}/{checkpoint}.chk')
                        checkpoint = 0 if checkpoint == (num_checkpoints - 1) else checkpoint + 1
                    print('-' * 64)
                    sys.stdout.flush()

            if updates == num_updates:
                print('Exiting')
                sys.stdout.flush()
                break

            if total_steps == 1000:
                break

        end = time()
        break
    print(end - start)


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
    parser.add_argument('--accum_steps', type=int, required=True, help='Frequency of backward steps')
    parser.add_argument('--num_checkpoints', type=int, default=10, help='How many checkpoints to store')
    parser.add_argument('--seed', type=int, default=42, help='The id of the current repetition')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    sys.stdout.flush()

    run(flip=args.flip,
        model=Model[args.model],
        vocab_size=args.vocab_size,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        data_path=args.data_path,
        store_path=args.store_path,
        num_updates=args.num_updates,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        num_checkpoints=args.num_checkpoints,
        sos_token_id=0,
        eos_token_id=1,
        seed=args.seed
    )
