import os
import sys
import time

if (slurm_submit_dir := os.environ.get('SLURM_SUBMIT_DIR', default=None)) is not None:
    sys.path.append(os.environ['SLURM_SUBMIT_DIR'])


import argparse
import torch

from eval.models.nmt import Model, MTUnitary, MTVanilla
from eval.tasks.nmt import make_collator, Dataloader, load_datasets, read_vocab, devectorize


from unitaryPE.nn.schedule import make_schedule
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from typing import Literal


def run(
        model: Model,
        data_path: str,
        store_path: str,
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
    train_set, dev_set, test_set = load_datasets(data_path)
    train_dl, dev_dl, test_dl = map(Dataloader, (train_set, dev_set, test_set))
    collator = make_collator('cuda')

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

    model = model.cuda()

    print(f'Effective batch size: {batch_size * update_every}')

    torch.manual_seed(seed)
    optim = AdamW(model.parameters(), lr=1)
    scheduler = LambdaLR(
        optimizer=optim,
        lr_lambda=make_schedule(
            warmup_steps=4000,
            warmdown_steps=max_updates - 4000,
            total_steps=max_updates,
            min_lr=1e-9,
            max_lr=5e-4,
            init_lr=1e-7))

    steps, updates, train_rml = 0, 0, 0
    while updates < max_updates:
        model.train()
        for (input_ids, output_ids, input_mask, causal_mask) \
                in map(collator, train_dl.get_batches(batch_size=batch_size, flip=flip)):
            steps += 1
            loss = model.go_batch(
                source_ids=input_ids,
                source_mask=input_mask,
                target_ids=output_ids,
                causal_mask=causal_mask,
            )
            loss.backward()

            train_rml = loss.item() if train_rml is None else (0.98 * train_rml + 0.02 * loss.item())

            if steps % update_every == 0:
                updates += 1
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

                if updates % 50 == 0:
                    dev_loss, numels = 0., 0
                    model.eval()
                    with torch.no_grad():
                        for (input_ids, output_ids, input_mask, causal_mask) \
                                in map(collator, dev_dl.get_batches(batch_size=batch_size, flip=flip)):
                            dev_loss += model.go_batch(
                                source_ids=input_ids,
                                source_mask=input_mask,
                                target_ids=output_ids,
                                causal_mask=causal_mask,
                                reduction='sum').item()
                            numels += output_ids.ne(-1).sum().item()
                    print(f'{updates}:{train_rml}:{dev_loss/numels}')
                    model.train()

                if updates % save_every == 0:
                    torch.save(model.state_dict(), f'{store_path}/{updates//save_every}.chk')


def eval(model_paths: list[str],
         data_path: str = '/home/kokos/Projects/tree_PE/eval/tasks/nmt/data/bpe',
         vocab_path: str = '/home/kokos/Projects/tree_PE/vocab.txt',
         dim: int = 512,
         num_heads: int = 8,
         sos_token_id: int = 0,
         eos_token_id: int = 1,
         num_layers: tuple[int, int] = (6, 6),
         flip: bool = True):

    model = MTUnitary(
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        sos_token_id=sos_token_id,
        eos_token_id=eos_token_id,
        vocab_size=32000,
    )

    dev_set, test_set = load_datasets(data_path, subsets={'dev', 'test'})
    dev_dl, test_dl = map(Dataloader, (dev_set, test_set))
    collator = make_collator('cpu')

    vocab = read_vocab(vocab_path)
    ivocab = {v: k for k, v in vocab.items()}

    model: MTUnitary
    for path in model_paths:
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        model.source_pe.precompute(500)
        model.target_pe.precompute(500)

        with torch.no_grad():
            loss = 0.
            for (input_ids, output_ids, input_mask, causal_mask) \
                    in map(collator, dev_dl.get_batches(batch_size=1000, flip=flip, shuffle=False)):
                loss = model.go_batch(
                    source_ids=input_ids,
                    target_ids=output_ids,
                    source_mask=input_mask,
                    causal_mask=causal_mask).item() * 0.02 + loss * 0.98
            print(loss)
            exit()
                # beams, _ = model.forward_dev(
                #     source_ids=input_ids,
                #     source_mask=input_mask,
                #     max_decode_length=2 * input_ids.size(1),
                #     beam_width=1)
                # beams = beams.cpu().tolist()
                # print(beams.shape, input_ids.shape)
                # beams = beams.cpu().tolist()
                # tokens = [[devectorize(beam, ivocab, False) for beam in sample]
                #           for sample in beams]
                # print(tokens)
                # exit()


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

    run(model=Model[args.model],
        dim=args.dim,
        max_updates=args.num_updates,
        batch_size=args.batch_size,
        update_every=args.update_every,
        save_every=args.save_every,
        data_path=args.data_path,
        store_path=args.store_path,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        seed=args.seed,
        flip=False)
