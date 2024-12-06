import os
import argparse

import torch
from torch import Tensor

from collections import OrderedDict
from typing import overload

from eval.tasks.nmt.utils import (
    read_vocab, load_datasets, devectorize as _devectorize, make_collator, merge_bpe, readlines)
from eval.models.nmt import Model, MTAlgebraic, MTVanilla, MTRotary, MTRelative, MTAbsolute, make_decoder_mask
from sacrebleu import BLEU
from sacremoses import MosesDetokenizer

from tqdm import tqdm

from time import time


def generate(
        flip: bool,
        model: Model,
        vocab_size: int,
        dim: int,
        num_layers: tuple[int, int],
        num_heads: int,
        data_path: str,
        vocab_path: str,
        store_path: str,
        sos_token_id: int = 0,
        eos_token_id: int = 1,
        beam_width: int = 4,
        alpha: float = 0.6,
):
    match model:
        case Model.Unitary:
            model = MTAlgebraic(
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
            raise

    def load(file: str) -> OrderedDict:
        print(f'Loading {file}')
        return torch.load(os.path.join(store_path, file), map_location='cuda')

    state_dicts = [load(file) for file in os.listdir(store_path) if file.endswith('.chk')]
    model.load_state_dict(checkpoint_average(*state_dicts))
    model.eval()
    model.cuda()

    try:
        model.positional_encoder.precompute(500)
    except AttributeError:
        pass

    vocab = read_vocab(vocab_path)
    ivocab = {v: k for k, v in vocab.items()}
    ivocab[-1] = '<PAD>'

    detk = MosesDetokenizer(lang='de')

    def devectorize(xs: list[int]) -> str:
        return detk.detokenize(merge_bpe(_devectorize(xs, ivocab, True)))

    test_ds, = load_datasets(data_path, ('test',), flip=flip)
    indices = sorted(range(len(test_ds)), key=lambda idx: (len(test_ds[idx][1]), idx))
    reverted_indices = sorted(range(len(indices)), key=lambda idx: indices[idx])
    test_ds = [test_ds[idx] for idx in indices]
    collate_fn = make_collator('cuda')

    ref_file = f'{data_path}/test.{"en" if flip else "de"}.detok'
    print(f"Reading references from {ref_file}")
    references = list(readlines(ref_file))

    starts = len(test_ds) // 128
    output_ids = []
    begin = time()
    with torch.no_grad():
        for start in tqdm(range(starts + 1)):
            (source_ids, target_ids, source_mask, _) = collate_fn(test_ds[start*128:(start+1)*128])
            max_decode_length = int(source_ids.size(1) * 1.5)
            preds, _ = model.forward_dev(
                source_ids=source_ids,
                source_mask=source_mask,
                causal_mask=make_decoder_mask(max_decode_length, device=source_ids.device),
                beam_width=beam_width,
                max_decode_length=max_decode_length,
                alpha=alpha
            )
            preds = preds[:, 0].cpu()
            output_ids += [p for p in preds.tolist()]
    end = time()
    print(f"Took {(end - begin):.2f} seconds")

    output_sentences = list(map(devectorize, output_ids))
    output_sentences = [output_sentences[idx] for idx in reverted_indices]

    with open(f'{store_path}/hypotheses.txt', 'w') as f:
        f.write('\n'.join(output_sentences))

    scorer = BLEU(tokenize='13a', lowercase=False, trg_lang='de')
    print(scorer.corpus_score(output_sentences, [references]))
    print(scorer.get_signature())
    exit(0)


@overload
def checkpoint_average(*xs: Tensor) -> Tensor: ...
@overload
def checkpoint_average(*xs: OrderedDict) -> OrderedDict: ...


def checkpoint_average(*xs):
    (x, *_) = xs
    if isinstance(x, dict):
        return OrderedDict([(k, checkpoint_average(*[x[k] for x in xs])) for k in x.keys()])
    else:
        return sum(xs) / len(xs)


def parse_args():
    parser = argparse.ArgumentParser(description='Run a single training iteration')
    parser.add_argument('--model', type=str, required=True, choices=['Unitary', 'Sinusoidal', 'Rotary', 'Relative', 'Absolute'], help='Type of model to use')
    parser.add_argument('--flip', action="store_true", help='Flip translation direction.')
    parser.add_argument('--vocab_size', type=int, required=True, help='Size of vocabulary')
    parser.add_argument('--dim', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--num_layers', type=int, nargs=2, default=(6, 6), help='Number of layers for the model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--data_path', type=str, required=True, help='Where to load the vectorized data from')
    parser.add_argument('--store_path', type=str, required=True, help='Where to load the trained model from')
    parser.add_argument('--vocab_path', type=str, required=True, help='Vocab path.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    generate(
        flip=args.flip,
        model=Model[args.model],
        data_path=args.data_path,
        store_path=args.store_path,
        vocab_path=args.vocab_path,
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size
    )
