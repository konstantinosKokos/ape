import pickle
from collections import Counter, defaultdict
from typing import Iterator
from random import randint
from itertools import takewhile, groupby
from math import sqrt, ceil

import torch

from torch.nn.utils.rnn import pad_sequence
from torch import Tensor


def readlines(file: str) -> Iterator[str]:
    with open(file, 'r') as f:
        yield from f


def build_vocab(files: list[str]):
    return Counter(
        word
        for file in files
        for line in readlines(file)
        for word in line.split()
    )


def write_vocab(vocab: dict[str, int], path: str, truncate: int) -> None:
    with open(path, 'w') as f:
        f.write('<SOS>\n<EOS>\n<UNK>\n')
        f.write('\n'.join(
            v
            for v, c in sorted(vocab.items(), key=lambda pair: pair[1], reverse=True)[:truncate - 3]))


def read_vocab(path: str) -> dict[str, int]:
    return defaultdict(
        lambda: 2,
        {k.rstrip('\n'): i for i, k in enumerate(readlines(path))}
    )


def vectorize(line: str, vocab: dict[str, int], wrap: bool) -> list[int]:
    tokens = [vocab[word] for word in line.split()]
    if wrap:
        tokens = [vocab['<SOS>'], *tokens, vocab['<EOS>']]
    return tokens


def devectorize(tokens: list[int], ivocab: dict[int, str], unwrap: bool) -> str:
    subwords = [ivocab[token] for token in tokens]
    if unwrap:
        subwords = takewhile(lambda sw: sw != '<EOS>', subwords[1:])
    return ' '.join(subwords)


def vectorize_file(file: str, vocab: dict[str, int], wrap: bool) -> list[list[int]]:
    return [vectorize(line, vocab, wrap) for line in readlines(file)]


def vectorize_files(path: str, vocab_path: str) -> None:
    vocab = read_vocab(vocab_path)
    for lang in {'en', 'de'}:
        for subset in {'train', 'dev', 'test'}:
            vectorized = vectorize_file(f'{path}/{subset}.{lang}.bpe', vocab, True)
            with open(f'{path}/{subset}.{lang}.vec', 'wb') as f:
                pickle.dump(vectorized, f)


PairSample = tuple[list[int], list[int]]


def load_datasets(
        path: str,
        subsets: tuple[str, ...] = ('train', 'dev', 'test'),
        flip: bool = False) -> Iterator[list[PairSample]]:
    for subset in subsets:
        with open(f'{path}/{subset}.en.vec', 'rb') as f:
            src = pickle.load(f)
        with open(f'{path}/{subset}.de.vec', 'rb') as f:
            tgt = pickle.load(f)
        if flip:
            src, tgt = tgt, src
        pairs = list(zip(src, tgt))
        yield pairs


def split_ds(dataset: list[PairSample], world_size: int, rank: int) -> list[PairSample]:
    dataset = sorted(dataset, key=lambda pair: sum(map(len, pair)))
    return dataset[rank::world_size]


class Dataloader:
    def __init__(self, dataset: list[PairSample]):
        self.dataset = dataset
        self.token_counts = [sum(map(len, pair)) for pair in self.dataset]

    def get_batches(self, batch_size: int) -> Iterator[list[PairSample]]:
        indices = [
            v
            for _, vs in groupby(
                iterable=sorted(
                    range(len(self.dataset)),
                    key=lambda idx: self.token_counts[idx] + randint(a=0, b=ceil(sqrt(self.token_counts[idx])))),
                key=lambda idx: self.token_counts[idx]
            )
            for v in vs
        ]

        ptr = 0
        while ptr < len(indices) - 1:
            num_tokens, batch = 0, []

            while num_tokens <= batch_size:
                if ptr == len(indices) - 1:
                    break
                pair = self.dataset[indices[ptr]]
                added = sum(map(len, pair))
                if num_tokens + added > batch_size:
                    yield batch
                    batch = [pair]
                    num_tokens = added
                else:
                    batch.append(pair)
                    num_tokens += added
                ptr += 1
            yield batch


def make_collator(device: str | int = 'cpu'):
    def collate_fn(
            samples: list[PairSample]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        input_ids = pad_sequence([torch.tensor(src, dtype=torch.long) for src, _ in samples], batch_first=True, padding_value=-1)
        output_ids = pad_sequence([torch.tensor(tgt, dtype=torch.long) for _, tgt in samples], batch_first=True, padding_value=-1)
        input_mask = input_ids.ne(-1)
        causal_mask = torch.tril(torch.ones(output_ids.shape[1], output_ids.shape[1], dtype=torch.bool), diagonal=0)
        return (input_ids.to(device),
                output_ids.to(device),
                input_mask.to(device),
                causal_mask[None, :].to(device))
    return collate_fn
