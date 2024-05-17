import pickle
from typing import Iterator, TypeVar, Iterable
from collections import Counter, defaultdict
from itertools import takewhile
from random import sample
from math import ceil

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


T = TypeVar('T')


def readlines(file: str) -> Iterator[str]:
    with open(file, 'r') as f:
        yield from f


PairSample = tuple[list[int], list[int]]
Dataset = list[PairSample]


def clean_dataset(dataset: Dataset) -> Dataset:
    def passing(s: PairSample) -> bool:
        lx, ly = tuple(map(len, s))
        return 0 < lx < 100 and 0 < ly < 100 and max(lx, ly) / min(lx, ly) < 1.5

    return list(filter(passing, dataset))


def build_vocab(files: list[str]):
    return Counter(
        word
        for file in files
        for line in readlines(file)
        for word in line.split()
    )


def write_vocab(vocab: dict[str, int], path: str) -> None:
    with open(path, 'w') as f:
        f.write('<SOS>\t0\n<EOS>\t1\n<UNK>\t2\n')
        f.write('\n'.join(
            f'{v}\t{idx+3}'
            for idx, v in enumerate(sorted(vocab.keys(), key=lambda k: (vocab[k], k), reverse=True))))


def read_vocab(path: str) -> dict[str, int]:
    def splitline(line: str) -> tuple[str, int]:
        key, value = line.strip('\n').split('\t')
        return key, eval(value)

    return defaultdict(lambda: 2, {k: v for k, v in map(splitline, readlines(path))})


def vectorize(line: str, vocab: dict[str, int], wrap: bool) -> list[int]:
    tokens = [vocab[word] for word in line.split()]
    if wrap:
        tokens = [vocab['<SOS>'], *tokens, vocab['<EOS>']]
    return tokens


def devectorize(tokens: list[int], ivocab: dict[int, str], unwrap: bool) -> list[str]:
    subwords = [ivocab[token] for token in tokens]
    if unwrap:
        subwords = takewhile(lambda sw: sw != '<EOS>', subwords[1:])
    return list(subwords)


def vectorize_file(file: str, vocab: dict[str, int], wrap: bool) -> list[list[int]]:
    return [vectorize(line, vocab, wrap) for line in readlines(file)]


def vectorize_files(path: str, vocab_path: str) -> None:
    vocab = read_vocab(vocab_path)
    for lang in {'en', 'de'}:
        for subset in {'train', 'dev', 'test'}:
            vectorized = vectorize_file(f'{path}/bpe.{subset}.tok.{lang}', vocab, True)
            with open(f'{path}/{subset}.{lang}.vec', 'wb') as f:
                pickle.dump(vectorized, f)


def load_datasets(
        path: str,
        subsets: tuple[str, ...] = ('train', 'dev', 'test'),
        flip: bool = False) -> Iterator[list[PairSample]]:
    for subset in subsets:
        with open(f'{path}/{subset}.en.vec', 'rb') as f:
            src = pickle.load(f)
        with open(f'{path}/{subset}.de.vec', 'rb') as f:
            tgt = pickle.load(f)
        assert len(src) == len(tgt)
        if flip:
            src, tgt = tgt, src
        pairs = list(zip(src, tgt))
        yield pairs


def shuffle(vs: Iterable[T]) -> list[T]:
    vs = list(vs)
    return sample(vs, len(vs))


def split_dataset(dataset: list[PairSample], world_size: int, rank: int) -> list[PairSample]:
    dataset = sorted(dataset, key=lambda pair: sum(map(len, pair)))
    return dataset[rank::world_size]


class Dataloader:
    def __init__(self, dataset: list[PairSample], num_buckets: int = 10):
        self.dataset = dataset
        self.token_counts = [len(target) for _, target in self.dataset]
        sorted_indices = sorted(list(range(len(self.dataset))), key=lambda i: self.token_counts[i])
        bucket_size = ceil(len(sorted_indices) / num_buckets)
        self.buckets = [sorted_indices[i:i + bucket_size] for i in range(0, len(sorted_indices), bucket_size)]

    def get_batches(self, batch_size: int) -> Iterator[list[PairSample]]:
        bucket_indices = tuple(map(shuffle, self.buckets))
        indices = sum(bucket_indices, [])

        batches, num_tokens, batch = [], 0, []
        for idx in indices:
            sample_size = self.token_counts[idx]
            if num_tokens + sample_size > batch_size:
                batches.append(batch)
                batch = [self.dataset[idx]]
                num_tokens = sample_size
            else:
                batch.append(self.dataset[idx])
                num_tokens += sample_size

        if batch:
            batches.append(batch)

        batches = shuffle(batches)
        yield from iter(batches)


def make_collator(device: str | int = 'cpu'):
    def collate_fn(
            samples: list[PairSample]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        input_ids = pad_sequence([torch.tensor(src, dtype=torch.long) for src, _ in samples], batch_first=True, padding_value=-1)
        output_ids = pad_sequence([torch.tensor(tgt, dtype=torch.long) for _, tgt in samples], batch_first=True, padding_value=-1)
        input_mask = input_ids.ne(-1)
        causal_mask = torch.tril(torch.ones(output_ids.shape[1], output_ids.shape[1], dtype=torch.bool), diagonal=0)[None]
        causal_mask = causal_mask.repeat(input_ids.size(0), *(1 for _ in range(input_ids.ndim)))
        return (input_ids.to(device),
                output_ids.to(device),
                input_mask.to(device),
                causal_mask.to(device))
    return collate_fn


def merge_bpe(xs: list[str]) -> list[str]:
    match xs:
        case [x]:
            return [x]
        case (x1, x2, *rest):
            if x1.endswith('@@'):
                return merge_bpe([x1.rstrip('@@')+x2, *rest])
            return [x1, *merge_bpe([x2, *rest])]

