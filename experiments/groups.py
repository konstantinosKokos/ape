from treePE.data.tree import Tree, Binary, Leaf, make_eval
from treePE.data.tokenization import mask_bottom_right, TreeTokenizer, store_tokenizer, pad_to_depth

from enum import Enum
from operator import eq
from random import seed, shuffle
from functools import cache

import pickle


class Three(Enum):
    x = 'x'
    y = 'y'
    z = 'z'


C3 = {(Three.x, Three.x): Three.x,
      (Three.x, Three.y): Three.y,
      (Three.x, Three.z): Three.z,
      (Three.y, Three.x): Three.y,
      (Three.y, Three.y): Three.z,
      (Three.y, Three.z): Three.x,
      (Three.z, Three.x): Three.z,
      (Three.z, Three.y): Three.x,
      (Three.z, Three.z): Three.y}


def c3(left: Three, right: Three) -> Three:
    return C3[(left, right)]


leaf_semantics = {'x': Three.x, 'y': Three.y, 'z': Three.z}
op_semantics = {'=': eq, 'c3': c3}


def options(matrix: dict[tuple[Three, Three], Three]) -> dict[Three, set[tuple[Three, Three]]]:
    return {out: {k for k, v in matrix.items() if v == out} for out in matrix.values()}


def gen(matrix: dict[tuple[Three, Three], Three], depth_max: int) -> list[Tree[str]]:
    inverted = options(matrix)

    @cache
    def go(value: Three, depth: int) -> list[Tree[str]]:
        match depth:
            case 0:
                return []
            case 1:
                return [Binary('c3', Leaf(l_arg.name), Leaf(r_arg.name)) for l_arg, r_arg in inverted[value]]
            case _:
                return [Binary('c3', l_expand, r_expand)
                        for (l_arg, r_arg) in inverted[value]
                        for l_expand in go(l_arg, depth - 1) + [Leaf(l_arg.name)]
                        for r_expand in go(r_arg, depth - 1) + [Leaf(r_arg.name)]]

    def append_eval(tree: Tree[str], value: str) -> Tree[str]: return Binary('=', tree, Leaf(value))
    return [append_eval(tree, value.name) for value in inverted.keys() for tree in go(value, depth_max)]


c3_eval = make_eval(lambda x: Three[x], lambda _: c3)
# asserted eval matches gen value
# asserted uniqueness of generated trees


def prepare_c3():
    seed(42)
    trees = gen(C3, 3)
    shuffle(trees)
    train_trees, dev_trees, test_trees = trees[:15000], trees[15000:18000], trees[18000:]
    print(f'Generated {len(trees)} trees (train: {len(train_trees)}, dev: {len(dev_trees)}, test: {len(test_trees)})')
    train_ds = [(mask_bottom_right(t), t) for t in train_trees]
    dev_ds = [(mask_bottom_right(t), t) for t in dev_trees]
    test_ds = [(mask_bottom_right(t), t) for t in test_trees]
    print(f'Expanded into (train: {len(train_ds)}, dev: {len(dev_ds)}, test: {len(test_ds)}) masked samples')

    store_tokenizer(trees, 'data/c3_tokenizer.tsv')
    tokenizer = TreeTokenizer.from_file('data/c3_tokenizer.tsv')
    pad = (tokenizer.PAD_token_id, tokenizer.PAD_token_id, 1)

    with open('./data/c3_trees.pkl', 'wb') as f:
        pickle.dump((
            [pad_to_depth(tokenizer.encode_masked(tree, mask), 4, pad=pad) for mask, tree in train_ds],
            [pad_to_depth(tokenizer.encode_masked(tree, mask), 4, pad=pad) for mask, tree in dev_ds],
            [pad_to_depth(tokenizer.encode_masked(tree, mask), 4, pad=pad) for mask, tree in test_ds]), f)
