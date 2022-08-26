from typing import Callable, TypeVar
from operator import add, sub
from random import seed

from treePE.data.tree import Tree, Binary, Leaf, TreeGenerator, make_eval
from treePE.data.tokenization import mask_bottom_right, store_tokenizer, TreeTokenizer, pad_to_depth
import pickle


T = TypeVar('T')
BOp = Callable[[bool, bool], bool]

operator_semantics: dict[str, BOp]
operator_semantics = {
    '+': add,
    '-': sub,
}

leaf_semantics = eval
leaves = set(map(str, range(-10, 10)))


eval_tree = make_eval(leaf_semantics, lambda x: operator_semantics[x])


def append_eval_to_tree(tree: Tree[str]) -> Tree[str]:
    result = str(eval_tree(tree))
    return Binary('=', tree, Leaf(result))


def prepare_data():
    seed(42)
    generator = TreeGenerator(leaves, set(operator_semantics.keys()))
    trees = list(filter(lambda tree: tree.right.node in leaves,
                        map(append_eval_to_tree, set(generator.generate(2, 20000)))))
    train_trees, dev_trees, test_trees = trees[:8000], trees[8000:9000], trees[9000:]
    print(f'Generated {len(trees)} trees (train: {len(train_trees)}, dev: {len(dev_trees)}, test: {len(test_trees)})')
    # train_ds = [(random_mask(t, False), t) for t in train_trees]
    # train_ds = [(m, t) for t in train_trees for m in all_masks(t, False)]
    train_ds = [(mask_bottom_right(t), t) for t in train_trees]
    # dev_ds = [(m, t) for t in dev_trees for m in all_masks(t, False)]
    dev_ds = [(mask_bottom_right(t), t) for t in dev_trees]
    # test_ds = [(m, t) for t in test_trees for m in all_masks(t, False)]
    test_ds = [(mask_bottom_right(t), t) for t in test_trees]
    print(f'Expanded into (train: {len(train_ds)}, dev: {len(dev_ds)}, test: {len(test_ds)}) masked samples')

    store_tokenizer(trees, 'data/arithm_tokenizer.tsv')
    tokenizer = TreeTokenizer.from_file('data/arithm_tokenizer.tsv')

    pad = (tokenizer.PAD_token_id, tokenizer.PAD_token_id, 1)

    with open('./data/arithm_trees.pkl', 'wb') as f:
        pickle.dump((
            [pad_to_depth(tokenizer.encode_masked(tree, mask), 3, pad=pad) for mask, tree in train_ds],
            [pad_to_depth(tokenizer.encode_masked(tree, mask), 3, pad=pad) for mask, tree in dev_ds],
            [pad_to_depth(tokenizer.encode_masked(tree, mask), 3, pad=pad) for mask, tree in test_ds]), f)

