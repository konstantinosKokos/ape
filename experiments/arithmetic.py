import pdb
from typing import Callable, TypeVar
from operator import add, sub
from random import seed
from torch import device
from torch.optim import AdamW

from treePE.data.tree import Tree, Binary, Leaf, TreeGenerator, Node, depth_first, breadth_first
from treePE.data.tokenization import all_masks, random_mask, store_tokenizer, TreeTokenizer, pad_to_depth
from treePE.data.batching import make_cfn_mtm

from treePE.neural.MTM import MTM
from treePE.neural.positional_encoders import PositionalEncoder

from torch.utils.data import DataLoader

import pickle


T = TypeVar('T')
BOp = Callable[[bool, bool], bool]

operator_semantics: dict[str, BOp]
operator_semantics = {
    '+': add,
    '-': sub,
}

leaf_semantics = eval
leaves = set(map(str, range(-10, 11)))


def eval_tree(tree: Tree[str]) -> bool:
    match tree:
        case Leaf(value): return leaf_semantics(value)
        case Binary(op, left, right): return operator_semantics[op](eval_tree(left), eval_tree(right))
    raise ValueError


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
    train_ds = [(m, t) for t in train_trees for m in all_masks(t, False)]
    dev_ds = [(m, t) for t in dev_trees for m in all_masks(t, False)]
    test_ds = [(m, t) for t in test_trees for m in all_masks(t, False)]
    print(f'Expanded into (train: {len(train_ds)}, dev: {len(dev_ds)}, test: {len(test_ds)}) masked samples')

    store_tokenizer(trees, 'data/arithm_tokenizer.tsv')
    tokenizer = TreeTokenizer.from_file('data/arithm_tokenizer.tsv')

    pad = (tokenizer.PAD_token_id, tokenizer.PAD_token_id, 1)

    with open('./data/arithm_trees.pkl', 'wb') as f:
        pickle.dump((
            [pad_to_depth(tokenizer.encode_masked(tree, mask), 3, pad=pad) for mask, tree in train_ds],
            [pad_to_depth(tokenizer.encode_masked(tree, mask), 3, pad=pad) for mask, tree in dev_ds],
            [pad_to_depth(tokenizer.encode_masked(tree, mask), 3, pad=pad) for mask, tree in test_ds]), f)


def run(traversal: Callable[[Tree[Node]], list[Node]], positional_encoder: PositionalEncoder) -> None:
    with open('./data/arithm_trees.pkl', 'rb') as f:
        data = pickle.load(f)
    tokenizer = TreeTokenizer.from_file('data/arithm_tokenizer.tsv')
    cfn = make_cfn_mtm(tokenizer.PAD_token_id, device('cuda'), traversal)

    train, dev, test = data
    train_dl = DataLoader(train, batch_size=64, collate_fn=cfn, shuffle=True)   # type: ignore
    dev_dl = DataLoader(dev, batch_size=512, collate_fn=cfn, shuffle=False)     # type: ignore
    test_dl = DataLoader(test, batch_size=512, collate_fn=cfn, shuffle=False)   # type: ignore

    model = MTM(vocab_size=len(tokenizer), num_heads=1, num_layers=3, dim=32, positional_encoder=positional_encoder)
    model = model.to(device('cuda'))

    optim = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    for epoch in range(99):
        print(f'Epoch {epoch}')
        print('=' * 64)
        model.train()
        train_loss, train_correct, train_total = \
            model.go_epoch(data=train_dl, masking_value=tokenizer.MASK_token_id, optimizer=optim)
        print(f'Train loss: {train_loss}, accuracy: {train_correct}/{train_total} ({train_correct / train_total})')
        model.eval()
        dev_loss, dev_correct, dev_total = \
            model.go_epoch(data=dev_dl, masking_value=tokenizer.MASK_token_id, optimizer=None)
        print(f'Dev loss: {dev_loss}, accuracy: {dev_correct}/{dev_total} ({dev_correct / dev_total})')
        test_loss, test_correct, test_total = \
            model.go_epoch(data=test_dl, masking_value=tokenizer.MASK_token_id, optimizer=None)
        print(f'Test loss: {test_loss}, accuracy: {test_correct}/{test_total} ({test_correct / test_total})')
        print('\n')


