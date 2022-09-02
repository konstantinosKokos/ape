from random import seed

from treePE.data.tree import Tree, TreeGenerator
from treePE.data.tokenization import store_tokenizer, TreeTokenizer, pad_to_depth
import pickle


nodes = set(map(str, range(100)))


def prepare_data():
    seed(42)
    generator = TreeGenerator(nodes, nodes)
    trees = list(set(generator.generate(3, 30000)))
    train_trees, dev_trees, test_trees = trees[:15000], trees[15000:17000], trees[17000:]
    print(f'Generated {len(trees)} trees (train: {len(train_trees)}, dev: {len(dev_trees)}, test: {len(test_trees)})')

    store_tokenizer(trees, 'data/copy_tokenizer.tsv')
    tokenizer = TreeTokenizer.from_file('data/copy_tokenizer.tsv')

    pad = (tokenizer.PAD_token_id, 1)

    def _copy(tree: Tree[str]) -> tuple[Tree[tuple[int, int]], Tree[tuple[int, int]]]:
        padded = pad_to_depth(tokenizer.encode(tree), 4, pad)
        return padded, padded

    with open('./data/copy_trees.pkl', 'wb') as f:
        pickle.dump(([_copy(tree) for tree in train_trees],
                     [_copy(tree) for tree in dev_trees],
                     [_copy(tree) for tree in test_trees]),
                    f)
