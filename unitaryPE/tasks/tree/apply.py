from dataclasses import dataclass
from .task import TreeTask, TreeSample, TreeGenerator
from .abstract import flip, Binary, Leaf
from random import random, choice


@dataclass
class TreeApply(TreeTask):
    vocab_size: int

    def __post_init__(self):
        super(TreeApply, self).__post_init__()
        self.generator = TreeGenerator(
            leaves=set(range(1, self.vocab_size // 2 + 1)),
            operators=set(range(self.vocab_size // 2 + 1, self.vocab_size - 1)))

    def sample(self, depth: int) -> TreeSample:
        x = self.generator.random_tree(depth - 1, unique_nodes=True)
        subtree = choice(x.subtrees())
        ptr = subtree.node
        if random() < 0.5:
            op = self.vocab_size - 1
            y = flip(subtree)
        else:
            op = self.vocab_size
            y = subtree
        x = Binary(node=op, left=Leaf(ptr), right=x)
        return TreeSample(x=x, y=y, task=self)
