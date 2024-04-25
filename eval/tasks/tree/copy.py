from dataclasses import dataclass
from .task import TreeTask, TreeSample, TreeGenerator
from .abstract import rotate


@dataclass
class TreeCopy(TreeTask):
    vocab_size: int

    def __post_init__(self):
        super(TreeCopy, self).__post_init__()
        self.generator = TreeGenerator(
            leaves=set(range(1, self.vocab_size//2 + 1)),
            operators=set(range(self.vocab_size//2 + 1, self.vocab_size + 1)),)

    def sample(self, depth: int) -> TreeSample:
        x = self.generator.random_tree(depth)
        return TreeSample(x=x, y=x, task=self)


@dataclass
class TreeReorder(TreeCopy):
    def sample(self, depth: int) -> TreeSample:
        x = self.generator.random_tree(depth)
        return TreeSample(x=x, y=rotate(x), task=self)
