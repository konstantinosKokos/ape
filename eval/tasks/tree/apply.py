from dataclasses import dataclass
from .task import TreeTask, TreeSample, TreeGenerator
from .abstract import flip, Binary, Leaf, Tree
from random import choice


def cut_at(tree: Tree[int], at: int) -> Tree[int]:
    if tree.node == at:
        return Leaf(at)
    match tree:
        case Binary(op, left, right):
            return Binary(op, cut_at(left, at), cut_at(right, at))
        case Leaf(_):
            return tree
    raise TypeError


@dataclass
class TreeApply(TreeTask):
    vocab_size: int

    @property
    def meta_ops(self) -> int:
        return 4

    def __post_init__(self):
        super(TreeApply, self).__post_init__()
        self.generator = TreeGenerator(
            leaves=set(range(1, self.vocab_size // 2 + 1)),
            operators=set(range(self.vocab_size // 2 + 1, self.vocab_size - (self.meta_ops - 1))))

    def sample(self, depth: int) -> TreeSample:
        x = self.generator.random_tree(depth - 1, unique_nodes=True)
        subtree = choice(x.subtrees())
        ptr = subtree.node
        op = choice(list(range(self.meta_ops)))
        match op:
            case 0:
                y = subtree
            case 1:
                y = flip(subtree)
            case 2:
                y = cut_at(x, ptr)
            case 3:
                y = x
            case _:
                raise ValueError
        x = Binary(node=self.vocab_size - (self.meta_ops - op), left=Leaf(ptr), right=x)
        return TreeSample(x=x, y=y, task=self)
