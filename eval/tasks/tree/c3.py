from dataclasses import dataclass
from .task import TreeTask, TreeSample, TreeGenerator
from .abstract import make_step_eval


def c3(op: int, left: int, right: int) -> int:
    if op != 4:
        raise ValueError
    match left, right:
        case (1, 1):
            return 1
        case (1, 2) | (2, 1):
            return 2
        case (1, 3) | (3, 1):
            return 3
        case (2, 2):
            return 3
        case (2, 3) | (3, 2):
            return 1
        case (3, 3):
            return 2
        case _:
            raise ValueError


@dataclass
class C3(TreeTask):
    def __post_init__(self):
        super(C3, self).__post_init__()
        self.generator = TreeGenerator(leaves={1, 2, 3}, operators={4})
        self.reducer = make_step_eval(c3)

    def sample(self, depth: int) -> TreeSample:
        x = self.generator.random_tree(depth)
        return TreeSample(x=x, y=self.reducer(x), task=self)
