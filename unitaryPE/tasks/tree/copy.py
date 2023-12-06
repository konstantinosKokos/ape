from dataclasses import dataclass
from .task import TreeTask, TreeSample, TreeGenerator, Tree
from .abstract import flip


@dataclass
class TreeCopy(TreeTask):
    vocab_size: int

    def __post_init__(self):
        super(TreeCopy, self).__post_init__()
        self.sos_token_id = 0
        self.generator = TreeGenerator(leaves=set(range(1, self.vocab_size + 1)),
                                       operators=set(range(1, self.vocab_size + 1)),)

    def sample(self, depth: int) -> TreeSample:
        x = self.generator.random_tree(depth)
        return TreeSample(x=x, y=x, task=self)

    def process(self, x: Tree[int], y: Tree[int]) -> tuple[tuple[list[int], list[int]],
                                                           tuple[list[int], list[int]],
                                                           list[list[bool]]]:
        (input_nodes, input_pos), (output_nodes, output_pos), causal_mask = self.preprocess(x, y)
        input_nodes = [self.sos_token_id] + input_nodes
        input_pos = [0] + input_pos
        output_nodes = [self.sos_token_id] + output_nodes
        output_pos = [0] + output_pos
        causal_mask = [[True] + row for row in causal_mask]
        sos_row = [True] + [False] * len(causal_mask)
        return (input_nodes, input_pos), (output_nodes, output_pos), [sos_row] + causal_mask


@dataclass
class TreeReorder(TreeTask):
    vocab_size: int

    def __post_init__(self):
        super(TreeReorder, self).__post_init__()
        self.sos_token_id = 0
        self.generator = TreeGenerator(leaves=set(range(1, self.vocab_size + 1)),
                                       operators=set(range(1, self.vocab_size + 1)),)

    def sample(self, depth: int) -> TreeSample:
        x = self.generator.random_tree(depth)
        return TreeSample(x=x, y=flip(y), task=self)

    def process(self, x: Tree[int], y: Tree[int]) -> tuple[tuple[list[int], list[int]],
                                                           tuple[list[int], list[int]],
                                                           list[list[bool]]]:
        (input_nodes, input_pos), (output_nodes, output_pos), causal_mask = self.preprocess(x, y)
        input_nodes = [self.sos_token_id] + input_nodes
        input_pos = [0] + input_pos
        output_nodes = [self.sos_token_id] + output_nodes
        output_pos = [0] + output_pos
        causal_mask = [[True] + row for row in causal_mask]
        sos_row = [True] + [False] * len(causal_mask)
        return (input_nodes, input_pos), (output_nodes, output_pos), [sos_row] + causal_mask
