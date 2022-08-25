from __future__ import annotations

from .tree import Tree, Binary, Leaf, Node
from random import choice


MASK = '[MASK]'
PAD = '[PAD]'


def all_masks(tree: Tree[Node], mask_ops: bool = True) -> list[Tree[bool]]:
    match tree:
        case Leaf(_):
            return [Leaf(False)]
        case Binary(_, left, right):
            left_pass = left.fmap(lambda _: True)
            right_pass = right.fmap(lambda _: True)
            ret = [*[Binary(True, left_mask, right_pass) for left_mask in all_masks(left)],
                   *[Binary(True, left_pass, right_mask) for right_mask in all_masks(right)]]
            if mask_ops:
                ret += [Binary(False, left_pass, right_pass)]
            return ret
    raise ValueError


def random_mask(tree: Tree[Node], mask_ops: bool = True) -> Tree[bool]:
    return choice(all_masks(tree, mask_ops))


def positionally_encode(tree: Tree[Node]) -> Tree[int]:
    def go(_tree: Tree[Node], parent: int) -> Tree[int]:
        match _tree:
            case Leaf(_):
                return Leaf(parent)
            case Binary(_, left, right):
                return Binary(parent, go(left, 2 * parent), go(right, 2 * parent + 1))
        raise TypeError
    return go(tree, 1)


class TreeTokenizer:
    def __init__(self, symbol_map: dict[int, str]):
        self.id_to_token = symbol_map
        self.token_to_id = {v: k for k, v in symbol_map.items()}
        self.PAD_token_id = self.token_to_id[PAD]
        self.MASK_token_id = self.token_to_id[MASK]

    def __len__(self) -> int: return len(self.id_to_token)
    def atom_to_id(self, atom: str) -> int: return self.token_to_id[atom]
    def id_to_atom(self, idx: int) -> str: return self.id_to_token[idx]

    def encode(self, tree: Tree[str]) -> Tree[tuple[int, int]]:
        return tree.fmap(self.atom_to_id).zip(positionally_encode(tree))

    def encode_masked(self, tree: Tree[str], mask: Tree[bool]) -> Tree[tuple[int, int, int]]:
        def mask_tuple_to_io(atom: str, mask_value: bool) -> tuple[int, int]:
            return (atom_id := self.atom_to_id(atom)), atom_id if mask_value else self.MASK_token_id

        def flatten(x: tuple[int, int], y: int) -> tuple[int, int, int]: return x[0], x[1], y

        return tree.zipwith(mask, _with=mask_tuple_to_io).zipwith(positionally_encode(tree), _with=flatten)

    @staticmethod
    def from_file(file_path: str) -> TreeTokenizer:
        id_to_sym = {}
        with open(file_path, 'r') as f:
            for line in f.readlines():
                idx, name = line.rstrip('\n').split('\t')
                id_to_sym[eval(idx)] = str(name)
        return TreeTokenizer(id_to_sym)


def pad_to_depth(tree: Tree[Node], depth: int, pad: Node) -> Tree[Node]:
    def go(_tree: Tree[Node], current_depth: int) -> Tree[Node]:
        if current_depth == depth:
            return _tree
        match _tree:
            case Leaf(node):
                return Binary(node, go(Leaf(pad), current_depth + 1), go(Leaf(pad), current_depth + 1))
            case Binary(node, left, right):
                return Binary(node, go(left, current_depth + 1), go(right, current_depth + 1))
        raise ValueError
    return go(tree, 0)


def extract_unique_symbols(trees: list[Tree[str]]) -> set[tuple[str, int]]:
    return {(symbol, arity) for tree in trees for symbol, arity in tree.nodes_and_arities()}


def make_symbol_map(symbols: set[tuple[str, int]]) -> tuple[dict[int, str], dict[str, int]]:
    sorted_symbols = sorted(symbols, key=lambda s: (s[1], s[0]))
    id_to_symbol = {i: s for i, s in enumerate([s for s, _ in sorted_symbols])}
    symbol_to_arity = {s: a for s, a in sorted_symbols}
    return id_to_symbol, symbol_to_arity


def store_tokenizer(trees: list[Tree[str]], file_path: str) -> None:
    symbols = extract_unique_symbols(trees)
    id_to_symbol, _ = make_symbol_map(symbols)
    with open(file_path, 'w') as f:
        for i, s in id_to_symbol.items():
            f.write(f'{i}\t{s}\n')
        f.write(f'{(i := i+1)}\t{MASK}\n')
        f.write(f'{(i := i + 1)}\t{PAD}')
