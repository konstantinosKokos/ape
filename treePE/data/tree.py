from __future__ import annotations
from typing import TypeVar, Generic, Callable, Any, Iterator
from abc import ABC
from itertools import zip_longest
from random import choice, randint


Node = TypeVar('Node')
Other = TypeVar('Other')
T = TypeVar('T')


class Tree(ABC, Generic[Node]):
    node: Node
    def depth(self) -> int: return tree_depth(self)
    def __eq__(self, other: Any) -> bool: return tree_eq(self, other)
    def __hash__(self) -> int: return tree_hash(self)
    def __repr__(self) -> str: return tree_repr(self)
    def nodes_and_arities(self) -> list[tuple[Node, int]]: return nodes_and_arities(self)
    def fmap(self, f: Callable[[Node], Other]) -> 'Tree[Other]': return tree_fmap(self, f)
    def levels(self) -> list[list[Node]]: return levels(self)
    def numel(self) -> int: return numel(self)
    def infix(self) -> str: return infix(self)
    def zip(self, other: Tree[Other]) -> Tree[tuple[Node, Other]]: return self.zipwith(other, lambda x, y: (x, y))

    def zipwith(self, other: Tree[Other], _with: Callable[[Node, Other], T]) -> Tree[T]:
        return tree_zip(self, other, _with)


class Binary(Tree[Node]):
    __match_args__ = ('node', 'left', 'right')

    def __init__(self, node: Node, left: Tree[Node], right: Tree[Node]):
        self.node = node
        self.left = left
        self.right = right


class Leaf(Tree[Node]):
    __match_args__ = ('node',)

    def __init__(self, node: Node):
        self.node = node


def tree_depth(tree: Tree[Node]) -> int:
    match tree:
        case Leaf(_): return 0
        case Binary(_, left, right): return 1 + max(tree_depth(left), tree_depth(right))


def tree_eq(left: Tree[Node], right: Tree[Node]) -> bool:
    match left, right:
        case Leaf(lnode), Leaf(rnode):
            return lnode == rnode
        case Binary(lnode, lleft, lright), Binary(rnode, rleft, rright):
            return lnode == rnode and tree_eq(lleft, rleft) and tree_eq(lright, rright)
        case _:
            return False


def tree_hash(tree: Tree[Node]) -> int:
    match tree:
        case Leaf(node): return hash((node,))
        case Binary(node, left, right): return hash((node, left, right))
        case _: raise TypeError(f'{tree} is not a tree')


def tree_repr(tree: Tree[Node]) -> str:
    match tree:
        case Leaf(node): return f'Leaf({node})'
        case Binary(node, left, right): return f'Binary({node}, {tree_repr(left)}, {tree_repr(right)})'
        case _: raise TypeError(f'{tree} is not a tree')


def nodes_and_arities(tree: Tree[Node]) -> list[tuple[Node, int]]:
    match tree:
        case Leaf(node): return [(node, 0)]
        case Binary(node, left, right): return [(node, 2)] + nodes_and_arities(left) + nodes_and_arities(right)
        case _: raise TypeError(f'{tree} is not a tree')


def tree_fmap(tree: Tree[Node], f: Callable[[Node], Other]) -> 'Tree[Other]':
    match tree:
        case Leaf(node): return Leaf(f(node))
        case Binary(node, left, right): return Binary(f(node), tree_fmap(left, f), tree_fmap(right, f))
        case _: raise TypeError(f'{tree} is not a tree')


def depth_first(tree: Tree[Node]) -> list[Node]:
    match tree:
        case Leaf(node): return [node]
        case Binary(node, left, right): return [node] + depth_first(left) + depth_first(right)
        case _: raise TypeError(f'{tree} is not a tree')


def breadth_first(tree: Tree[Node]) -> list[Node]:
    return sum(levels(tree), [])


def levels(tree: Tree[Node]) -> list[list[Node]]:
    match tree:
        case Leaf(node): return [[node]]
        case Binary(node, left, right):
            return [[node]] + [sum(xs, []) for xs in zip_longest(levels(left), levels(right), fillvalue=[])]


def numel(tree: Tree[Node]) -> int:
    match tree:
        case Leaf(_): return 1
        case Binary(_, left, right): return 1 + numel(left) + numel(right)
        case _: raise TypeError(f'{tree} is not a tree')


def infix(tree: Tree[Node]) -> str:
    match tree:
        case Leaf(node): return str(node)
        case Binary(node, left, right): return f'({infix(left)}{node}{infix(right)})'
        case _: raise TypeError(f'{tree} is not a tree')


def tree_zip(left: Tree[Node], right: Tree[Other], f: Callable[[Node, Other], T]) -> Tree[T]:
    def go(_left: Tree[Node], _right: Tree[Other]) -> Tree[T]:
        return tree_zip(_left, _right, f)

    match left, right:
        case Leaf(left_node), Leaf(right_node):
            return Leaf(f(left_node, right_node))
        case Binary(left_node, left_left, left_right), Binary(right_node, right_left, right_right):
            return Binary(f(left_node, right_node), go(left_left, right_left), go(left_right, right_right))
    raise ValueError


def depth_slice(trees: list[Tree[Node]], depth: int) -> list[list[Node]]:
    return [treelvls[depth] if len(treelvls := levels(tree)) > depth else [] for tree in trees]


def combine_with(op: Node, lefts: set[Tree[Node]], rights: set[Tree[Node]]) -> set[Tree[Node]]:
    return {Binary(op, left, right) for left in lefts for right in rights}


def make_eval(leaf_semantics: Callable[[str], T],
              operator_semantics: Callable[[str], Callable[[T, T], T]]) -> Callable[[Tree[str]], T]:
    def eval_tree(tree: Tree[str]) -> T:
        match tree:
            case Leaf(value): return leaf_semantics(value)
            case Binary(op, left, right): return operator_semantics(op)(eval_tree(left), eval_tree(right))
        raise ValueError
    return eval_tree


def constant(depth: int, value: Node) -> Tree[Node]:
    if depth == 0:
        return Leaf(value)
    return Binary(value, constant(depth - 1, value), constant(depth - 1, value))


def positionally_encode(tree: Tree[Node]) -> Tree[int]:
    def go(_tree: Tree[Node], parent: int) -> Tree[int]:
        match _tree:
            case Leaf(_):
                return Leaf(parent)
            case Binary(_, left, right):
                return Binary(parent, go(left, 2 * parent), go(right, 2 * parent + 1))
        raise TypeError
    return go(tree, 1)


def descendant_nodes(tree: Tree[Node]) -> Tree[list[Node]]:
    def go(_tree: Tree[Node], history: list[Node]) -> Tree[list[Node]]:
        match _tree:
            case Leaf(_): return Leaf(history)
            case Binary(node, left, right):
                return Binary(history, go(left, history + [node]), go(right, history + [node]))  # type: ignore
    return go(tree, [])


class TreeGenerator(Generic[Node]):
    def __init__(self, leaves: set[Node], operators: set[Node]):
        self.leaves = list(leaves)
        self.operators = list(operators)

    def random_tree(self, depth: int) -> Tree[Node]:
        if depth == 0:
            return Leaf(choice(self.leaves))
        return Binary(choice(self.operators),
                      self.random_tree(ldepth := randint(0, depth - 1)),
                      self.random_tree(depth - 1 if ldepth < (depth - 1) else randint(0, depth - 1)))

    def generate(self, depth: int, num_trees: int) -> Iterator[Tree[Node]]:
        yield from (self.random_tree(depth) for _ in range(num_trees))


example_tree: Tree[int] = Binary(1, Binary(2, Leaf(4), Leaf(5)), Binary(3, Leaf(6), Leaf(7)))