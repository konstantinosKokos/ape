from .algebraic import TreeAlgebraic
from .shiv_quirk import ShivQuirk
from .note import NoTE

from enum import Enum, auto


class Model(Enum):
    Unitary = auto()
    ShivQuirk = auto()
    NoTE = auto()
