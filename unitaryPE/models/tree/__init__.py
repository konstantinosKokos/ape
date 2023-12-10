from .applicative import TreeUnitary
from .shiv_quirk import ShivQuirk

from enum import Enum, auto


class Model(Enum):
    Unitary = auto()
    ShivQuirk = auto()
