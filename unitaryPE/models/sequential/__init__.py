from .applicative import SequentialUnitary
from .vanilla import SequentialVanilla
from .relative import SequentialRelative

from enum import Enum, auto


class Model(Enum):
    Relative = auto()
    Unitary = auto()
    Sinusoidal = auto()
