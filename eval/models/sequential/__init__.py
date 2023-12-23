from .applicative import SequentialUnitary
from .vanilla import SequentialVanilla
from .relative import SequentialRelative
from .rotary import SequentialRotary
from .absolute import SequentialAbsolute

from enum import Enum, auto


class Model(Enum):
    Relative = auto()
    Unitary = auto()
    Sinusoidal = auto()
    Rotary = auto()
    # Absolute = auto()     # todo
