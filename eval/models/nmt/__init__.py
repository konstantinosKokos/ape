from .applicative import MTUnitary
from .vanilla import MTVanilla
from .rotary import MTRotary
from .relative import MTRelative
from .absolute import MTAbsolute
from enum import Enum, auto


class Model(Enum):
    Unitary = auto()
    Sinusoidal = auto()
    Rotary = auto()
    Relative = auto()
    Absolute = auto()
