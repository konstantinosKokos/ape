from torch.nn.functional import cross_entropy
from torch import Tensor
from abc import abstractmethod, ABC


class Base(ABC):
    @abstractmethod
    def forward(self):
        ...

    def go_batch(self):
        ...