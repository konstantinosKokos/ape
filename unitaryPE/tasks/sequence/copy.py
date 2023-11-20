from dataclasses import dataclass
import random
from .task import SequentialTask, SequentialSample


@dataclass
class SequenceCopy(SequentialTask):
    vocab_size: int

    def __post_init__(self):
        self.sos_token_id = 0
        self.eos_token_id = self.vocab_size + 1

    def sample(self, length: int) -> SequentialSample:
        word = tuple(random.randint(1, self.vocab_size) for _ in range(length))
        word = (self.sos_token_id, *word, self.eos_token_id)
        return SequentialSample(x=word, y=word, task=self)


@dataclass
class SequenceRepeat(SequenceCopy):
    num_repeats: int

    def sample(self, length: int) -> SequentialSample:
        word = tuple(random.randint(1, self.vocab_size - 1) for _ in range(length))
        return SequentialSample(x=word, y=word * self.num_repeats, task=self)


@dataclass
class SequenceReverse(SequenceCopy):
    def sample(self, length: int) -> SequentialSample:
        word = tuple(random.randint(1, self.vocab_size - 1) for _ in range(length))
        return SequentialSample(x=word, y=word[::-1], task=self)
