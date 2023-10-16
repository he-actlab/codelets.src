import enum
import dataclasses


@dataclasses.dataclass
class LoopIterator:
    iteration: int
    number_of_iterations: int
    stride: int


class LoopStatus(enum.Enum):
    CONTINUE = enum.auto()
    BREAK = enum.auto()


class IteratorTable:
    _iterators: dict[str, LoopIterator]

    def __init__(self) -> None:
        self._iterators = {}
    
    def reset(self) -> None:
        self._iterators = {}
    
    def add_iterator(self, name: str, number_of_iterations: int, stride: int) -> None:
        self._iterators[name] = LoopIterator(0, number_of_iterations, stride)
    
    def get_iterator_value(self, name: str) -> int:
        return self._iterators[name].iteration * self._iterators[name].stride
    
    def iterate_iterator(self, name: str) -> LoopStatus:
        self._iterators[name].iteration += 1 
        if self._iterators[name].iteration >= self._iterators[name].number_of_iterations:
            return LoopStatus.BREAK
        else:
            return LoopStatus.CONTINUE
