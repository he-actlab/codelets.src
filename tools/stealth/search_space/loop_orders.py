from .point import SearchSpacePoint
from .space import SearchSpace
from typing import Any
import random
import itertools


class LoopOrderSpacePoint(SearchSpacePoint):
    _loop_order: tuple[int, ...]

    def __init__(self, loop_order: tuple[int, ...]) -> None:
        super().__init__()
        self._loop_order = loop_order
    
    @property
    def loop_order(self) -> tuple[int, ...]:
        return self._loop_order
    
    def __hash__(self) -> int:
        return hash(self._loop_order)
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, LoopOrderSpacePoint):
            return False
        return self._loop_order == other._loop_order

    def __repr__(self) -> str:
        return str(self._loop_order)


class LoopOrderSpace(SearchSpace):
    _number_of_loops: int

    def __init__(self, number_of_loops: int) -> None:
        super().__init__()
        self._number_of_loops = number_of_loops
    
    def get_space(self) -> set[LoopOrderSpacePoint]:
        space = set()
        for permutation in itertools.permutations(range(self._number_of_loops)):
            space.add(LoopOrderSpacePoint(permutation))
        return space
    
    def get_random_point(self) -> LoopOrderSpacePoint:
        loop_order = list(range(self._number_of_loops))
        random.shuffle(loop_order)
        return LoopOrderSpacePoint(tuple(l for l in loop_order))

    def get_cache_key(self) -> Any:
        return self._number_of_loops
