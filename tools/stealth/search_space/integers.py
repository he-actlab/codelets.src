from .point import SearchSpacePoint
from .space import SearchSpace
from typing import Any
import random


class IntegerSpacePoint(SearchSpacePoint):
    _value: int

    def __init__(self, value: int) -> None:
        super().__init__()
        self._value = value
    
    @property
    def value(self) -> int:
        return self._value
    
    def __hash__(self) -> int:
        return hash(self._value)
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, IntegerSpacePoint):
            return False
        return self._value == other._value

    def __repr__(self) -> str:
        return str(self._value)


class IntegerSpace(SearchSpace):
    _min_value: int
    _max_value: int

    def __init__(self, min_value: int, max_value: int) -> None:
        super().__init__()
        self._min_value = min_value
        self._max_value = max_value
    
    def get_space(self) -> set[IntegerSpacePoint]:
        return set(IntegerSpacePoint(i) for i in range(self._min_value, self._max_value + 1))
    
    def get_random_point(self) -> IntegerSpacePoint:
        return IntegerSpacePoint(random.randint(self._min_value, self._max_value))

    def get_cache_key(self) -> Any:
        return (self._min_value, self._max_value)
