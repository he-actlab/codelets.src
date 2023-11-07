from .space import SearchSpace
from .integers import IntegerSpacePoint
from typing import Callable
import random


class DimensionSizeSpacePoint(IntegerSpacePoint):
    def __init__(self, dimension_size: int) -> None:
        super().__init__(dimension_size) 
    
    @property
    def dimension_size(self) -> int:
        assert self.value > 0, f"Dimension size must be positive, got {self.value}" 
        return self.value


class DimensionSizeSpace(SearchSpace):
    _generating_functions: tuple[Callable[[], set[int]], ...]

    def __init__(self, generating_functions: tuple[Callable[[], set[int]], ...]) -> None:
        super().__init__()
        self._generating_functions = generating_functions

    def get_space(self) -> set[DimensionSizeSpacePoint]:
        space = set()
        for generating_function in self._generating_functions:
            space.update(DimensionSizeSpacePoint(i) for i in generating_function())
        return space
    
    def get_random_point(self) -> DimensionSizeSpacePoint:
        generating_function = random.choice(self._generating_functions) 
        return DimensionSizeSpacePoint(random.choice(list(generating_function())))
