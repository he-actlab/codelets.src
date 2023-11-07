import abc
from typing import Optional
import math
from stealth.search_space import SearchSpace, SearchSpacePoint


class Searcher(abc.ABC):
    _search_spaces: tuple[SearchSpace]

    def __init__(self, search_spaces: tuple[SearchSpace]) -> None:
        super().__init__()
        self._search_spaces = search_spaces
    
    @abc.abstractmethod
    def get_next_search_space_point(self) -> Optional[tuple[SearchSpacePoint, ...]]:
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        ...
    
    def get_size_of_search_space(self) -> int:
        return math.prod(len(search_space) for search_space in self._search_spaces)

    def __iter__(self):
        return self
    
    def __next__(self) -> tuple[SearchSpacePoint, ...]:
        ret = self.get_next_search_space_point()
        if ret is None:
            raise StopIteration
        return ret
