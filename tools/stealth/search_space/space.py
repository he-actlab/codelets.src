import abc
from typing import Any
from .point import SearchSpacePoint


class SearchSpace(abc.ABC):
    _cache: dict[Any, set[SearchSpacePoint]] = {}

    _iter_index: int

    def __init__(self) -> None:
        super().__init__()
        self._iter_index = 0

    @abc.abstractmethod
    def get_space(self) -> set[SearchSpacePoint]:
        ...
    
    @abc.abstractmethod
    def get_cache_key(self) -> Any:
        ...
    
    @abc.abstractmethod
    def get_random_point(self) -> SearchSpacePoint:
        ...
    
    def __len__(self) -> int:
        return len(self.get_space())
    
    def __iter__(self):
        self._iter_index = 0
        return self
    
    def __next__(self) -> SearchSpacePoint:
        if self.get_cache_key() in self._cache:
            space = self._cache[self.get_cache_key()]
        else:
            space = self.get_space()
        
        if self._iter_index >= len(space):
            self._iter_index = 0
            raise StopIteration
        value = list(space)[self._iter_index]
        self._iter_index += 1
        return value
