from stealth.search_space import SearchSpace, SearchSpacePoint
from .base import Searcher
from typing import Iterator, Optional
import random


class ExhaustiveSearcher(Searcher):
    _search_space_contents: list[list[SearchSpacePoint]]
    _search_space_iterators: list[Iterator[SearchSpacePoint]]
    _current_output: Optional[list[SearchSpacePoint]]

    def __init__(self, search_spaces: list[SearchSpace], shuffle: bool = False) -> None:
        super().__init__(search_spaces)
        self._search_space_contents = [list(search_space.get_space()) for search_space in self._search_spaces]
        if shuffle:
            for search_space in self._search_space_contents:
                random.shuffle(search_space)
        self._search_space_iterators = [iter(search_space) for search_space in self._search_space_contents]
        self._current_output = None
    
    def reset(self) -> None:
        self._search_space_iterators = [iter(search_space) for search_space in self._search_space_contents]

    def get_next_search_space_point(self) -> Optional[tuple[SearchSpacePoint, ...]]:
        if self._current_output is None:
            self._current_output = [next(iterator) for iterator in self._search_space_iterators]
        else:
            i = 0
            while i < len(self._search_space_contents):
                try:
                    self._current_output[i] = next(self._search_space_iterators[i])
                    break
                except StopIteration:
                    self._search_space_iterators[i] = iter(self._search_space_contents[i])
                    self._current_output[i] = next(self._search_space_iterators[i])
                    i += 1
            if i == len(self._search_space_contents):
                return None
        return tuple(o for o in self._current_output)
