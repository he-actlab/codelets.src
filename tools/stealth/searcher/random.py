from stealth.search_space import SearchSpace, SearchSpacePoint
from .base import Searcher
from typing import Optional
from threading import Thread, Semaphore


class RandomSearcher(Searcher):
    _allow_repeats: bool
    _seen_points: set[tuple[SearchSpacePoint, ...]]
    _max_attempts: Optional[int]

    def __init__(self, search_spaces: list[SearchSpace], allow_repeats=False, max_attempts: Optional[int] = 100) -> None:
        super().__init__(search_spaces)
        self._allow_repeats = allow_repeats
        self._seen_points = set()
        self._max_attempts = max_attempts
    
    def reset(self) -> None:
        self._seen_points = set()
    
    def get_next_search_space_point(self) -> Optional[tuple[SearchSpacePoint, ...]]:
        def get_unique_point() -> Optional[tuple[SearchSpacePoint, ...]]:
            point = tuple(search_space.get_random_point() for search_space in self._search_spaces)
            if not self._is_point_seen(point):
                self._seen_points.add(point)
                return point
            else:
                return None

        if self._max_attempts is None:
            while True:
                ret = get_unique_point()
                if ret is not None:
                    return ret 
        else:
            for _ in range(self._max_attempts):
                ret = get_unique_point()
                if ret is not None:
                    return ret
        return None
    
    def _is_point_seen(self, point: tuple[SearchSpacePoint, ...]) -> bool:
        if self._allow_repeats:
            return False
        else:
            return point in self._seen_points
    