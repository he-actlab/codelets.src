from .point import SearchSpacePoint
from .space import SearchSpace
from typing import Any, Union
import random


class TileSpacePoint(SearchSpacePoint):
    _number_of_tiles: int
    _tile_size: int

    def __init__(self, number_of_tiles: int, tile_size: int) -> None:
        super().__init__()
        self._number_of_tiles = number_of_tiles
        self._tile_size = tile_size
    
    @property
    def number_of_tiles(self) -> int:
        return self._number_of_tiles
    
    @property
    def tile_size(self) -> int:
        return self._tile_size
    
    def __hash__(self) -> int:
        return hash((self._number_of_tiles, self._tile_size))
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TileSpacePoint):
            return False
        return self._number_of_tiles == other._number_of_tiles and self._tile_size == other._tile_size

    def __repr__(self) -> str:
        return f"({self._number_of_tiles}, {self._tile_size})"


class TileSpace(SearchSpace):
    _dimension_sizes: tuple[int, ...]

    def __init__(self, dimension_sizes: Union[int, tuple[int, ...]]) -> None:
        super().__init__()
        self._dimension_sizes = tuple(dimension_sizes) if isinstance(dimension_sizes, (tuple, list, set)) else (dimension_sizes, )
    
    def get_space(self) -> set[TileSpacePoint]:
        space = set()
        for dimension_size in self._dimension_sizes:
            for i in range(1, dimension_size + 1):
                if dimension_size % i == 0:
                    space.add(TileSpacePoint(dimension_size // i, i))
        return space
    
    def get_random_point(self) -> TileSpacePoint:
        dimension_size = random.choice(self._dimension_sizes)
        possible_number_of_tiles = list(range(1, dimension_size + 1))
        random.shuffle(possible_number_of_tiles)
        for i in possible_number_of_tiles:
            if dimension_size % i == 0:
                return TileSpacePoint(dimension_size // i, i)
        raise Exception("No valid tile size found")

    def get_cache_key(self) -> Any:
        return self._dimension_sizes
