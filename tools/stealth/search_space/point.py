import abc
from typing import Any


class SearchSpacePoint(abc.ABC):
    @abc.abstractmethod
    def __hash__(self) -> int:
        ...
    
    @abc.abstractmethod
    def __eq__(self, other: Any) -> bool:
        ...
