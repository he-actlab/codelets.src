import dataclasses
from typing import Union
import numpy as np


class InterpreterOperand:
    _is_readable: bool
    _is_writable: bool
    _value: np.ndarray

    def __init__(self, value: np.ndarray, is_readable: bool = True, is_writable: bool = True) -> None:
        self._value = value
        self._is_readable = is_readable
        self._is_writable = is_writable
    
    @property
    def shape(self) -> tuple[int]:
        return self._value.shape
    
    @property
    def size(self) -> int:
        return self._value.size
    
    def __getitem__(self, index: Union[int, tuple[int, ...], slice, tuple[slice, ...]]):
        if self._is_readable:
            return self._value[index].copy()
        else:
            raise RuntimeError("Operand is not readable")
    
    def __setitem__(self, index: Union[int, tuple[int, ...], slice, tuple[slice, ...]], value: np.ndarray):
        if self._is_writable:
            self._value[index] = value.copy()
        else:
            raise RuntimeError("Operand is not writable")
    
    def __str__(self) -> str:
        return str(self._value)
    

@dataclasses.dataclass
class Arguments:
    inputs: list[InterpreterOperand]


