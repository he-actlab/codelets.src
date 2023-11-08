import numpy as np
from .arguments import InterpreterOperand


def _pad_tuple(t: tuple[int, ...], length: int, pad_value: int = 0) -> tuple[int, ...]:
    return (pad_value,) * (length - len(t)) + t


def _get_padded_size_and_offset(size: tuple[int, ...], offset: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    indexed_shape_length = len(offset)
    size = _pad_tuple(size, indexed_shape_length, pad_value=1)
    offset = _pad_tuple(offset, indexed_shape_length)
    return size, offset


class Memory:
    _DTYPE_TO_NP_DTYPE: dict[str, str] = {
        "i8": "int8",
        "i32": "int32",
    }

    _name: str
    _memory: dict[str, InterpreterOperand]

    def __init__(self, name: str) -> None:
        self._name = name
        self._memory = {}

    def reset(self) -> None:
        self._memory = {}

    def allocate(self, name: str, shape: tuple[int, ...], dtype: str) -> InterpreterOperand:
        new_operand = InterpreterOperand(np.zeros(shape, dtype=self._DTYPE_TO_NP_DTYPE[dtype]))
        self._memory[name] = new_operand
        return new_operand

    def load(self, source_operand_name: str, offset: tuple[int, ...], size: tuple[int, ...]) -> InterpreterOperand:
        if source_operand_name not in self._memory:
            raise RuntimeError(f"{self._name} does not contain a variable named {source_operand_name}")
        if isinstance(size, tuple) and isinstance(offset, tuple):
            assert all(isinstance(s, int) for s in size)
            assert all(isinstance(o, int) for o in offset)
            size, offset = _get_padded_size_and_offset(size, offset)
            assert len(offset) == len(size[len(size) - len(offset):]) 
            indices = tuple(slice(o, o + s) for o, s in zip(offset, size[len(size) - len(offset):]))
            loaded_value: np.ndarray = self._memory[source_operand_name][indices]
            loaded_value = loaded_value.reshape(size)
        else:
            raise RuntimeError(f"Cannot load {size} elements from {offset} in {self._name}")
        return InterpreterOperand(loaded_value)

    def store(self, destination_operand_name: str, source_operand: InterpreterOperand, offset: tuple[int, ...]) -> None:
        if destination_operand_name not in self._memory:
            raise RuntimeError(f"{self._name} does not contain a variable named {destination_operand_name}")
        size: tuple[int] = source_operand.shape
        if isinstance(offset, tuple):
            assert all(isinstance(o, int) for o in offset)
            size, offset = _get_padded_size_and_offset(size, offset)
            indices = tuple(slice(o, o + s) for o, s in zip(offset, size))
            self._memory[destination_operand_name][indices] = source_operand._value
        else:
            raise RuntimeError(f"Cannot store {size} elements from {offset} in {self._name}")
    
    def is_operand_in_memory(self, name: str) -> bool:
        return name in self._memory


class SIMDMemory(Memory):
    _size: int

    def __init__(self, size: int) -> None:
        super().__init__("SIMD")
        self._size = size
    
    def allocate(self, name: str, shape: tuple[int, ...], dtype: str) -> InterpreterOperand:
        raise RuntimeError("Cannot allocate SIMD memory")
    
    def load(self, source_operand_name: str, offset: tuple[int, ...], size: tuple[int, ...], destination_operand_name: str) -> InterpreterOperand:
        if size != (self._size,):
            raise RuntimeError(f"Cannot load {size} elements into SIMD memory")
        return super().load(source_operand_name, offset, size, destination_operand_name)


class SystolicArrayMemory(Memory):
    _array_n: int
    _array_m: int

    def __init__(self, array_n: int, array_m: int) -> None:
        super().__init__("PE_ARRAY")
        self._array_n = array_n
        self._array_m = array_m

    def allocate(self, name: str, shape: tuple[int, ...], dtype: str) -> InterpreterOperand:
        raise RuntimeError("Cannot allocate systolic array memory")
    
    def load(self, source_operand_name: str, offset: tuple[int, ...], size: tuple[int, ...], destination_operand_name: str) -> InterpreterOperand:
        if len(size) != 2:
            raise RuntimeError(f"Cannot load {size} elements into systolic array memory")
        return super().load(source_operand_name, offset, size, destination_operand_name)
    

class State:
    _memories: dict[str, Memory] 

    def __init__(self, array_n: int, array_m: int) -> None:
        self._memories = {
            "DRAM": Memory("DRAM"),
            "IBUF": Memory("IBUF"),
            "WBUF": Memory("WBUF"),
            "BBUF": Memory("BBUF"),
            "OBUF": Memory("OBUF"),
            "VMEM1": Memory("VMEM1"),
            "VMEM2": Memory("VMEM2"),
            "SIMD": SIMDMemory(array_n),
            "PE_ARRAY": SystolicArrayMemory(array_n, array_m)
        }

    def reset(self) -> None:
        for memory in self._memories.values():
            memory.reset()
    
    def allocate(self, name: str, shape: tuple[int, ...], dtype: str, memory: str) -> InterpreterOperand:
        return self._memories[memory].allocate(name, shape, dtype)

    def load(self, source_operand_name: str, offset: tuple[int, ...], size: tuple[int, ...], memory: str) -> InterpreterOperand:
        return self._memories[memory].load(source_operand_name, offset, size)
    
    def store(self, destination_operand_name: str, value: InterpreterOperand, offset: tuple[int, ...], memory: str) -> None:
        self._memories[memory].store(destination_operand_name, value, offset)
    
    def place_operand_in_memory(self, name: str, operand: InterpreterOperand, memory: str) -> None:
        self._memories[memory]._memory[name] = operand
    
    def is_operand_in_memory(self, name: str, memory: str) -> bool:
        return self._memories[memory].is_operand_in_memory(name)
    