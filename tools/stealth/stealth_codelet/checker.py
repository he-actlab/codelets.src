from typing import Optional, Union
import math

from tools.stealth.stealth_codelet.core import StealthLoop
from .core import *
from .expression import *
from .tiling_collector import collect_tiling
from .visitor import StealthCodeletVisitor
from .variable_substituter import substitute_variables


class TilingSplitsErrorMessageGetter(StealthCodeletVisitor):
    _compile_time_params: dict[str, int]
    _error_message: Optional[str]

    def __init__(self, compile_time_params: dict[str, int]) -> None:
        self._compile_time_params = compile_time_params.copy()
        self._error_message = None

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message
    
    def visit(self, codelet: StealthCodelet) -> None:
        self._error_message = None
        super().visit(codelet)

    def visit_loop(self, statement: StealthLoop) -> None:
        number_of_iterations = statement.number_of_iterations
        evaluated_number_of_iterations = evaluate_expression(number_of_iterations, self._compile_time_params)
        if not isinstance(evaluated_number_of_iterations, StealthLiteral):
            raise RuntimeError(f"Number of iterations {number_of_iterations} is not able to be set at compile-time.")
        if evaluated_number_of_iterations.value <= 0:
            self._error_message = f"Number of iterations {number_of_iterations} for loop \"{statement.loop_index_variable_name} in ({statement.number_of_iterations, {statement.stride}})\" evaluates to {evaluated_number_of_iterations} at compile-time which is not greater than 0 and invalid.\nTry checking the tiling to ensure that each dimension has a valid split."
        stride = statement.stride
        evaluated_stride = evaluate_expression(stride, self._compile_time_params)
        if not isinstance(evaluated_stride, StealthLiteral):
            raise RuntimeError(f"Stride {stride} is not able to be set at compile-time.")
        if evaluated_stride.value <= 0:
            self._error_message = f"Stride {stride} for loop \"{statement.loop_index_variable_name} in ({statement.number_of_iterations, {statement.stride}})\" evaluates to {evaluated_stride} at compile-time which is not greater than 0 and invalid.\nTry checking the tiling to ensure that each dimension has a valid split."
        super().visit_loop(statement)


def _get_tiling_splits_error_message(compile_time_params: dict[str, int], codelet: StealthCodelet) -> Optional[str]:
    getter = TilingSplitsErrorMessageGetter(compile_time_params)
    getter.visit(codelet)
    return getter.error_message


# def _get_tiling_splits_error_message(operand_dim_sizes: dict[str, int], codelet: StealthCodelet) -> Optional[str]:
#     tiling: dict[str, int] = collect_tiling(codelet)
#     for operand_dim_name, operand_dim_size in operand_dim_sizes.items():
#         number_of_tiles: int = tiling[operand_dim_name]
#         if operand_dim_size % number_of_tiles != 0:
#             return f"Dimension {operand_dim_name} of operand has size {operand_dim_size}, which is not divisible by the number of tiles specified {number_of_tiles}"
#         tile_size: int = operand_dim_size // number_of_tiles
#         if tile_size == 0:
#             return f"Dimension {operand_dim_name} of operand has size {operand_dim_size}, which is less than the number of tiles specified {number_of_tiles}"
#     return None


class Memory:
    _name: str
    _content: dict[str, int]
    _capacity: int
    _alignment: int

    def __init__(self, name: str, capacity: int, alignment: int = 1) -> None:
        self._name = name
        self._content = {}
        self._capacity = capacity
        self._alignment = alignment
    
    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def current_capacity(self) -> int:
        return sum(self._content.values())
    
    @property
    def remaining_capacity(self) -> int:
        return self.capacity - self.current_capacity

    def reset(self) -> None:
        self._content = {}

    def place_in_memory(self, name: str, size: int) -> None:
        if name in self._content:
            raise RuntimeError(f"Operand {name} has already been put in {self._name}")
        aligned_size: int = self.get_aligned_size(size)
        if aligned_size > self.remaining_capacity:
            raise RuntimeError(f"Operand {name} does not fit in {self._name} as the current remaning capacity is {self.remaining_capacity} bits which is smaller than the aligned size of the operand ({aligned_size} bits)")
        self._content[name] = aligned_size
    
    def get_aligned_size(self, size: int) -> int:
        return size + (self._alignment - math.ceil(size % self._alignment))
    
    def __str__(self) -> str:
        ret: str = self._name + " contents:\n"
        for operand_name, operand_size in sorted(self._content.items(), key=lambda t: t[1]):
            ret += f"{operand_name}: {operand_size} bits\n"
        return ret

    def __in__(self, other) -> bool:
        return other in self._content


class MemoryTracker:
    _memory: dict[str, Memory]

    def __init__(self, config: dict[str, Union[int, str, bool]]) -> None:
        self._memory = {
            "DRAM": Memory("DRAM", config["DRAM_DEPTH"] * config["DRAM_BANKS"] * config["DRAM_WIDTH"], alignment=config["ADDR_ALIGNMENT"]),
            "IBUF": Memory("IBUF", config["IBUF_DEPTH"] * config["ARRAY_N"] * config["DATA_WIDTH"], alignment=8),
            "WBUF": Memory("WBUF", config["WBUF_DEPTH"] * config["ARRAY_N"] * config["WGT_WIDTH"], alignment=8),
            "BBUF": Memory("BBUF", config["BBUF_DEPTH"] * config["ARRAY_N"] * config["BIAS_WIDTH"], alignment=8),
            "OBUF": Memory("OBUF", config["OBUF_DEPTH"] * config["ARRAY_N"] * config["ACC_WIDTH"], alignment=8),
            "VMEM1": Memory("VMEM1", config["VMEM_DEPTH"] * config["VMEM_BANKS"] * config["ACC_WIDTH"], alignment=8),
            "VMEM2": Memory("VMEM2", config["VMEM_DEPTH"] * config["VMEM_BANKS"] * config["ACC_WIDTH"], alignment=8),
            "IMM": Memory("IMM", config["IMM_DEPTH"] * config["ACC_WIDTH"], alignment=8),
        }
    
    def reset(self) -> None:
        for memory in self._memory.values():
            memory.reset()
    
    def get_memory_capacity(self, location: str) -> int:
        return self._memory[location].capacity
    
    def get_remaining_memory_capacity(self, location: str) -> int:
        return self._memory[location].remaining_capacity
    
    def place_operand_in_memory(self, operand_name: str, operand_size: int, location: str) -> None:
        self._memory[location].place_in_memory(operand_name, operand_size)
    
    def get_operand_location(self, operand_name: str) -> str:
        for memory_name, memory in self._memory:
            if operand_name in memory:
                return memory_name
        raise RuntimeError(f"Operand {operand_name} not found in any memory")
    
    def get_memory_content_string(self, location: str) -> str:
        return str(self._memory[location])


# TODO: Need to do a dataflow analysis if you want to be able to overwrite things inside of a layer

class OnChipMemoryUseErrorMessageGetter(StealthCodeletVisitor):
    _VMEMS = ("VMEM1", "VMEM2")
    _BUFS = ("IBUF", "WBUF", "BBUF", "OBUF")
    _COMPUTES = ("SIMD", "PE_ARRAY")

    _config: dict[str, Union[int, str, bool]]

    _memory_tracker: MemoryTracker
    _error_message: Optional[str]
    _break: bool

    def __init__(self, config: dict[str, Union[int, str, bool]]) -> None:
        self._config = config.copy()

        self._memory_tracker = MemoryTracker(self._config)
        self._error_message = None
        self._break = False
    
    def reset(self) -> None:
        self._memory_tracker.reset()
        self._error_message = None
        self._break = False

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message
    
    @property
    def _acc_dtype_width(self) -> int:
        return self._config["ACC_WIDTH"]
    
    @property
    def _data_dtype_width(self) -> int:
        return self._config["DATA_WIDTH"]
    
    @property
    def _pe_array_width(self) -> int:
        return self._config["ARRAY_N"]
    
    @property
    def _pe_array_height(self) -> int:
        return self._config["ARRAY_M"]
    
    @property
    def _vmem_depth(self) -> int:
        return self._config["VMEM_DEPTH"]
    
    @property
    def _vmem_total_size(self) -> int:
        return self._vmem_depth * self._simd_width * self._acc_dtype_width

    @property
    def _simd_width(self) -> int:
        return self._config["ARRAY_N"]
    
    @property
    def _address_alignment(self) -> int:
        return self._config["ADDR_ALIGNMENT"]

    def visit(self, codelet: StealthCodelet) -> None:
        self.reset()
        super().visit(codelet)
    
    def visit_allocation(self, statement: StealthAllocation) -> None:
        if self._alloc_load_store_guard(statement):
            return
        
        allocated_operand_integer_dimensions = tuple(map(lambda s: s.value, statement.size)) 
        if statement.location in OnChipMemoryUseErrorMessageGetter._VMEMS: 
            allocated_operand_total_size = math.prod(allocated_operand_integer_dimensions) * self._acc_dtype_width
        elif statement.location in OnChipMemoryUseErrorMessageGetter._BUFS:
            allocated_operand_total_size = math.prod(allocated_operand_integer_dimensions) * self._data_dtype_width
        elif statement.location == "DRAM":
            allocated_operand_total_size = math.prod(allocated_operand_integer_dimensions) * self._acc_dtype_width
        
        if allocated_operand_total_size > self._memory_tracker.get_memory_capacity(statement.location):
            self._error_message = self._get_total_size_error_message(statement.operand_name, allocated_operand_integer_dimensions, allocated_operand_total_size, statement.location)
            return
        elif allocated_operand_total_size > self._memory_tracker.get_remaining_memory_capacity(statement.location):
            self._error_message = self._get_remaining_size_error_message(statement.operand_name, allocated_operand_integer_dimensions, allocated_operand_total_size, statement.location) 
            return
            
        self._memory_tracker.place_operand_in_memory(statement.operand_name, allocated_operand_total_size, statement.location)

    
    def visit_load(self, statement: StealthLoad) -> None:
        if self._alloc_load_store_guard(statement):
            return
        
        loaded_operand_integer_dimensions = tuple(map(lambda s: s.value, statement.size)) 
        if statement.location in OnChipMemoryUseErrorMessageGetter._VMEMS: 
            loaded_operand_total_size = math.prod(loaded_operand_integer_dimensions) * self._acc_dtype_width
        elif statement.location in OnChipMemoryUseErrorMessageGetter._BUFS:
            loaded_operand_total_size = math.prod(loaded_operand_integer_dimensions) * self._data_dtype_width
        
        if loaded_operand_total_size > self._memory_tracker.get_memory_capacity(statement.location):
            self._error_message = self._get_total_size_error_message(statement.destination_operand_name, loaded_operand_integer_dimensions, loaded_operand_total_size, statement.location)
            return
        elif loaded_operand_total_size > self._memory_tracker.get_remaining_memory_capacity(statement.location):
            self._error_message = self._get_remaining_size_error_message(statement.destination_operand_name, loaded_operand_integer_dimensions, loaded_operand_total_size, statement.location)
            return
        
        self._memory_tracker.place_operand_in_memory(statement.destination_operand_name, loaded_operand_total_size, statement.location)
    
    def _general_guard(self) -> bool:
        if self._break:
            return True
        else:
            return False
    
    def _alloc_load_store_guard(self, statement: Union[StealthAllocation, StealthStore, StealthLoad]) -> bool:
        if self._general_guard():
            return True

        if statement.location in OnChipMemoryUseErrorMessageGetter._COMPUTES:
            return True

        if any(not isinstance(s, StealthLiteral) for s in statement.size):
            raise TypeError(f"Expected all size fields of load {statement} to be set as integers")

        return False
    
    def _get_total_size_error_message(self, operand_name: str, dimensions: tuple[int, ...], total_size: int, location: str) -> str:
        return f"Operand {operand_name} with dimensions {dimensions} has total size {total_size} bits which is larger than the location it should be stored at ({location}) which has a capacity of {self._memory_tracker.get_memory_capacity(location)} bits" 
    
    def _get_remaining_size_error_message(self, operand_name: str, dimensions: tuple[int, ...], total_size: int, location: str) -> str:
        return f"Operand {operand_name} with dimensions {dimensions} has total size {total_size} bits which is larger than the current remaining capacity in the location it should be stored at ({location}) which has a remaining capacity of {self._memory_tracker.get_remaining_memory_capacity(location)} bits\n{self._memory_tracker.get_memory_content_string(location)}"


def get_codelet_check_error_message(codelet: StealthCodelet, operand_dim_sizes: dict[str, int], config: dict[str, Union[int, str, bool]]) -> Optional[str]:
    tiling_splits_error_message: Optional[str] = _get_tiling_splits_error_message(operand_dim_sizes, codelet)
    if tiling_splits_error_message is not None:
        return tiling_splits_error_message
    
    set_codelet: StealthCodelet = substitute_variables(codelet, operand_dim_sizes)
    getter = OnChipMemoryUseErrorMessageGetter(config)
    getter.visit(set_codelet)
    return getter.error_message
