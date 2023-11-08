from collections import ChainMap
import numpy as np
from ..core import *
from ..expression import StealthLiteral, evaluate_expression
from .arguments import Arguments, InterpreterOperand
from .state import State
from .iterator_table import IteratorTable, LoopStatus


def _sequence_of_expressions_to_sequence_of_integers(expressions: list[StealthExpression], name_to_value_map: dict[str, int]) -> tuple[int, ...]:
    ret: list[int] = []
    for expression in expressions:
        evaluated_expression: StealthExpression = evaluate_expression(expression, name_to_value_map)
        if isinstance(evaluated_expression, StealthLiteral):
            ret.append(evaluated_expression.value)
        else:
            raise RuntimeError(f"Expected a literal, but got {evaluated_expression}")
    return tuple(ret)


def _perform_compute_unit_operation(operation_name: str, compute_unit: str, arguments: list[np.ndarray], accumulator: np.ndarray) -> np.ndarray:
    if operation_name == "mvmul":
        if len(arguments) != 3:
            raise RuntimeError(f"Expected 3 arguments, but got {len(arguments)}")
        if len(arguments[0].shape) != 2 and arguments[0].shape[0] != 1:
            raise RuntimeError(f"Expected a 2D array with shape (1, N), but got {arguments[0].shape}")
        if len(arguments[1].shape) != 2 and arguments[0].shape[1] != arguments[1].shape[0]:
            raise RuntimeError(f"Expected a 2D array with shape (N, M), but got {arguments[1].shape}")
        if len(arguments[2].shape) != 2 and arguments[2].shape != arguments[0].shape:
            raise RuntimeError(f"Expected a 2D array with shape (1, M), but got {arguments[2].shape}")
        
        if compute_unit == "PE_ARRAY":
            if arguments[0].dtype != np.int8:
                raise RuntimeError(f"Expected an array with dtype int8, but got {arguments[0].dtype}")
            if arguments[1].dtype != np.int8:
                raise RuntimeError(f"Expected an array with dtype int8, but got {arguments[1].dtype}")
            if arguments[2].dtype != np.int32:
                raise RuntimeError(f"Expected an array with dtype int32, but got {arguments[2].dtype}")
        return np.matmul(np.int32(arguments[0]), np.int32(arguments[1])) + arguments[2]
    elif operation_name == "mvmul_bias":
        if len(arguments) != 4:
            raise RuntimeError(f"Expected 4 arguments, but got {len(arguments)}")
        if len(arguments[0].shape) != 2 and arguments[0].shape[0] != 1:
            raise RuntimeError(f"Expected a 2D array with shape (1, N), but got {arguments[0].shape}")
        if len(arguments[1].shape) != 2 and arguments[0].shape[1] != arguments[1].shape[0]:
            raise RuntimeError(f"Expected a 2D array with shape (N, M), but got {arguments[1].shape}")
        if len(arguments[2].shape) != 2 and arguments[2].shape != arguments[0].shape:
            raise RuntimeError(f"Expected a 2D array with shape (1, M), but got {arguments[2].shape}")
        if len(arguments[3].shape) != 2 and arguments[3].shape != arguments[0].shape:
            raise RuntimeError(f"Expected a 2D array with shape (1, M), but got {arguments[3].shape}")
        
        if compute_unit == "PE_ARRAY":
            if arguments[0].dtype != np.int8:
                raise RuntimeError(f"Expected an array with dtype int8, but got {arguments[0].dtype}")
            if arguments[1].dtype != np.int8:
                raise RuntimeError(f"Expected an array with dtype int8, but got {arguments[1].dtype}")
            if arguments[2].dtype != np.int32:
                raise RuntimeError(f"Expected an array with dtype int32, but got {arguments[2].dtype}")
            if arguments[3].dtype != np.int32:
                raise RuntimeError(f"Expected an array with dtype int32, but got {arguments[3].dtype}")
        return np.matmul(np.int32(arguments[0]), np.int32(arguments[1])) + arguments[2] + arguments[3] 
    elif operation_name in ["add", "sub", "mul", "div", "max", "min", "macc"]:
        if len(arguments) != 2:
            raise RuntimeError(f"Expected 2 arguments, but got {len(arguments)}")
        
        if operation_name == "add":
            return np.add(arguments[0], arguments[1], dtype=np.int32)
        elif operation_name == "sub":
            return np.subtract(arguments[0], arguments[1], dtype=np.int32)
        elif operation_name == "mul":
            return np.multiply(arguments[0], arguments[1], dtype=np.int32)
        elif operation_name == "div":
            return np.floor_divide(arguments[0], arguments[1], dtype=np.int32)
        elif operation_name == "max":
            return np.maximum(arguments[0], arguments[1])
        elif operation_name == "min":
            return np.minimum(arguments[0], arguments[1])
        elif operation_name == "macc":
            return arguments[0] * arguments[1] + accumulator
        else:
            raise RuntimeError(f"Unknown operation: {operation_name}")
    elif operation_name in ["relu", "sigmoid", "tanh", "sqrt", "inv_sqrt"]:
        if len(arguments) != 1:
            raise RuntimeError(f"Expected 1 argument, but got {len(arguments)}")
        
        if operation_name == "relu":
            return np.maximum(arguments[0], 0)
        elif operation_name == "sigmoid":
            return 1 / (1 + np.exp(-arguments[0]))
        elif operation_name == "tanh":
            return np.tanh(arguments[0])
        elif operation_name == "sqrt":
            return np.sqrt(arguments[0])
        elif operation_name == "inv_sqrt":
            return 1 / np.sqrt(arguments[0])
        else:
            raise RuntimeError(f"Unknown operation: {operation_name}")
    else:
        raise NotImplementedError(f"Unknown operation: {operation_name}")


class Interpreter:
    _array_n: int
    _array_m: int

    _state: State
    _operands: dict[str, StealthOperand]
    _locals: ChainMap
    _iterator_table: IteratorTable 

    def __init__(self, array_n: int, array_m: int) -> None:
        self._array_n = array_n
        self._array_m = array_m

        self._state = State(self._array_n, self._array_m)
        self._operands = {}
        self._locals = ChainMap()
        self._iterator_table = IteratorTable()
        self._accumulator = np.zeros((1, self._array_n), dtype=np.int32)
    
    def reset(self) -> None:
        self._state.reset()
        self._operands = {}
        self._locals = ChainMap()
        self._iterator_table.reset()
        self.reset_accumulator()
    
    def reset_accumulator(self) -> None:
        self._accumulator = np.zeros((1, self._array_n), dtype=np.int32)

    def interpret(self, codelet: StealthCodelet, arguments: Arguments) -> tuple[InterpreterOperand, ...]:
        self._operands = codelet._operands.copy()
        for immediate_name, immediate_value in codelet._immediates.items():
            self._locals[immediate_name] = InterpreterOperand(np.array([immediate_value.value] * self._array_n), is_writable=False)
        if len(codelet._inputs) != len(arguments.inputs):
            raise RuntimeError(f"Expected {len(codelet._inputs)} inputs, but got {len(arguments.inputs)}")
        for input_operand, interpreter_operand in zip(codelet._inputs, arguments.inputs):
            self._state.place_operand_in_memory(input_operand.name, interpreter_operand, input_operand.location)
            self._locals[input_operand.name] = interpreter_operand
        self.interpret_statements(codelet._statements)
        return tuple(self._locals[output.name] for output in codelet._outputs)

    def interpret_statements(self, statements: list[StealthStatement]) -> None:
        for statement in statements:
            self.interpret_statement(statement)
    
    def interpret_statement(self, statement: StealthStatement):
        if isinstance(statement, StealthAllocation):
            self.interpret_allocation(statement)
        elif isinstance(statement, StealthLoad):
            self.interpret_load(statement)
        elif isinstance(statement, StealthStore):
            self.interpret_store(statement)
        elif isinstance(statement, StealthLoop):
            self.interpret_loop(statement)
        elif isinstance(statement, StealthCompute):
            self.interpret_compute(statement)
        else:
            raise RuntimeError(f"Unknown statement type: {statement}")
    
    def interpret_allocation(self, statement: StealthAllocation) -> None:
        operand_name = statement.operand_name
        allocated_operand: InterpreterOperand = self._state.allocate(operand_name, _sequence_of_expressions_to_sequence_of_integers(statement.size, {}), statement.dtype, statement.location)
        self._locals[operand_name] = allocated_operand

    def interpret_load(self, statement: StealthLoad) -> None:
        if statement.source_operand_name not in self._locals:
            raise RuntimeError(f"Unknown operand: {statement.source_operand_name}")
        destination_name: str = statement.destination_operand_name
        if destination_name in self._locals:
            raise RuntimeError(f"Destination operand {destination_name} already exists")
        
        load_size: tuple[int, ...] = _sequence_of_expressions_to_sequence_of_integers(statement.size, {})
        load_offset: tuple[int, ...] = _sequence_of_expressions_to_sequence_of_integers(statement.source_operand_offset, self._get_all_current_loop_index_values())
        loaded_operand: InterpreterOperand = self._state.load(statement.source_operand_name, load_offset, load_size, self._operands[statement.source_operand_name].location) 
        self._state.place_operand_in_memory(destination_name, loaded_operand, statement.location)
        self._locals[destination_name] = loaded_operand

    def interpret_store(self, statement: StealthStore) -> None:
        if statement.source_operand_name not in self._locals:
            raise RuntimeError(f"Unknown operand: {statement.source_operand_name}")
        if statement.destination_operand_name not in self._locals:
            raise RuntimeError(f"Unknown operand: {statement.destination_operand_name}")

        store_offset: tuple[int, ...] = _sequence_of_expressions_to_sequence_of_integers(statement.destination_operand_offset, self._get_all_current_loop_index_values()) 
        self._state.store(statement.destination_operand_name, self._locals[statement.source_operand_name], store_offset, self._operands[statement.destination_operand_name].location)

    def interpret_loop(self, statement: StealthLoop) -> None:
        if not isinstance(statement.number_of_iterations, StealthLiteral):
            raise RuntimeError(f"Expected a literal, but got {statement.number_of_iterations}")
        if not isinstance(statement.stride, StealthLiteral):
            raise RuntimeError(f"Expected a literal, but got {statement.stride}")
         
        self._add_loop_index(statement.loop_index_variable_name, statement.number_of_iterations.value, statement.stride.value)
        status = LoopStatus.CONTINUE
        while status == LoopStatus.CONTINUE:
            self._locals = self._locals.new_child()
            for loop_statement in statement.body:
                self.interpret_statement(loop_statement)
            status = self._iterate_loop(statement.loop_index_variable_name)
            self._locals = self._locals.parents
    
    def interpret_compute(self, statement: StealthCompute) -> None:
        compute_unit: str = statement.location
        arguments: list[np.ndarray] = []
        for operand in statement.operands:
            if not isinstance(operand, str):
                raise RuntimeError(f"Expected a string, but got {operand}")
            arguments.append(self._locals[operand]._value)

        operation_name = statement.operation_name
        output_array = _perform_compute_unit_operation(operation_name, compute_unit, arguments, self._accumulator)
        output_operand = InterpreterOperand(output_array, is_writable=True)
        self._locals[statement.destination_operand_name] = output_operand
    
    def expression_offsets_to_integer_offset(self, expressions: list[StealthExpression], operand_name: str) -> int:
        integer_offsets: tuple[int] = _sequence_of_expressions_to_sequence_of_integers(expressions, self._get_all_current_loop_index_values())
        dim_sizes: list[StealthLiteral] = []
        current_prevous_dim_size = StealthLiteral(1)
        for dim_size in reversed(self._operands[operand_name].shape):
            dim_sizes.append(current_prevous_dim_size)
            if not isinstance(dim_size, StealthLiteral):
                raise RuntimeError(f"Expected a literal, but got {dim_size}")
            current_prevous_dim_size = StealthLiteral(current_prevous_dim_size.value * dim_size.value)
        dim_sizes.reverse()
        
        ret: int = 0
        for integer_offset, dim_size in zip(integer_offsets, dim_sizes):
            ret += integer_offset * dim_size.value
        return ret
    
    def expression_sizes_to_integer_size(self, expressions: list[StealthExpression]) -> int:
        integer_sizes: tuple[int] = _sequence_of_expressions_to_sequence_of_integers(expressions, {})
        ret: int = 1
        for integer_size in integer_sizes:
            ret *= integer_size
        return ret
    
    def _add_loop_index(self, loop_index_name: str, loop_index_end: int, loop_index_stride: int) -> None:
        self._iterator_table.add_iterator(loop_index_name, loop_index_end, loop_index_stride)
    
    def _iterate_loop(self, loop_index_name: str) -> LoopStatus:
        return self._iterator_table.iterate_iterator(loop_index_name)
    
    def _get_all_current_loop_index_values(self) -> dict[str, int]:
        return {name: self._iterator_table.get_iterator_value(name) for name in self._iterator_table._iterators.keys()}

    def _get_current_loop_index_value(self, loop_index_name: str) -> int:
        return self._iterator_table.get_iterator_value(loop_index_name)


def interpret(codelet: StealthCodelet, inputs: tuple[np.ndarray], array_n: int, array_m: int) -> tuple[np.ndarray]:
    interpreter = Interpreter(array_n, array_m)
    interpreter.reset()
    input_operands: list[InterpreterOperand] = [InterpreterOperand(input_array, is_writable=False) for input_array in inputs]
    arguments = Arguments(input_operands)
    output_arrays = tuple(output._value for output in interpreter.interpret(codelet, arguments))
    return output_arrays
