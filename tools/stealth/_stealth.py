import polymath as pm
import abc
from functools import partial
from typing import Any, Callable, Iterator, Optional, Union
import dataclasses
import random
import math
import tqdm
import itertools
import os
import json
from collections import OrderedDict, defaultdict
from lark import ParseTree, Lark, Token, Tree, indenter
from lark.visitors import Interpreter
from .parse import parse_stealth_codelet
from .utils import UniqueNameGenerator
from onnx import helper, numpy_helper
from codelets.examples.genesys import compile_genesys, load_config, DataGen
from codelets.templates.codelet_template import CodeletTemplate, DummyOp, FlexParam
from codelets.adl.graph import ArchitectureGraph
from codelets.examples.genesys import DTYPE_MAP, OP_DTYPES


# ==================== Codelet Information Collection ==================== #


@dataclasses.dataclass
class ComputeOperationGroup:
    operations: list[StealthCompute]
    transfer_and_compute_operations: list[Union[StealthLoad, StealthStore, StealthCompute]]
    loop_index_variables: list[StealthIndex]
    immediates: list[str]

    @property
    def inner_loop_index_variables(self) -> list[StealthIndex]:
        return self.loop_index_variables[len(self.loop_index_variables) // 2:] 
    
    @property
    def outer_loop_index_variables(self) -> list[StealthIndex]:
        return self.loop_index_variables[:len(self.loop_index_variables) // 2]

    @property
    def is_systolic_array_operation(self) -> bool:
        if len(self.operations) == 1 and self.operations[0].location == "PE_ARRAY":
            return True
        else:
            return False
    
    @property
    def is_simd_operation(self) -> bool:
        if all(o.location == "SIMD" for o in self.operations):
            return True
        else:
            return False
    
    def __str__(self) -> str:
        ret = ""
        ret += "Compute Operation Group:\n"
        ret += "\tOperations:\n"
        for operation in self.operations:
            ret += f"\t\t{operation}\n"
        ret += "\tTransfer and Compute Operations:\n"
        for operation in self.transfer_and_compute_operations:
            ret += f"\t\t{operation}\n"
        ret += "\tLoop Index Variables:\n"
        for loop_index_variable in self.loop_index_variables:
            ret += f"\t\t{loop_index_variable}\n"
        return ret


class CodeletInformationCollector(Interpreter):
    _COMPUTE_OUTPUT_DTYPE: dict[str, str] = {
        "PE_ARRAY": "i32",
        "SIMD": "i32"
    }
    _CODELET_COMPUTE_NAME_TO_COMPUTE_OPERATION_NAME: dict[str, str] = {
        "mvmul": "MVMUL",
        "mvmul_bias": "MVMUL",
        "add": "ADD",
        "sub": "SUB",
        "mul": "MUL",
        "div": "DIV",
        "max": "MAX",
        "min": "MIN",
        "relu": "RELU",
        "tanh": "TANH",
        "sigmoid": "SIGMOID",
        "sqrt": "SQRT",
        "inv_sqrt": "INV_SQRT"
    }
    _OPERATION_TYPE_TO_OPERATION_STR: dict[str, str] = {
        "add_expr": "+",
        "sub_expr": "-",
        "mul_expr": "*",
        "div_expr": "/",
        "floordiv_expr": "//",
        "eq_expr": "==",
        "ne_expr": "!=",
        "lt_expr": "<",
        "le_expr": "<=",
        "gt_expr": ">",
        "ge_expr": ">=",
        "and_expr": "and",
        "or_expr": "or",
    }

    _codelet_string: str

    _operation_name: Optional[str]
    _used_var_names: set[str]
    _operands: dict[str, StealthOperand]
    _inputs: list[StealthOperand]
    _outputs: list[StealthOperand]
    _params: list[StealthParameter]
    _immediates: dict[str, StealthExpression]
    _loop_indices: list[StealthIndex]
    _statements: list[StealthStatement]
    _current_loop_stack: list[StealthLoop]

    def __init__(self, codelet_string: str) -> None:
        super().__init__()
        self._codelet_string = codelet_string

        self._operation_name = None
        self._used_var_names = set()
        self._operands = {}
        self._inputs = []
        self._outputs = []
        self._params = []
        self._immediates = {}
        self._loop_indices = []
        self._statements = []
        self._current_loop_stack = []

    
    # ==================== Collector Information Getters ==================== # 
    def get_operation_name(self) -> str:
        if self._operation_name is None:
            raise RuntimeError("Operation name was not set")
        return self._operation_name
    
    def get_input_operands(self) -> list[StealthOperand]:
        return self._inputs.copy()
    
    def get_output_operands(self) -> list[StealthOperand]:
        return self._outputs.copy()
    
    def get_params(self) -> list[StealthParameter]:
        return self._params.copy()
    
    def get_operand_name_to_operand_map(self) -> dict[str, StealthOperand]:
        return self._operands.copy()
    
    def get_number_of_tiles_for_each_dimension(self) -> dict[str, int]:
        loop_counter = 0
        loop_to_number_of_tiles = {}
        outer_loop_index_variables = self.get_outer_loop_index_variables()

        def helper(statement: StealthStatement):
            nonlocal loop_counter
            if isinstance(statement, StealthLoop):
                if statement.loop_index_variable_name in map(lambda i: i.name, outer_loop_index_variables):
                    number_of_tiles = evaluate_expression(statement.end)
                    assert isinstance(number_of_tiles, StealthLiteral)
                    stride = evaluate_expression(statement.stride)
                    assert isinstance(stride, StealthBinaryExpression) and stride.operation == "//" and isinstance(stride.lhs, StealthVariableName)
                    loop_to_number_of_tiles[stride.lhs.name] = number_of_tiles.value
                    loop_counter += 1
                for loop_statement in statement.body:
                    helper(loop_statement)
            
        for statement in self._statements:
            helper(statement)
        
        return loop_to_number_of_tiles
    
    def get_compute_operation_groups(self) -> list[ComputeOperationGroup]:
        compute_groups: list[list[StealthCompute]] = [[]] 

        def helper1(statement: StealthStatement):
            nonlocal compute_groups
            if isinstance(statement, StealthLoop):
                if len(compute_groups[-1]) != 0:
                    compute_groups.append([])
                for loop_statement in statement.body:
                    helper1(loop_statement)
            elif isinstance(statement, StealthCompute):
                if len(compute_groups[-1]) != 0 and compute_groups[-1][-1].location == "PE_ARRAY":
                    compute_groups.append([])
                elif len(compute_groups[-1]) != 0 and compute_groups[-1][-1].location == "SIMD" and statement.location == "PE_ARRAY":
                    compute_groups.append([])
                compute_groups[-1].append(statement)
        
        for statement in self._statements:
            helper1(statement)
        if len(compute_groups[-1]) == 0:
            compute_groups.pop()
        
        transfer_and_compute_operations: list[list[list[Union[StealthLoad, StealthStore, StealthCompute]]]] = [[[] for _ in range(len(compute_operations))] for compute_operations in compute_groups]
        compute_group_index: int = 0
        compute_operation_index: int = 0
        ready_to_move_on_to_next_operation: bool = False
        def helper2(statement: StealthStatement):
            nonlocal transfer_and_compute_operations
            nonlocal compute_group_index
            nonlocal compute_operation_index
            nonlocal ready_to_move_on_to_next_operation
            if isinstance(statement, StealthLoop):
                for loop_statement in statement.body:
                    helper2(loop_statement)
            elif isinstance(statement, (StealthLoad, StealthStore, StealthCompute)):
                if isinstance(statement, StealthStore):
                    ready_to_move_on_to_next_operation = True
                if isinstance(statement, StealthLoad) and ready_to_move_on_to_next_operation:
                    compute_operation_index += 1
                    if compute_operation_index >= len(compute_groups[compute_group_index]):
                        compute_group_index += 1
                        compute_operation_index = 0
                        ready_to_move_on_to_next_operation = False
                if compute_group_index < len(compute_groups) and compute_operation_index < len(compute_groups[compute_group_index]):
                    transfer_and_compute_operations[compute_group_index][compute_operation_index].append(statement)
        for statement in self._statements:
            helper2(statement)
        
        loop_index_variables: list[list[StealthIndex]] = [[] for _ in range(len(compute_groups))]
        for i, compute_group_transfer_and_compute_operations in enumerate(transfer_and_compute_operations):
            def helper3(statement: StealthStatement):
                nonlocal loop_index_variables
                if isinstance(statement, StealthLoop):
                    for compute_operation_transfer_and_compute_operations in compute_group_transfer_and_compute_operations:
                        for operation in compute_operation_transfer_and_compute_operations:
                            loop_index_variables_in_offset = set()
                            if isinstance(operation, StealthLoad):
                                offsets = operation.source_operand_offset
                            elif isinstance(operation, StealthStore):
                                offsets = operation.destination_operand_offset
                            else:
                                offsets = []

                            for offset in offsets:
                                loop_index_variables_in_offset.update(self.get_loop_index_variable_names_from_expression(offset))
                            if statement.loop_index_variable_name in loop_index_variables_in_offset and statement.loop_index_variable_name not in map(lambda l: l.name, loop_index_variables[i]):
                                loop_index_variables[i].append(StealthIndex(statement.loop_index_variable_name, statement.end, statement.stride))

                    for loop_statement in statement.body:
                        helper3(loop_statement) 
        
            for statement in self._statements:
                helper3(statement)
        
        immediates: list[list[str]] = [[] for _ in range(len(compute_groups))]
        def helper4(statement: StealthStatement):
            nonlocal immediates
            if isinstance(statement, StealthLoop):
                for loop_statement in statement.body:
                    helper4(loop_statement)
            elif isinstance(statement, StealthCompute):
                for operand in statement.operands:
                    if isinstance(operand, str) and operand not in self._operands:
                        if operand not in immediates[-1]:
                            immediates[-1].append(operand.name)
        
        for statement in self._statements:
            helper4(statement)

        # Flatten transfer and compute operations
        new_transfer_and_compute_operations = []
        for compute_group_transfer_and_compute_operations in transfer_and_compute_operations:
            new_transfer_and_compute_operations.append([])
            for compute_operation_transfer_and_compute_operations in compute_group_transfer_and_compute_operations:
                new_transfer_and_compute_operations[-1].extend(compute_operation_transfer_and_compute_operations)
        
        return [ComputeOperationGroup(compute_group, compute_group_transfer_and_compute_operations, compute_group_loop_index_variables, compute_group_immediates) for compute_group, compute_group_transfer_and_compute_operations, compute_group_loop_index_variables, compute_group_immediates in zip(compute_groups, new_transfer_and_compute_operations, loop_index_variables, immediates)]
    
    def get_outer_loop_index_variables(self) -> list[StealthIndex]:
        if len(self._loop_indices) % 2 != 0:
            raise RuntimeError("Expected loop indices to be a multiple of 2")
        return self._loop_indices.copy()[:len(self._loop_indices) // 2]
    
    def get_inner_loop_index_variables(self) -> list[StealthIndex]:
        if len(self._loop_indices) % 2 != 0:
            raise RuntimeError("Expected loop indices to be a multiple of 2")
        return self._loop_indices.copy()[len(self._loop_indices) // 2:]
    
    def get_compute_output_operand_that_stores_to_output_operand(self) -> StealthOperand:
        store_to_output_operand: Optional[StealthStore] = None
        def helper1(statement: StealthStatement):
            if isinstance(statement, StealthStore) and statement.destination_operand_name in map(lambda o: o.name, self._outputs):
                nonlocal store_to_output_operand
                store_to_output_operand = statement
            elif isinstance(statement, StealthLoop):
                for statement in statement.body:
                    helper1(statement)

        compute_output_operand_that_stores_to_output_operand: Optional[StealthOperand] = None  
        def helper2(statement: StealthStatement):
            if isinstance(statement, StealthStore) and statement.destination_operand_name == store_to_output_operand.source_operand_name:
                nonlocal compute_output_operand_that_stores_to_output_operand
                if compute_output_operand_that_stores_to_output_operand is not None:
                    raise RuntimeError("Found multiple compute output operands that store to the output operand")
                else:
                    compute_output_operand_that_stores_to_output_operand = self._operands[statement.source_operand_name]
            elif isinstance(statement, StealthLoop):
                for statement in statement.body:
                    helper2(statement)
        
        for statement in self._statements:
            helper1(statement)
        if store_to_output_operand is None:
            raise RuntimeError("Could not find store to output operand")
        for statement in self._statements:
            helper2(statement)
        if compute_output_operand_that_stores_to_output_operand is None:
            raise RuntimeError("Could not find compute output operand that stores to the output operand")

        return compute_output_operand_that_stores_to_output_operand
    
    # ==================== Collector State Modification Functions ==================== # 
    def add_input(self, operand_name: str, operand_dimensions: list[Union[int, str]], operand_dtype: str, operand_location: str) -> None: 
        for dimension in operand_dimensions:
            if isinstance(dimension, str):
                self._used_var_names.add(dimension)
        operand = self.add_operand(operand_name, operand_dimensions, operand_dtype, operand_location) 
        self._inputs.append(operand)
        return operand

    def add_output(self, operand_name: str) -> None:
        self._outputs.append(self._operands[operand_name])
    
    def add_param(self, param_name: str, param_dimensions: list[Union[int, str]]) -> StealthParameter:
        new_param_dimensions = []
        for dimension in param_dimensions:
            if isinstance(dimension, int):
                new_param_dimensions.append(StealthLiteral(dimension))
            elif isinstance(dimension, str):
                new_param_dimensions.append(StealthVariableName(dimension))
            elif isinstance(dimension, StealthExpression):
                new_param_dimensions.append(dimension)
            else:
                raise TypeError(f"Unknown dimension type: {type(dimension)}") 
        
        if not isinstance(param_name, str):
            raise TypeError(f"Expected parameter name to be a string but instead got {type(param_name)}")
        if not isinstance(param_dimensions, list):
            raise TypeError(f"Expected parameter dimensions to be a list but instead got {type(param_dimensions)}")
        if any(not isinstance(dimension, StealthExpression) for dimension in new_param_dimensions):
            raise TypeError(f"Expected parameter dimensions to be a list of expressions but instead got {param_dimensions}")
        
        if param_name in self._used_var_names:
            raise RuntimeError(f"Parameter name {param_name} was not checked for duplicates before being added") 

        param = StealthParameter(param_name, new_param_dimensions)
        self._params.append(param)
        self._used_var_names.add(param.name)
        return param
    
    def add_operand(self, operand_name: str, operand_dimensions: list[Union[int, str]], operand_dtype: str, operand_location: str) -> StealthOperand:
        new_operand_dimensions = []
        for dimension in operand_dimensions:
            if isinstance(dimension, int):
                new_operand_dimensions.append(StealthLiteral(dimension))
            elif isinstance(dimension, str):
                new_operand_dimensions.append(StealthVariableName(dimension))
            elif isinstance(dimension, StealthExpression):
                new_operand_dimensions.append(dimension)
            else:
                raise TypeError(f"Unknown dimension type: {type(dimension)}")

        if not isinstance(operand_name, str):
            raise TypeError(f"Expected operand name to be a string but instead got {type(operand_name)}")
        if not isinstance(operand_dimensions, list):
            raise TypeError(f"Expected operand dimensions to be a list but instead got {type(operand_dimensions)}")
        if any(not isinstance(dimension, StealthExpression) for dimension in new_operand_dimensions):
            raise TypeError(f"Expected operand dimensions to be a list of expressions but instead got {operand_dimensions}")
        if not isinstance(operand_dtype, str):
            raise TypeError(f"Expected operand dtype to be a string but instead got {type(operand_dtype)}")
        if not isinstance(operand_location, str):
            raise TypeError(f"Expected operand location to be a string but instead got {type(operand_location)}")
        
        if operand_name in self._used_var_names:
            raise RuntimeError(f"Operand name {operand_name} was not checked for duplicates before being added")

        operand = StealthOperand(operand_name, new_operand_dimensions, operand_dtype, operand_location)
        self._used_var_names.add(operand.name)
        self._operands[operand_name] = operand
        return operand
    
    def add_loop_index(self, loop_index_name: str, end: StealthExpression, stride: StealthExpression) -> StealthIndex:
        if not isinstance(loop_index_name, str):
            raise TypeError(f"Expected loop index name to be a string but instead got {type(loop_index_name)}")
        if not isinstance(end, StealthExpression):
            raise TypeError(f"Expected end to be an expression but instead got {type(end)}")
        if not isinstance(stride, StealthExpression):
            raise TypeError(f"Expected stride to be an expression but instead got {type(stride)}")
        
        if loop_index_name in self._used_var_names:
            raise RuntimeError(f"Loop index name {loop_index_name} was not checked for duplicates before being added")

        loop_index = StealthIndex(loop_index_name, end, stride)
        self._loop_indices.append(loop_index)
        self._used_var_names.add(loop_index.name)
        return loop_index
    
    def add_statement(self, statement: StealthStatement) -> None:
        if len(self._current_loop_stack) != 0: 
            self._current_loop_stack[-1].body.append(statement)
        else:
            self._statements.append(statement)
        
        if isinstance(statement, StealthLoop):
            self._current_loop_stack.append(statement)
    
    def end_current_loop(self) -> None:
        self._current_loop_stack.pop()

    # ==================== Collector Visitor Functions ==================== # 
    def function(self, tree: ParseTree) -> None:
        for child in tree.children:
            if isinstance(child, Token) and child.type == "NAME":
                self._operation_name = str(child.value)
            else:
                self.visit(child)
    
    def parameters(self, tree: ParseTree) -> None:
        for child in tree.children:
            if isinstance(child, Token):
                self.raise_codelet_parse_error(f"Expected parameter but instead got {child.type}", child)
            assert isinstance(child, Tree)
            if child.data != "parameter":
                self.raise_codelet_parse_error(f"Expected parameter but instead got {child.data}", child)

            operand_name_child = child.children[0]
            operand_type_child = child.children[1]
            operand_location_child = child.children[2]

            operand_name: str = self.get_operand_name_from_operand_name_child_tree(operand_name_child) 
            operand_dimensions: list[Union[int, str]] = self.get_operand_dimensions_from_operand_type_child_tree(operand_type_child)
            operand_dtype: str = self.get_operand_dtype_from_operand_type_child_tree(operand_type_child) 

            if operand_dtype == "param":
                self.add_param(operand_name, operand_dimensions)
            else:
                operand_location: str = self.get_operand_location_from_operand_location_child_tree(operand_location_child)
                if operand_location != "DRAM":
                    self.raise_codelet_parse_error(f"Expected input operand {operand_name} location to be DRAM but instead got {operand_location}", child)
                self.add_input(operand_name, operand_dimensions, operand_dtype, operand_location)
    
    def for_loop(self, tree: ParseTree) -> None:
        assert isinstance(tree, Tree)
        loop_index_variable_name_child = tree.children[0]
        end_child = tree.children[1]
        stride_child = tree.children[2]
        body_child = tree.children[3]

        loop_index_variable_name: str = self.get_name(loop_index_variable_name_child, "loop index variable name")
        end: StealthExpression = self.get_stealth_expression_from_tree(end_child)
        stride: StealthExpression = self.get_stealth_expression_from_tree(stride_child) if stride_child else StealthLiteral(1)

        self.check_for_variable_name_is_not_defined(loop_index_variable_name, loop_index_variable_name_child)
        self.add_loop_index(loop_index_variable_name, end, stride)
        self.add_statement(StealthLoop(loop_index_variable_name, end, stride, []))
        self.visit(body_child)
        self.end_current_loop()
    
    def call(self, tree: ParseTree) -> None:
        assert isinstance(tree, Tree)
        operation_destination_operand_name_child = tree.children[0]
        operation_name_child = tree.children[1]
        operation_args_child = tree.children[2]

        operation_name = self.get_name(operation_name_child, "operation name")

        if operation_name == "store":
            if operation_destination_operand_name_child is not None:
                self.raise_codelet_parse_error(f"Did not expect destination operand for operation {operation_name}", tree)
            destination_operand_name = None
        else:
            if operation_destination_operand_name_child is None:
                self.raise_codelet_parse_error(f"Expected destination operand for operation {operation_name}", tree)
            destination_operand_name = self.get_name(operation_destination_operand_name_child, "destination operand name")
        
        if isinstance(operation_args_child, Token):
            self.raise_codelet_parse_error(f"Expected operation arguments but instead got {operation_args_child.type}", operation_args_child)
        assert isinstance(operation_args_child, Tree)
        
        if operation_name == "alloc":
            assert destination_operand_name is not None
            self.check_for_variable_name_is_not_defined(destination_operand_name, operation_destination_operand_name_child)
            alloc_size, alloc_location, alloc_dtype = self.get_alloc_args_from_operation_args_child(operation_args_child)   
            self.add_operand(destination_operand_name, alloc_size, alloc_dtype, alloc_location)
            alloc = StealthAllocation(destination_operand_name, alloc_size, alloc_location, alloc_dtype)
            self.add_statement(alloc)
        elif operation_name == "load":
            assert destination_operand_name is not None
            self.check_for_variable_name_is_not_defined(destination_operand_name, operation_destination_operand_name_child)
            load_source_operand_name, load_source_operand_offset, load_size, load_location = self.get_load_args_from_operation_args_child(operation_args_child)
            self.add_operand(destination_operand_name, load_size, self._operands[load_source_operand_name].dtype, load_location)
            load = StealthLoad(destination_operand_name, load_source_operand_name, load_source_operand_offset, load_size, load_location)
            self.add_statement(load)
        elif operation_name == "store":
            assert destination_operand_name is None
            store_destination_operand_name, store_destination_operand_offset, store_source_operand_name = self.get_store_args_from_operation_args_child(operation_args_child)
            store = StealthStore(store_destination_operand_name, store_destination_operand_offset, store_source_operand_name)
            self.add_statement(store)
        else:
            assert destination_operand_name is not None
            self.check_for_variable_name_is_not_defined(destination_operand_name, operation_destination_operand_name_child)
            compute_arguments, location = self.get_compute_args_from_operation_args_child(operation_args_child)
            self.check_compute_arguments(operation_name, compute_arguments, location)
            output_operand_dimensions = self.get_output_operand_size(operation_name, compute_arguments, location)
            self.add_operand(destination_operand_name, output_operand_dimensions, self._COMPUTE_OUTPUT_DTYPE[location], location)
            compute = StealthCompute(destination_operand_name, operation_name, compute_arguments, location)
            self.add_statement(compute)
    
    def return_output(self, tree: ParseTree) -> None:
        assert isinstance(tree, Tree)
        for child in tree.children:
            output_operand_name = self.get_name(child, "output operand name")
            self.check_for_variable_name_defined(output_operand_name, child)
            self.add_output(output_operand_name)
    
    # ==================== Collector Helper Functions ==================== #
    def raise_codelet_parse_error(self, message: str, obj):
        _raise_codelet_parse_error(message, obj, self._codelet_string) 
    
    def check_for_variable_name_defined(self, variable_name: str, tree: ParseTree) -> None:
        if variable_name not in self._used_var_names:
            self.raise_codelet_parse_error(f"A variable with name \"{variable_name}\" does not exist.", tree)

    def check_for_variable_name_is_not_defined(self, variable_name: str, tree: ParseTree) -> None:
        if variable_name in self._used_var_names:
            self.raise_codelet_parse_error(f"A variable with name \"{variable_name}\" already exists.", tree)
    
    def check_compute_arguments(self, operation_name: str, compute_arguments: list[Union[str, StealthExpression]], location: str) -> None:
        if operation_name == "mvmul":
            if location != "PE_ARRAY":
                raise CodeletError("Matrix-vector multiplication operation \"mvmul\" is only supported on the PE_ARRAY")
            if len(compute_arguments) != 3:
                raise CodeletError(f"Expected 4 arguments for operation \"mvmul\" (input, weight, intermediate output) but instead got {len(compute_arguments)}")
            if any(isinstance(arg, StealthExpression) for arg in compute_arguments):
                raise CodeletError("Matrix-vector multiplication operation \"mvmul\" does not support constants as arguments")
            
            input_operand = self._operands[compute_arguments[0]]
            weight_operand = self._operands[compute_arguments[1]]
            intermediate_output_operand = self._operands[compute_arguments[2]]

            if len(input_operand.shape) != 2:
                raise CodeletError(f"Expected first argument of matrix-vector multiplication operation \"mvmul\" to be a 2D tensor but instead got a {len(input_operand.shape)}D tensor")
            if len(weight_operand.shape) != 2:
                raise CodeletError(f"Expected second argument of matrix-vector multiplication operation \"mvmul\" to be a 2D tensor but instead got a {len(weight_operand.shape)}D tensor")
            if len(intermediate_output_operand.shape) != 2:
                raise CodeletError(f"Expected third argument of matrix-vector multiplication operation \"mvmul\" to be a 2D tensor but instead got a {len(intermediate_output_operand.shape)}D tensor")

            # TODO: Check operand dimension values to ensure valid matrix-vector multiplication
            
        elif operation_name == "mvmul_bias":
            if location != "PE_ARRAY":
                raise CodeletError("Matrix-vector multiplication operation \"mvmul_bias\" is only supported on the PE_ARRAY")
            if len(compute_arguments) != 4:
                raise CodeletError(f"Expected 5 arguments for operation \"mvmul_bias\" (input, weight, bias, intermediate output) but instead got {len(compute_arguments)}")
            if any(isinstance(arg, StealthExpression) for arg in compute_arguments):
                raise CodeletError("Matrix-vector multiplication operation \"mvmul_bias\" does not support constants as arguments")
            
            input_operand = self._operands[compute_arguments[0]]
            weight_operand = self._operands[compute_arguments[1]]
            bias_operand = self._operands[compute_arguments[2]]
            intermediate_output_operand = self._operands[compute_arguments[3]]

            if len(input_operand.shape) != 2:
                raise CodeletError(f"Expected first argument of matrix-vector multiplication operation \"mvmul\" to be a 2D tensor but instead got a {len(input_operand.shape)}D tensor")
            if len(weight_operand.shape) != 2:
                raise CodeletError(f"Expected second argument of matrix-vector multiplication operation \"mvmul\" to be a 2D tensor but instead got a {len(weight_operand.shape)}D tensor")
            if len(bias_operand.shape) != 2:
                raise CodeletError(f"Expected third argument of matrix-vector multiplication operation \"mvmul\" to be a 2D tensor but instead got a {len(bias_operand.shape)}D tensor")
            if len(intermediate_output_operand.shape) != 2:
                raise CodeletError(f"Expected fourth argument of matrix-vector multiplication operation \"mvmul\" to be a 2D tensor but instead got a {len(intermediate_output_operand.shape)}D tensor")

            # TODO: Check operand dimension values to ensure valid matrix-vector multiplication
        elif operation_name in ["add", "sub", "mul", "div", "max", "min", "rshift", "lshift"]:
            if location != "SIMD":
                raise CodeletError(f"Operation \"{operation_name}\" is only supported on the SIMD")
            if len(compute_arguments) != 2:
                raise CodeletError(f"Expected 2 arguments for operation \"{operation_name}\" but instead got {len(compute_arguments)}")
            
            operand1 = self._operands[compute_arguments[0]]
            operand2 = self._operands[compute_arguments[1]]

            if len(operand1.shape) != 1:
                raise CodeletError(f"Expected first argument of operation \"{operation_name}\" to be a 1D tensor but instead got a {len(operand1.shape)}D tensor")
            if len(operand2.shape) != 1:
                raise CodeletError(f"Expected second argument of operation \"{operation_name}\" to be a 1D tensor but instead got a {len(operand2.shape)}D tensor")
            if operand1.shape != operand2.shape:
                raise CodeletError(f"Expected arguments of operation \"{operation_name}\" to have the same shape but instead got {operand1.shape} and {operand2.shape}")
        elif operation_name == "macc":
            raise NotImplementedError
        elif operation_name in ["relu", "leaky_relu", "sigmoid", "tanh", "exp", "log2", "sqrt", "inv_sqrt"]:
            if location != "SIMD":
                raise CodeletError(f"Operation \"{operation_name}\" is only supported on the SIMD")
            if len(compute_arguments) != 1:
                raise CodeletError(f"Expected 1 argument for operation \"{operation_name}\" but instead got {len(compute_arguments)}")
            
            operand = self._operands[compute_arguments[0]]

            if len(operand.shape) != 1:
                raise CodeletError(f"Expected argument of operation \"{operation_name}\" to be a 1D tensor but instead got a {len(operand.shape)}D tensor")
        else:
            raise NotImplementedError(f"{operation_name} is not implemented yet.")
    
    def get_name(self, tree, name_type: Optional[str] = None) -> str:
        name_type_str: str = name_type if name_type else "name"
        if isinstance(tree, Token):
            if tree.type != "NAME":
                self.raise_codelet_parse_error(f"Expected a {name_type_str} but instead got {tree.type}", tree)
            return str(tree.value)
        elif isinstance(tree, Tree):
            if (len(tree.children) != 1) or isinstance(tree.children[0], Tree) or (tree.children[0].type != "NAME"):
                self.raise_codelet_parse_error(f"Expected a {name_type_str} but instead got {tree.data}", tree)
            assert isinstance(tree.children[0], Token)
            return str(tree.children[0].value)
        else:
            raise TypeError(f"Unknown type: {type(tree)}")

    def get_operand_name_from_operand_name_child_tree(self, operand_name_child) -> str:
        return self.get_name(operand_name_child)

    def get_operand_dimensions_from_operand_type_child_tree(self, operand_type_child) -> list[Union[int, str]]:
        if isinstance(operand_type_child, Token):
            self.raise_codelet_parse_error(f"Expected operand type but instead got {operand_type_child.type}", operand_type_child)
        assert isinstance(operand_type_child, Tree)
        if operand_type_child.data != "type":
            self.raise_codelet_parse_error(f"Expected operand type but instead got {operand_type_child.data}", operand_type_child)
        
        operand_dimensions: list[Union[int, str]] = []
        if len(operand_type_child.children) == 2:
            operand_type_child_dimensions_child = operand_type_child.children[1]
            for dimension_child in operand_type_child_dimensions_child.children:
                dimension = dimension_child.children[0]
                if dimension.type == "NAME":
                    operand_dimensions.append(str(dimension.value))
                elif dimension.type == "INT":
                    operand_dimensions.append(int(dimension.value))
                else:
                    raise ValueError(f"Unknown dimension type: {dimension_child.type}")
        
        return operand_dimensions

    def get_operand_dtype_from_operand_type_child_tree(self, operand_type_child) -> str:
        if isinstance(operand_type_child, Token):
            self.raise_codelet_parse_error(f"Expected operand type but instead got {operand_type_child.type}", operand_type_child)
        assert isinstance(operand_type_child, Tree)
        if operand_type_child.data != "type":
            self.raise_codelet_parse_error(f"Expected operand type but instead got {operand_type_child.data}", operand_type_child)
        
        return self.get_name(operand_type_child.children[0], name_type="dtype")

    def get_operand_location_from_operand_location_child_tree(self, operand_location_child) -> str:
        return self.get_name(operand_location_child, name_type="location")
    
    def get_alloc_args_from_operation_args_child(self, operation_args_child) -> tuple[list[StealthExpression], str, str]:
        if len(operation_args_child.children) != 3:
            self.raise_codelet_parse_error(f"Expected 3 arguments for alloc operation but instead got {len(operation_args_child.children)}", operation_args_child)
        assert all(isinstance(child, Tree) for child in operation_args_child.children)
        if any(child.data != "call_arg" and len(child.children) == 1 for child in operation_args_child.children):
            self.raise_codelet_parse_error(f"Expected all arguments to be call arguments but instead got {operation_args_child.children}", operation_args_child)

        size_child = operation_args_child.children[0].children[0]
        location_child = operation_args_child.children[1].children[0]
        dtype_child = operation_args_child.children[2].children[0]

        if isinstance(size_child, Token):
            self.raise_codelet_parse_error(f"Expected size argument but instead got {size_child.type}", size_child)
        assert isinstance(size_child, Tree)
        if size_child.data != "size":
            self.raise_codelet_parse_error(f"Expected first argument to be size but instead got {size_child.data}", size_child)
        if isinstance(location_child, Tree):
            self.raise_codelet_parse_error(f"Expected location argument but instead got {location_child.data}", location_child)
        if isinstance(dtype_child, Tree):
            self.raise_codelet_parse_error(f"Expected dtype argument but instead got {dtype_child.data}", dtype_child)

        size: list[StealthExpression] = [self.get_stealth_expression_from_tree(child) for child in size_child.children]
        location: str = self.get_name(location_child, name_type="location")
        dtype: str = self.get_name(dtype_child, name_type="dtype")

        return size, location, dtype
    
    def get_load_args_from_operation_args_child(self, operation_args_child) -> tuple[str, list[StealthExpression], list[StealthExpression], str]:
        if len(operation_args_child.children) != 3:
            self.raise_codelet_parse_error(f"Expected 3 arguments for load operation but instead got {len(operation_args_child.children)}", operation_args_child)
        assert all(isinstance(child, Tree) for child in operation_args_child.children)
        if any(child.data != "call_arg" or len(child.children) != 1 for child in operation_args_child.children):
            self.raise_codelet_parse_error(f"Expected all arguments to be call arguments but instead got {operation_args_child.children}", operation_args_child)
        if len(operation_args_child.children[0].children) != 1 or operation_args_child.children[0].children[0].data != "indexed_variable":
            self.raise_codelet_parse_error(f"Expected an indexed operand for source operand argument but instead got {len(operation_args_child.children[0].children)}", operation_args_child.children[0])
        if len(operation_args_child.children[0].children[0].children) < 1:
            self.raise_codelet_parse_error(f"Expected an indexed operand for source operand argument but instead got {len(operation_args_child.children[0].children[0].children)}", operation_args_child.children[0].children[0])
        
        source_operand_child = operation_args_child.children[0].children[0].children[0]
        source_operand_offset_children = operation_args_child.children[0].children[0].children[1:]
        size_child = operation_args_child.children[1].children[0]
        location_child = operation_args_child.children[2].children[0]

        if isinstance(source_operand_child, Tree):
            self.raise_codelet_parse_error(f"Expected source operand argument but instead got {source_operand_child.data}", source_operand_child)
        assert isinstance(source_operand_child, Token)
        for source_operand_offset_child in source_operand_offset_children:
            if isinstance(source_operand_offset_child, Token):
                self.raise_codelet_parse_error(f"Expected source operand offset argument but instead got {source_operand_offset_child.type}", source_operand_offset_child)
            assert isinstance(source_operand_offset_child, Tree)
        if isinstance(size_child, Token):
            self.raise_codelet_parse_error(f"Expected size argument but instead got {size_child.type}", size_child)
        assert isinstance(size_child, Tree)
        if size_child.data != "size":
            self.raise_codelet_parse_error(f"Expected size argument but instead got {size_child.data}", size_child)
        if isinstance(location_child, Tree):
            self.raise_codelet_parse_error(f"Expected location argument but instead got {location_child.data}", location_child)
        
        source_operand_name: str = self.get_name(source_operand_child, name_type="source operand")
        self.check_for_variable_name_defined(source_operand_name, source_operand_child)
        source_operand_offset: list[StealthExpression] = [self.get_stealth_expression_from_tree(child) for child in source_operand_offset_children]
        size: list[StealthExpression] = [self.get_stealth_expression_from_tree(child) for child in size_child.children]
        location: str = self.get_name(location_child, name_type="location")

        return source_operand_name, source_operand_offset, size, location

    def get_store_args_from_operation_args_child(self, operation_args_child):
        if len(operation_args_child.children) != 2:
            self.raise_codelet_parse_error(f"Expected 2 arguments for store operation but instead got {len(operation_args_child.children)}", operation_args_child)
        assert all(isinstance(child, Tree) for child in operation_args_child.children)
        if any(child.data != "call_arg" or len(child.children) != 1 for child in operation_args_child.children):
            self.raise_codelet_parse_error(f"Expected all arguments to be call arguments but instead got {operation_args_child.children}", operation_args_child)
        if len(operation_args_child.children[0].children) != 1 or operation_args_child.children[0].children[0].data != "indexed_variable":
            self.raise_codelet_parse_error(f"Expected an indexed operand for destination operand argument but instead got {len(operation_args_child.children[0].children)}", operation_args_child.children[0])
        if len(operation_args_child.children[0].children[0].children) < 1:
            self.raise_codelet_parse_error(f"Expected an indexed operand for destination operand argument but instead got {len(operation_args_child.children[0].children[0].children)}", operation_args_child.children[0].children[0])
        
        destination_operand_child = operation_args_child.children[0].children[0].children[0]
        destination_operand_offset_children = operation_args_child.children[0].children[0].children[1:]
        source_operand_child = operation_args_child.children[1].children[0]

        if isinstance(destination_operand_child, Tree):
            self.raise_codelet_parse_error(f"Expected destination operand argument but instead got {destination_operand_child.data}", destination_operand_child)
        assert isinstance(destination_operand_child, Token)
        for destination_operand_offset_child in destination_operand_offset_children:
            if isinstance(destination_operand_offset_child, Token):
                self.raise_codelet_parse_error(f"Expected destination operand offset argument but instead got {destination_operand_offset_child.type}", destination_operand_offset_child)
            assert isinstance(destination_operand_offset_child, Tree)
        if isinstance(source_operand_child, Tree):
            self.raise_codelet_parse_error(f"Expected source operand argument but instead got {source_operand_child.data}", source_operand_child)

        destination_operand_name: str = self.get_name(destination_operand_child, name_type="destination operand")
        self.check_for_variable_name_defined(destination_operand_name, destination_operand_child)
        destination_operand_offset: list[StealthExpression] = [self.get_stealth_expression_from_tree(child) for child in destination_operand_offset_children]
        source_operand_name: str = self.get_name(source_operand_child, name_type="source operand")
        self.check_for_variable_name_defined(source_operand_name, source_operand_child)

        return destination_operand_name, destination_operand_offset, source_operand_name

    def get_compute_args_from_operation_args_child(self, operation_args_child) -> tuple[list[StealthExpression], str]:
        assert all(isinstance(child, Tree) for child in operation_args_child.children)

        arguments: list[StealthExpression] = []
        for child in operation_args_child.children[:-1]:
            if len(child.children) != 1:
                self.raise_codelet_parse_error(f"Expected all arguments to be call arguments but instead got {operation_args_child.children}", operation_args_child) 
            expression_child = child.children[0]
            expression = self.get_stealth_expression_from_tree(expression_child)
            if self.is_expression_constant(expression):
                value = evaluate_expression(expression)
                immediate_name = UniqueNameGenerator.get_unique_name("immediate")
                self._immediates[immediate_name] = value
                arguments.append(immediate_name)
            elif isinstance(expression, StealthVariableName) and expression.name in self._operands:
                arguments.append(expression.name)
            else:
                self.raise_codelet_parse_error(f"Expected all arguments to be either a constant or an operand but instead got {expression}", expression_child)
        
        location_child = operation_args_child.children[-1]
        location = self.get_name(location_child, name_type="location")
        
        return arguments, location

    def get_stealth_expression_from_tree(self, expression_tree) -> StealthExpression:
        if isinstance(expression_tree, Tree):
            if len(expression_tree.children) > 2:
                raise RuntimeError(f"Something went wrong... Expressions cannot have more than two arguments")
            elif len(expression_tree.children) == 1:
                return self.get_stealth_expression_from_tree(expression_tree.children[0])
            elif len(expression_tree.children) == 2:
                lhs = self.get_stealth_expression_from_tree(expression_tree.children[0])
                rhs = self.get_stealth_expression_from_tree(expression_tree.children[1])
                operation = self._OPERATION_TYPE_TO_OPERATION_STR[expression_tree.data] 
                return StealthBinaryExpression(lhs, rhs, operation)
            else:
                raise RuntimeError()
        else:
            if expression_tree.type == "NAME":
                name = str(expression_tree.value)
                if name not in self._used_var_names:
                    self.raise_codelet_parse_error(f"Variable {name} is not defined", expression_tree)
                return StealthVariableName(name)
            elif expression_tree.type == "INT":
                return StealthLiteral(int(expression_tree.value))
            else:
                raise RuntimeError(f"Unknown expression type: {expression_tree.type}")
    
    def get_output_operand_size(self, compute_operation_name: str, compute_arguments: list[Union[str, StealthExpression]], compute_unit: str) -> list[StealthExpression]:
        if compute_operation_name in ["mvmul", "mvmul_bias"]:
            return self._operands[compute_arguments[-1]].shape
        elif compute_operation_name in ["add", "sub", "mul", "div", "max", "min",
                                        "rshift", "lshift",
                                        "relu", "leaky_relu", "sigmoid", "tanh",
                                        "exp", "sqrt", "inv_sqrt", "log2"]:
            return self._operands[compute_arguments[0]].shape
        else:
            raise NotImplementedError(f"{compute_operation_name} is not implemented yet")
    
    def is_expression_constant(self, expression) -> bool:
        if isinstance(expression, StealthLiteral):
            return True
        elif isinstance(expression, StealthVariableName):
            return expression.name in map(lambda p: p.name, self._params)
        elif isinstance(expression, StealthUnaryExpression):
            return self.is_expression_constant(expression.operand)
        elif isinstance(expression, StealthBinaryExpression):
            return self.is_expression_constant(expression.lhs) and self.is_expression_constant(expression.rhs)
        else:
            raise RuntimeError(f"Unknown expression type: {type(expression)}")
    
    def get_loop_index_variable_names_from_expression(self, expression: StealthExpression) -> set[str]:
        all_loop_index_variable_names: set[str] = set(map(lambda l: l.name, self._loop_indices))
        ret = set()
        def helper(expression: StealthExpression) -> None:
            if isinstance(expression, StealthVariableName):
                if expression.name in all_loop_index_variable_names:
                    nonlocal ret
                    ret.add(expression.name)
            elif isinstance(expression, StealthUnaryExpression):
                helper(expression.operand)
            elif isinstance(expression, StealthBinaryExpression):
                helper(expression.lhs)
                helper(expression.rhs)
        
        helper(expression)
        return ret
    
    # ==================== Collector Checking Functions ==================== #
    def check_codelet(self):
        self.check_codelet_number_of_inner_and_outer_loops_is_same()
        self.check_codelet_loop_format_valid()
        self.check_codelet_outer_inner_loop_order_same()
        self.check_codelet_only_has_one_store_to_output()

    def check_codelet_number_of_inner_and_outer_loops_is_same(self):
        setattr(self, "_number_of_outer_loops", 0)
        setattr(self, "_number_of_inner_loops", 0)
        for statement in self._statements:
            self._check_codelet_number_of_inner_and_outer_loops_is_same(statement)
        if self._number_of_outer_loops != self._number_of_inner_loops:
            raise CodeletError(f"Codelet {self._operation_name} has a different number of inner and outer loops")
        delattr(self, "_number_of_outer_loops")
        delattr(self, "_number_of_inner_loops")

    def _check_codelet_number_of_inner_and_outer_loops_is_same(self, statement: StealthStatement):
        if isinstance(statement, StealthLoop):
            # Outer loops must have a constant first argument which is the number of tiles set by the LLM
            if isinstance(evaluate_expression(statement.end), StealthLiteral):
                self._number_of_outer_loops += 1
            else:
                self._number_of_inner_loops += 1
            
            for loop_statement in statement.body:
                self._check_codelet_number_of_inner_and_outer_loops_is_same(loop_statement)
    
    def check_codelet_loop_format_valid(self):
        # Can use inner and outer loop index getter functions now that we know that the number of inner and outer loops is the same
        outer_loop_index_variables = self.get_outer_loop_index_variables()
        inner_loop_index_variables = self.get_inner_loop_index_variables()

        for outer_loop_index_variable in outer_loop_index_variables:
            end = evaluate_expression(outer_loop_index_variable.end)
            stride = evaluate_expression(outer_loop_index_variable.stride)
            if not isinstance(end, StealthLiteral):
                raise CodeletError(f"Codelet {self._operation_name} has an outer loop with an end that is not a valid representation of the number of tiles")
            if (not isinstance(stride, StealthBinaryExpression) or not stride.operation == "//") and not isinstance(stride, StealthVariableName): 
                raise CodeletError(f"Codelet {self._operation_name} has an outer loop with a stride that is not a valid representation of the tile size")
        for inner_loop_index_variable in inner_loop_index_variables:
            end = evaluate_expression(inner_loop_index_variable.end)
            stride = evaluate_expression(inner_loop_index_variable.stride)
            if (not isinstance(end, StealthBinaryExpression) or not end.operation == "//") and not isinstance(end, StealthVariableName):
                raise CodeletError(f"Codelet {self._operation_name} has an inner loop with an end that is not a valid representation of the tile size")
    
    def check_codelet_outer_inner_loop_order_same(self):
        def raise_exception_for_not_matching_tile_size(_outer, _inner):
            raise CodeletError(f"Codelet {self._operation_name} has an outer loop ({_outer}) with a stride (tile size) that is not the same as the inner loop ({_inner}) end (tile size)")

        outer_loop_index_variables = self.get_outer_loop_index_variables()
        inner_loop_index_variables = self.get_inner_loop_index_variables() 
        assert len(outer_loop_index_variables) == len(inner_loop_index_variables)
        for outer_loop_index_variable, inner_loop_index_variable in zip(outer_loop_index_variables, inner_loop_index_variables):
            outer_loop_stride = evaluate_expression(outer_loop_index_variable.stride)
            inner_loop_end = evaluate_expression(inner_loop_index_variable.end)
            if isinstance(outer_loop_stride, StealthBinaryExpression) and outer_loop_stride.operation == "//":
                if not isinstance(inner_loop_end, StealthBinaryExpression) or not inner_loop_end.operation == "//":
                    raise_exception_for_not_matching_tile_size(outer_loop_index_variable, inner_loop_index_variable)
                if outer_loop_stride.lhs != inner_loop_end.lhs:
                    raise_exception_for_not_matching_tile_size(outer_loop_index_variable, inner_loop_index_variable)
                if outer_loop_stride.rhs != inner_loop_end.rhs:
                    raise_exception_for_not_matching_tile_size(outer_loop_index_variable, inner_loop_index_variable)
            elif isinstance(outer_loop_stride, StealthVariableName):
                if not isinstance(inner_loop_end, StealthVariableName):
                    raise_exception_for_not_matching_tile_size(outer_loop_index_variable, inner_loop_index_variable)
                if outer_loop_stride.name != inner_loop_end.name:
                    raise_exception_for_not_matching_tile_size(outer_loop_index_variable, inner_loop_index_variable)
            else:
                raise RuntimeError(f"Outer loop {outer_loop_index_variable} has an invalid format")
            
    
    def check_codelet_only_has_one_store_to_output(self):
        setattr(self, "_number_of_stores_to_output", defaultdict(lambda: 0))
        for statement in self._statements:
            self._check_codelet_has_only_one_store_to_output(statement)
        if any(store_to_output == 0 for store_to_output in self._number_of_stores_to_output.values()):
            raise CodeletError(f"Codelet {self._operation_name} has no store to the output operand")
        delattr(self, "_number_of_stores_to_output")

    def _check_codelet_has_only_one_store_to_output(self, statement: StealthStatement):
        if isinstance(statement, StealthLoop):
            for loop_statement in statement.body:
                self._check_codelet_has_only_one_store_to_output(loop_statement)
        elif isinstance(statement, StealthStore):
            for o in self._outputs:
                if statement.destination_operand_name == o.name:
                    if self._number_of_stores_to_output[o.name] == 1:
                        raise CodeletError(f"Codelet {self._operation_name} has more than one store to an output operand (second store is {statement})")
                    else:
                        self._number_of_stores_to_output[o.name] += 1

    
def collect_codelet_information_from_parse_tree(tree: ParseTree, codelet_string: str) -> CodeletInformationCollector:
    collector = CodeletInformationCollector(codelet_string)
    collector.visit(tree)
    collector.check_codelet()
    # print(collector)
    return collector


# ==================== Codelet Template Creator ==================== #
def int_to_name(i: int) -> str:
    D_TO_NAME_MAP = {
        0: "ZERO",
        1: "ONE",
        2: "TWO",
        3: "THREE",
        4: "FOUR",
        5: "FIVE",
        6: "SIX",
        7: "SEVEN",
        8: "EIGHT",
        9: "NINE"
    }

    i_str = str(i)
    ret = []
    for c in i_str:
        ret.append(D_TO_NAME_MAP[int(c)])
    return "_".join(ret)


def input_output_dimension_to_name(dimension: StealthExpression) -> str:
    if isinstance(dimension, StealthVariableName):
        return dimension.name
    elif isinstance(dimension, StealthLiteral):
        return int_to_name(dimension.value)
    else:
        raise TypeError(f"Unknown stealth codelet operand dimension type: {type(dimension)}")


def expression_to_loop_index_expression(input_expression: StealthExpression, loop_indices: dict, dummy_ops: dict):
    def helper(expression: StealthExpression):
        if isinstance(expression, StealthLiteral):
            return expression.value
        elif isinstance(expression, StealthVariableName):
            if expression.name in loop_indices:
                return loop_indices[expression.name]
            elif expression.name in dummy_ops:
                return dummy_ops[expression.name]
            else:
                raise RuntimeError(f"Unknown variable name: {expression.name}")
        elif isinstance(expression, StealthUnaryExpression):
            if expression.operation == "-":
                return -helper(expression.operand)
            else:
                raise RuntimeError(f"Unknown unary operation: {expression.operation}")
        elif isinstance(expression, StealthBinaryExpression):
            lhs = helper(expression.lhs)
            rhs = helper(expression.rhs)
            if expression.operation == "+":
                return lhs + rhs
            elif expression.operation == "-":
                return lhs - rhs
            elif expression.operation == "*":
                return lhs * rhs
            elif expression.operation == "/":
                return lhs / rhs
            elif expression.operation == "//":
                return lhs / rhs
            else:
                raise RuntimeError(f"Unknown binary operation: {expression.operation}")
        else:
            raise RuntimeError(f"Unknown expression type: {type(expression)}")

    input_expression = evaluate_expression(input_expression)
    return helper(input_expression)


@dataclasses.dataclass
class CodeletTemplateTransfer:
    operand: Any
    source: str
    destination: str


@dataclasses.dataclass
class CodeletTemplateCompute:
    operation: str
    inputs: list[tuple]
    outputs: list[tuple]
    location: str


class CodeletTemplateBody:
    _COMPUTE_UNIT_LOCATION_MAP = {
        "SIMD": "SIMD",
        "PE_ARRAY": "pe_array"
    }
    loads: list[CodeletTemplateTransfer]
    compute: Optional[CodeletTemplateCompute]
    stores: list[CodeletTemplateTransfer]

    _staged_tile_loads: list[StealthLoad]
    _staged_element_loads: list[StealthLoad]
    _staged_tile_stores: list[StealthStore]
    _staged_element_stores: list[StealthStore]
    _staged_operation: str
    _staged_inputs: list[str]
    _staged_outputs: list[str]
    _staged_location: str

    def __init__(self) -> None:
        self.loads = []
        self.compute = None
        self.stores = []

        self._staged_tile_loads = []
        self._staged_element_loads = []
        self._staged_tile_stores = []
        self._staged_element_stores = []
        self._staged_operation = ""
        self._staged_inputs = []
        self._staged_outputs = []
        self._staged_location = ""

    def add_tile_load(self, load: StealthLoad) -> None:
        self._staged_tile_loads.append(load)
    
    def add_element_load(self, load: StealthLoad) -> None:
        self._staged_element_loads.append(load)
    
    def add_tile_store(self, store: StealthStore) -> None:
        self._staged_tile_stores.append(store)

    def add_element_store(self, store: StealthStore) -> None:
        self._staged_element_stores.append(store)
    
    def set_compute(self, operation: str, inputs: list[str], outputs: list[str], location: str) -> None:
        self._staged_operation = operation
        self._staged_inputs = inputs.copy()
        self._staged_outputs = outputs.copy()
        self._staged_location = location

    def finalize(self, collector: CodeletInformationCollector, operands: dict, loop_indices: dict, dummy_ops: dict) -> None:
        if len(self._staged_element_loads) != len(self._staged_inputs):
            raise RuntimeError(f"Expected {len(self._staged_inputs)} loads for inputs {self._staged_inputs} but instead got {len(self._staged_element_loads)}")
        if len(self._staged_element_stores) != len(self._staged_outputs):
            raise RuntimeError(f"Expected {len(self._staged_outputs)} stores for outputs {self._staged_outputs} but instead got {len(self._staged_element_stores)}")
        for load in self._staged_tile_loads:
            self.loads.append(CodeletTemplateTransfer(operands[load.source_operand_name], "DRAM", load.location))
        for store in self._staged_tile_stores:
            self.stores.append(CodeletTemplateTransfer(operands[store.destination_operand_name], collector._operands[store.source_operand_name].location, "DRAM"))

        input_operands = []
        input_offsets = []
        for input_operand_name in self._staged_inputs:
            input_operands.append(operands[input_operand_name])
            element_load = None
            for load in self._staged_element_loads:
                if load.destination_operand_name == input_operand_name:
                    element_load = load
                    break
            assert element_load is not None, f"Expected {input_operand_name} to be in {self._staged_element_loads}"
            input_offsets.append(tuple(map(lambda e: expression_to_loop_index_expression(e, loop_indices, dummy_ops), element_load.source_operand_offset)))
        output_operands = []
        output_offsets = []
        output_locations = []
        for output_operand_name in self._staged_outputs:
            output_operands.append(operands[output_operand_name])
            element_store = None
            for store in self._staged_element_stores:
                if store.source_operand_name == output_operand_name:
                    element_store = store
                    break
            assert element_store is not None, f"Expected {output_operand_name} to be in {self._staged_element_stores}"
            output_offsets.append(tuple(map(lambda e: expression_to_loop_index_expression(e, loop_indices, dummy_ops), element_store.destination_operand_offset)))
            output_locations.append(collector._operands[store.destination_operand_name].location)
        
        self.compute = CodeletTemplateCompute(self._staged_operation, list(zip(input_operands, input_offsets)), list(zip(output_operands, output_offsets, output_locations)), self._staged_location)
    
    def create_body(self, cdlt):
        for load in self.loads:
            cdlt.transfer(load.operand, [load.source, load.destination])
        assert self.compute is not None, "Compute operation group is not set"
        for output, _, output_location in self.compute.outputs:
            output.set_write_destination(output_location)
        cdlt.compute(self.compute.operation, list(map(lambda t: t[0][t[1]], self.compute.inputs)), list(map(lambda t: t[0][t[1]], self.compute.outputs)), self._COMPUTE_UNIT_LOCATION_MAP[self.compute.location])
        for store in self.stores:
            cdlt.transfer(store.operand, [store.source, store.destination])

def create_codelet_template_from_codelet_information_collector(collector: CodeletInformationCollector) -> Callable[[ArchitectureGraph], CodeletTemplate]:
    ON_CHIP_LOCATIONS = {"IBUF", "WBUF", "BBUF", "OBUF", "VMEM1", "VMEM2"}

    def codelet_template(hag: ArchitectureGraph) -> CodeletTemplate:
        # Configure the Data Types
        input_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
        acc_dtype = DTYPE_MAP[ f"FXP{hag.meta_cfg['ACC_WIDTH']}"] 

        with CodeletTemplate(collector.get_operation_name()) as cdlt:
            # ==================== Create DummyOps for Dimensions ==================== #
            dimension_dummy_ops = {}
            def dummy_op_creation_helper(collector_operands, node_operands, is_input=True):
                for i, stealth_codelet_operand in enumerate(collector_operands):
                    for j, stealth_codelet_operand_dimension in enumerate(stealth_codelet_operand.shape):
                        dimension_name = input_output_dimension_to_name(stealth_codelet_operand_dimension)
                        if dimension_name not in dimension_dummy_ops:
                            fn_body_str = f"node.inputs[{i}].shape[{j}]" if is_input else f"node.outputs[{i}].shape[{j}]"
                            new_flex_param = FlexParam(
                                name=dimension_name,
                                fn_args=["node"],
                                fn_body_str=fn_body_str,
                            )
                            new_flex_param.create_function_from_str(["node"], fn_body_str)
                            new_dummy_op = DummyOp(
                                ["NodePlaceholder"],
                                new_flex_param,
                                dtype=None
                            )
                            cdlt._dummy_ops[dimension_name] = new_dummy_op
                            dimension_dummy_ops[dimension_name] = new_dummy_op
            dummy_op_creation_helper(collector.get_input_operands(), cdlt.node.inputs, is_input=True)
            dummy_op_creation_helper(collector.get_output_operands(), cdlt.node.outputs, is_input=False)

            # ==================== Create Input/Output Operands ==================== #
            def operand_creation_helper(collector_operands) -> list:
                ret = []
                for collector_operand in collector_operands:
                    data_type = input_dtype if collector_operand.dtype == "i8" else acc_dtype
                    ret.append(cdlt.create_operand_template(collector_operand.name, OP_DTYPES, list(map(lambda d: dimension_dummy_ops[d.name], collector_operand.shape)), default_dtype=data_type))
                return ret
            input_operands = operand_creation_helper(collector.get_input_operands())
            output_operands = operand_creation_helper(collector.get_output_operands())
            cdlt.set_inputs(input_operands)
            cdlt.set_outputs(output_operands)
            name_to_operand = {operand.name: operand for operand in input_operands + output_operands}

            # ==================== Create Each Compute Operation Group ==================== #
            def constant_expression_to_dummy_op(_expression: StealthExpression) -> Union[DummyOp, int]:
                assert collector.is_expression_constant(_expression)
                if isinstance(_expression, StealthLiteral) and isinstance(_expression.value, int):
                    return _expression.value
                elif isinstance(_expression, StealthVariableName):
                    return dimension_dummy_ops[_expression.name]
                elif isinstance(_expression, StealthBinaryExpression):
                    if _expression.operation == "+":
                        return constant_expression_to_dummy_op(_expression.lhs) + constant_expression_to_dummy_op(_expression.rhs)
                    elif _expression.operation == "-":
                        return constant_expression_to_dummy_op(_expression.lhs) - constant_expression_to_dummy_op(_expression.rhs)
                    elif _expression.operation == "*":
                        return constant_expression_to_dummy_op(_expression.lhs) * constant_expression_to_dummy_op(_expression.rhs)
                    elif _expression.operation == "/":
                        return constant_expression_to_dummy_op(_expression.lhs) / constant_expression_to_dummy_op(_expression.rhs)
                    elif _expression.operation == "//":
                        return constant_expression_to_dummy_op(_expression.lhs) // constant_expression_to_dummy_op(_expression.rhs)
                    else:
                        raise RuntimeError(f"Unsupported binary operation: {_expression.operation}")
                else:
                    raise RuntimeError(f"Cannot convert {type(_expression)} to dummy op.")
             
            compute_operation_groups: list[ComputeOperationGroup] = collector.get_compute_operation_groups()
            for compute_operation_group in compute_operation_groups:
                print(compute_operation_group)

            # Link loaded tile and element operands to their program level counterparts
            for compute_operation_group in compute_operation_groups:
                for transfer_or_compute_operation in compute_operation_group.transfer_and_compute_operations:
                    if isinstance(transfer_or_compute_operation, StealthLoad):
                        if transfer_or_compute_operation.source_operand_name not in name_to_operand:
                            assert all(isinstance(d, StealthVariableName) for d in collector._operands[transfer_or_compute_operation.source_operand_name].shape), f"Expected all dimensions to be StealthVariableName but instead got {collector._operands[transfer_or_compute_operation.source_operand_name].shape} for {transfer_or_compute_operation.source_operand_name}"
                            temp_operand = cdlt.create_operand_template(transfer_or_compute_operation.source_operand_name, OP_DTYPES, list(map(lambda d: dimension_dummy_ops[d.name], collector._operands[transfer_or_compute_operation.source_operand_name].shape)), default_dtype=input_dtype)
                            cdlt.add_temp_operand(temp_operand)
                            name_to_operand[transfer_or_compute_operation.source_operand_name] = temp_operand
                        name_to_operand[transfer_or_compute_operation.destination_operand_name] = name_to_operand[transfer_or_compute_operation.source_operand_name]

            # Create intermediate operands for complicated codelets
            output_operand_elements = {}
            output_operand_tiles = {}
            intermediate_operands = set()
            for compute_operation_group in compute_operation_groups:
                for transfer_or_compute_operation in reversed(compute_operation_group.transfer_and_compute_operations):
                    if isinstance(transfer_or_compute_operation, StealthStore):
                        for output_operand in collector.get_output_operands():
                            if transfer_or_compute_operation.destination_operand_name == output_operand.name:
                                output_operand_tiles[transfer_or_compute_operation.source_operand_name] = name_to_operand[output_operand.name]
                        if transfer_or_compute_operation.destination_operand_name in output_operand_tiles:
                            output_operand_elements[transfer_or_compute_operation.source_operand_name] = output_operand_tiles[transfer_or_compute_operation.destination_operand_name]
                            name_to_operand[transfer_or_compute_operation.source_operand_name] = output_operand_tiles[transfer_or_compute_operation.destination_operand_name]
                            intermediate_operands.add(transfer_or_compute_operation.source_operand_name)
 
            for compute_operation_group in compute_operation_groups:
                for transfer_or_compute_operation in compute_operation_group.transfer_and_compute_operations:
                    if isinstance(transfer_or_compute_operation, StealthCompute):
                        destination_operand_name = transfer_or_compute_operation.destination_operand_name
                        if destination_operand_name not in output_operand_elements:
                            if transfer_or_compute_operation.operation_name in ["mvmul", "mvmul_bias"]:
                                intermediate_shape = name_to_operand[transfer_or_compute_operation.operands[-1]].shape_list
                            elif transfer_or_compute_operation.operation_name in ["add", "sub", "mul", "div", "max", "min",
                                                                    "rshift", "lshift",
                                                                    "relu", "leaky_relu", "sigmoid", "tanh",
                                                                    "exp", "sqrt", "inv_sqrt", "log2"]:
                                intermediate_shape = name_to_operand[transfer_or_compute_operation.operands[0]].shape_list
                            else:
                                raise NotImplementedError(f"{transfer_or_compute_operation.operation_name} is not implemented yet")

                            name_to_operand[destination_operand_name] = cdlt.create_operand_template(destination_operand_name, OP_DTYPES, intermediate_shape, default_dtype=acc_dtype) 
                            cdlt.add_temp_operand(name_to_operand[destination_operand_name])
                    elif isinstance(transfer_or_compute_operation, StealthStore) and transfer_or_compute_operation.source_operand_name in intermediate_operands:
                        name_to_operand[transfer_or_compute_operation.source_operand_name].start_location = collector._operands[transfer_or_compute_operation.destination_operand_name].location
            
            SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
            
            def systolic_array_start_config(_cdlt) -> None:
                _cdlt.configure("start", "systolic_array")
                _cdlt.configure("start", "WBUF")
                _cdlt.configure("start", "BBUF")
                _cdlt.configure("start", "IBUF")
                _cdlt.configure("start", "OBUF")
            
            def systolic_array_end_config(_cdlt) -> None:
                _cdlt.configure("end", "WBUF")
                _cdlt.configure("end", "BBUF")
                _cdlt.configure("end", "IBUF")
                _cdlt.configure("end", "OBUF")
                _cdlt.configure("end", "systolic_array")
            
            def simd_start_config(_cdlt, immediates: list[tuple[str, Any]]) -> None:
                immediate_dummy_ops = []
                for immediate_name, immediate_value in immediates:
                    immediate_dummy_ops.append(_cdlt.dummy_op(immediate_name, immediate_value))
                    temp_operand = _cdlt.create_temp_operand([SIMD_SIZE], "IMM", name=immediate_name)
                    name_to_operand[str(immediate_value)] = temp_operand
                _cdlt.configure("start", "SIMD")
                for immediate_dummy_op in immediate_dummy_ops:
                    _cdlt.configure("start", immediate_value=immediate_dummy_op)
                    
            def simd_end_config(_cdlt) -> None:
                _cdlt.configure("end", "SIMD")

            compute_operation_index: int = 0 
            for compute_operation_group in compute_operation_groups:
                if compute_operation_group.is_systolic_array_operation:
                    config_functions = (systolic_array_start_config, systolic_array_end_config)
                elif compute_operation_group.is_simd_operation:
                    config_functions = (partial(simd_start_config, immediates=[(immediate_name, constant_expression_to_dummy_op(collector._immediates[immediate_name])) for immediate_name in compute_operation_group.immediates]), simd_end_config)
                else:
                    raise RuntimeError(f"Somehow got a compute operation group that is neither a systolic array operation nor a SIMD operation.")

                # Start Config 
                config_functions[0](cdlt)

                # Loops
                loop_templates = OrderedDict()
                for loop_index_variable in compute_operation_group.inner_loop_index_variables:
                    tile_size = evaluate_expression(loop_index_variable.end)
                    if isinstance(tile_size, StealthBinaryExpression):
                        dimension = dimension_dummy_ops[tile_size.lhs.name]
                    elif isinstance(tile_size, StealthVariableName):
                        dimension = dimension_dummy_ops[tile_size.name]
                    else:
                        raise RuntimeError(f"Invalid tile size for inner loop {loop_index_variable}")
                    loop_template = cdlt.loop(dimension).__enter__()
                    loop_templates[loop_index_variable.name] = loop_template

                # Transfers and compute operations
                body = CodeletTemplateBody()
                for transfer_or_compute_operation in compute_operation_group.transfer_and_compute_operations:
                    if isinstance(transfer_or_compute_operation, StealthLoad):
                        if transfer_or_compute_operation.location in ON_CHIP_LOCATIONS:
                            operand = name_to_operand[transfer_or_compute_operation.source_operand_name]
                            name_to_operand[transfer_or_compute_operation.source_operand_name] = operand
                            name_to_operand[transfer_or_compute_operation.destination_operand_name] = operand
                            body.add_tile_load(transfer_or_compute_operation)
                        else:
                            operand = name_to_operand[transfer_or_compute_operation.source_operand_name]
                            name_to_operand[transfer_or_compute_operation.destination_operand_name] = operand
                            body.add_element_load(transfer_or_compute_operation) 
                for transfer_or_compute_operation in reversed(compute_operation_group.transfer_and_compute_operations):
                    if isinstance(transfer_or_compute_operation, StealthStore):
                        if collector._operands[transfer_or_compute_operation.source_operand_name].location in ON_CHIP_LOCATIONS:
                            operand = name_to_operand[transfer_or_compute_operation.destination_operand_name]
                            name_to_operand[transfer_or_compute_operation.source_operand_name] = operand
                            name_to_operand[transfer_or_compute_operation.destination_operand_name] = operand
                            body.add_tile_store(transfer_or_compute_operation)
                        else:
                            operand = name_to_operand[transfer_or_compute_operation.destination_operand_name]
                            name_to_operand[transfer_or_compute_operation.source_operand_name] = operand
                            body.add_element_store(transfer_or_compute_operation)
                for transfer_or_compute_operation in compute_operation_group.transfer_and_compute_operations:
                    if isinstance(transfer_or_compute_operation, StealthCompute):
                        compute_operation_index += 1
                        body.set_compute(collector._CODELET_COMPUTE_NAME_TO_COMPUTE_OPERATION_NAME[transfer_or_compute_operation.operation_name], transfer_or_compute_operation.operands, [transfer_or_compute_operation.destination_operand_name], transfer_or_compute_operation.location)
                body.finalize(collector, name_to_operand, loop_templates, dimension_dummy_ops)
                body.create_body(cdlt)
                    
                # End Loops
                for loop_template in reversed(loop_templates.values()):
                    loop_template.__exit__(None, None, None)
                
                # End Config
                config_functions[1](cdlt)

        return cdlt

    return codelet_template


class StealthCodelet:
    _codelet_string: str
    _codelet_ast: ParseTree
    _information_collector: CodeletInformationCollector
    _codelet_template: Callable[[ArchitectureGraph], CodeletTemplate]

    def __init__(self, codelet_string: str) -> None:
        self._codelet_string = codelet_string
        self._codelet_ast = parse_stealth_codelet(codelet_string)
        self._information_collector = collect_codelet_information_from_parse_tree(self._codelet_ast, codelet_string)
        self.write_tiling("stealth_outputs/tiling_info/tiling.json")
        self._codelet_template = create_codelet_template_from_codelet_information_collector(self._information_collector)
    
    def write_tiling(self, tiling_path: str) -> None:
        tiling = self._information_collector.get_number_of_tiles_for_each_dimension()
        with open(tiling_path, "w") as f:
            file_contents = {self._information_collector.get_operation_name() + "1": {"1": tiling}}
            json.dump(file_contents, f, indent=4) 
 

# ==================== PolyMath Node Generation ==================== #
class DummyTemplate(pm.Template):
    def __init__(self, *args, **kwargs):
        assert "number_of_inputs" in kwargs
        assert "number_of_outputs" in kwargs
        assert "custom_operation_name" in kwargs
        self._number_of_inputs = kwargs["number_of_inputs"]
        self._number_of_outputs = kwargs["number_of_outputs"]
        custom_operation_name = kwargs["custom_operation_name"]
        kwargs.pop("number_of_inputs")
        kwargs.pop("number_of_outputs")
        kwargs.pop("custom_operation_name")
        super().__init__(*args, **kwargs)
        self.op_name =custom_operation_name
    
    def define_graph(self, *args):
        pass

    @property
    def inputs(self):
        return tuple(self.args[0][i] for i in range(self._number_of_inputs))

    @property
    def outputs(self):
        return tuple(self.args[0][self._number_of_inputs + i] for i in range(self._number_of_outputs))


def create_dummy_polymath_node_from_codelet(codelet: StealthCodelet, dimension_sizes: dict[str, int]) -> pm.Graph:
    with pm.Node(name="test") as graph:
        top_inputs = []
        for operand in codelet._information_collector._inputs:
            input_name = UniqueNameGenerator.get_unique_name("input")
            top_inputs.append(pm.input(name=input_name, shape=tuple(dimension_sizes[s.name] if isinstance(s, StealthVariableName) else s.value for s in operand.shape)))
        top_outputs = []
        for output_operand in codelet._information_collector._outputs:
            output_name = UniqueNameGenerator.get_unique_name("output")
            top_outputs.append(pm.output(name=output_name, shape=tuple(dimension_sizes[s.name] if isinstance(s, StealthVariableName) else s.value for s in output_operand.shape)))
        
        args = top_inputs + top_outputs
        DummyTemplate(*args, number_of_inputs=len(top_inputs), number_of_outputs=len(top_outputs), custom_operation_name=codelet._information_collector.get_operation_name()) 
    return graph


# ==================== Compilation ==================== #
def compile(config_path: str, layer: StealthCodelet, dimension_sizes: dict[str, int]) -> None:
    graph = create_dummy_polymath_node_from_codelet(layer, dimension_sizes)
    cfg = load_config(config_path)
    program = compile_genesys(
        model_name=layer._information_collector.get_operation_name(),
        graph=graph,
        genesys_cfg=cfg,
        custom_codelets={layer._information_collector.get_operation_name(): layer._codelet_template},
        print_config=False,
        benchmark_path="stealth_outputs",
        # store_tiling=True,
        tiling_path="tiling.json"
    )

    sys_array_size = cfg['ARRAY_M']
    dgen = DataGen(program,
                    single_codelets=False,
                    shared_datagen=False,
                    dir_ext=f"benchmark{sys_array_size}x{sys_array_size}",
                    identifier="stealth",
                    generate_data=False,
                    verbose=False,
                    out_path=f"stealth_outputs/compilation_output",
                    store_whole_program=False)
    dgen.generate()


# ==================== Random Operation Generation ==================== #
UNARY_OPS = ["relu", "sigmoid", "tanh", "exp", "sqrt", "inv_sqrt"] 
BINARY_OPS = ["add", "sub", "mul", "div", "max", "min"]


class ComputationGraphNode:
    def __init__(self, operation, inputs):
        self.operation = operation
        self.inputs = inputs
    
    def __repr__(self):
        return f"ComputationGraphNode({self.operation}, {self.inputs})"


def generate_random_dag(num_nodes=10, num_inputs=3, max_tries=100, only_unary_ops=False):
    unary_ops = UNARY_OPS 
    binary_ops = BINARY_OPS
    
    for _ in range(max_tries):
        nodes: list[ComputationGraphNode] = []
        available_inputs = list(range(num_inputs)) 

        for _ in range(num_nodes):
            if only_unary_ops:
                op_type = "unary"
            else:
                op_type = random.choice(["unary", "binary"])
            
            if op_type == "unary":
                if len(available_inputs) == 0:
                    if len(nodes) == 0:
                        continue
                    else:
                        break
                chosen_input = random.choice(available_inputs)
                nodes.append(ComputationGraphNode(random.choice(unary_ops), [chosen_input]))
                available_inputs.append(len(nodes) - 1 + num_inputs)
            else:  
                if len(available_inputs) < 2:
                    if len(nodes) == 0:
                        continue
                    else:
                        break
                chosen_inputs = random.sample(available_inputs, 2)
                nodes.append(ComputationGraphNode(random.choice(binary_ops), chosen_inputs))
                available_inputs.append(len(nodes) - 1 + num_inputs)

    if len(nodes) == 0:
        raise RuntimeError(f"Could not generate a valid DAG for parameters: num_nodes={num_nodes}, num_inputs={num_inputs}, max_tries={max_tries}")
    return nodes


def generate_formula(dag, number_of_inputs):
    SYMBOL_OPERATIONS = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "lshift": "<<",
        "rshift": ">>",
    }
    
    def resolve_node(node_index):
        node = dag[node_index]
        if node.operation not in SYMBOL_OPERATIONS:
            return f"{node.operation}({', '.join(get_input_formula(i) for i in node.inputs)})"
        else:
            return f"({get_input_formula(node.inputs[0])} {SYMBOL_OPERATIONS[node.operation]} {get_input_formula(node.inputs[1])})"
    
    def get_input_formula(inp):
        if inp < number_of_inputs:
            return f"x{inp}"
        else:
            node_index = inp - number_of_inputs
            if node_index not in node_formulas:
                node_formulas[node_index] = resolve_node(node_index)
            return node_formulas[node_index]
    
    node_formulas = {}

    output_nodes = set(range(len(dag)))
    for i, node in enumerate(dag):
        for inp in node.inputs:
            if inp >= number_of_inputs:
                output_nodes.discard(inp - number_of_inputs)
    
    output_formulas = [resolve_node(i) for i in output_nodes]
    
    return tuple(output_formulas)


# ==================== Direct Codelet Generation ==================== #
TAB = "    "


def generate_conv2d_bias_codelet_from_config(stride: int, config: dict) -> str:
    N_tiles = config["N_tiles"]
    OC_tiles = config["OC_tiles"]
    IC_tiles = config["IC_tiles"]
    KH_tiles = config["KH_tiles"]
    KW_tiles = config["KW_tiles"]
    OH_tiles = config["OH_tiles"]
    OW_tiles = config["OW_tiles"]
    array_N = config["array_N"]
    array_M = config["array_M"]
    loop_order = config["loop_order"]

    oc_outer_loop = f"for oc in loop({OC_tiles}, OC // {OC_tiles}):"
    n_outer_loop = f"for n in loop({N_tiles}, N // {N_tiles}):"
    ic_outer_loop = f"for ic in loop({IC_tiles}, IC // {IC_tiles}):"
    kh_outer_loop = f"for kh in loop({KH_tiles}, KH // {KH_tiles}):"
    kw_outer_loop = f"for kw in loop({KW_tiles}, KW // {KW_tiles}):"
    oh_outer_loop = f"for oh in loop({OH_tiles}, OH // {OH_tiles}):"
    ow_outer_loop = f"for ow in loop({OW_tiles}, OW // {OW_tiles}):"
    outer_loop_statements = [oc_outer_loop, n_outer_loop, ic_outer_loop, kh_outer_loop, kw_outer_loop, oh_outer_loop, ow_outer_loop]

    # Create inner loop statements
    oc_inner_loop = f"for oc1 in loop(OC // {OC_tiles}, {array_N}):"
    n_inner_loop = f"for n1 in loop(N // {N_tiles}):"
    ic_inner_loop = f"for ic1 in loop(IC // {IC_tiles}, {array_N}):"
    kh_inner_loop = f"for kh1 in loop(KH // {KH_tiles}):"
    kw_inner_loop = f"for kw1 in loop(KW // {KW_tiles}):"
    oh_inner_loop = f"for oh1 in loop(OH // {OH_tiles}):"
    ow_inner_loop = f"for ow1 in loop(OW // {OW_tiles}):"
    inner_loop_statements = [oc_inner_loop, n_inner_loop, ic_inner_loop, kh_inner_loop, kw_inner_loop, oh_inner_loop, ow_inner_loop]

    # Create statements
    output_alloc = "o = alloc([N, OH, OW, OC], DRAM, i32)"
    bias_tile_load = f"b1 = load(bias[oc], [OC // {OC_tiles}], BBUF)"
    weight_tile_load = f"w1 = load(w[kh, kw, ic, oc], [KH // {KH_tiles}, KW // {KW_tiles}, IC // {IC_tiles}, OC // {OC_tiles}], WBUF)"
    data_tile_load = f"x1 = load(x[n, kh + {stride} * oh, kw + {stride} * ow, ic], [N // {N_tiles}, (OH // {OH_tiles} - 1) * {stride} + (KH // {KH_tiles} - 1) + 1, (OW // {OW_tiles} - 1) * {stride} + (KW // {KW_tiles} - 1) + 1, IC // {IC_tiles}], IBUF)"
    output_tile_load = f"o1 = load(o[n, oh, ow, oc], [N // {N_tiles}, OH // {OH_tiles}, OW // {OW_tiles}, OC // {OC_tiles}], OBUF)"
    bias_element_load = f"b2 = load(b1[oc1], [1, {array_N}], PE_ARRAY)"
    weight_element_load = f"w2 = load(w1[kh1, kw1, ic1, oc1], [{array_N}, {array_M}], PE_ARRAY)"
    data_element_load = f"x2 = load(x1[n1, kh1 + {stride} * oh1, kw1 + {stride} * ow1, ic1], [1, {array_N}], PE_ARRAY)"
    output_element_load = f"o2 = load(o1[n1, oh1, ow1, oc1], [1, {array_N}], PE_ARRAY)"
    compute = "o3 = mvmul_bias(x2, w2, b2, o2, PE_ARRAY)"
    output_element_store = "store(o1[n1, oh1, ow1, oc1], o3)"
    output_tile_store = "store(o[n, oh, ow, oc], o1)"
    output_return = "return o"

    # Put all statements together in correct order
    header = "def conv2d_bias(x: i8[N, IH, IW, IC] @ DRAM, w: i8[KH, KW, IC, OC] @ DRAM, bias: i32[OC] @ DRAM, OH: param, OW: param):\n"
    used_outer_loops = set()
    outer_loops = ""
    tabs = TAB
    is_weight_load = False
    is_bias_load = False
    for o in loop_order:
        outer_loops += tabs + outer_loop_statements[o] + "\n"
        used_outer_loops.add(o)
        tabs += TAB
        if 0 in used_outer_loops and not is_bias_load:
            outer_loops += tabs + bias_tile_load + "\n"
            is_bias_load = True
        if all([w_i in used_outer_loops for w_i in (0, 2, 3, 4)]) and not is_weight_load:
            outer_loops += tabs + weight_tile_load + "\n"
            is_weight_load = True
    outer_loops += tabs + data_tile_load + "\n"
    outer_loops += tabs + output_tile_load + "\n"

    used_inner_loops = set()
    inner_loops = ""
    is_weight_load = False
    is_bias_load = False
    for i in loop_order:
        inner_loops += tabs + inner_loop_statements[i] + "\n"
        used_inner_loops.add(i)
        tabs += TAB
        if 0 in used_inner_loops and not is_bias_load:
            inner_loops += tabs + bias_element_load + "\n"
            is_bias_load = True
        if all([w_i in used_inner_loops for w_i in (0, 2, 3, 4)]) and not is_weight_load:
            inner_loops += tabs + weight_element_load + "\n"
            is_weight_load = True
    inner_loops += tabs + data_element_load + "\n"
    inner_loops += tabs + output_element_load + "\n"
    inner_loops += tabs + compute + "\n"
    inner_loops += tabs + output_element_store + "\n"

    conv2d_bias_codelet = header + TAB + output_alloc + "\n" + outer_loops + inner_loops + (TAB * 8) + output_tile_store + "\n" + TAB + output_return + "\n"
    return conv2d_bias_codelet


def generate_conv2d_bias_relu_codelet_from_config(stride: int, config: dict) -> str:
    N_tiles = config["N_tiles"]
    OC_tiles = config["OC_tiles"]
    IC_tiles = config["IC_tiles"]
    KH_tiles = config["KH_tiles"]
    KW_tiles = config["KW_tiles"]
    OH_tiles = config["OH_tiles"]
    OW_tiles = config["OW_tiles"]
    array_N = config["array_N"]
    array_M = config["array_M"]
    loop_order = config["loop_order"]

    oc_outer_loop = f"for oc in loop({OC_tiles}, OC // {OC_tiles}):"
    n_outer_loop = f"for n in loop({N_tiles}, N // {N_tiles}):"
    ic_outer_loop = f"for ic in loop({IC_tiles}, IC // {IC_tiles}):"
    kh_outer_loop = f"for kh in loop({KH_tiles}, KH // {KH_tiles}):"
    kw_outer_loop = f"for kw in loop({KW_tiles}, KW // {KW_tiles}):"
    oh_outer_loop = f"for oh in loop({OH_tiles}, OH // {OH_tiles}):"
    ow_outer_loop = f"for ow in loop({OW_tiles}, OW // {OW_tiles}):"
    outer_loop_statements = [oc_outer_loop, n_outer_loop, ic_outer_loop, kh_outer_loop, kw_outer_loop, oh_outer_loop, ow_outer_loop]

    # Create inner loop statements
    oc_inner_loop = f"for oc1 in loop(OC // {OC_tiles}, {array_N}):"
    n_inner_loop = f"for n1 in loop(N // {N_tiles}):"
    ic_inner_loop = f"for ic1 in loop(IC // {IC_tiles}, {array_N}):"
    kh_inner_loop = f"for kh1 in loop(KH // {KH_tiles}):"
    kw_inner_loop = f"for kw1 in loop(KW // {KW_tiles}):"
    oh_inner_loop = f"for oh1 in loop(OH // {OH_tiles}):"
    ow_inner_loop = f"for ow1 in loop(OW // {OW_tiles}):"
    inner_loop_statements = [oc_inner_loop, n_inner_loop, ic_inner_loop, kh_inner_loop, kw_inner_loop, oh_inner_loop, ow_inner_loop]

    # Create statements
    output_alloc = "o = alloc([N, OH, OW, OC], DRAM, i32)"
    conv_output_alloc = "conv_o = alloc([N, OH, OW, OC], DRAM, i32)"
    bias_tile_load = f"b1 = load(bias[oc], [OC // {OC_tiles}], BBUF)"
    weight_tile_load = f"w1 = load(w[kh, kw, ic, oc], [KH // {KH_tiles}, KW // {KW_tiles}, IC // {IC_tiles}, OC // {OC_tiles}], WBUF)"
    data_tile_load = f"x1 = load(x[n, kh + {stride} * oh, kw + {stride} * ow, ic], [N // {N_tiles}, (OH // {OH_tiles} - 1) * {stride} + (KH // {KH_tiles} - 1) + 1, (OW // {OW_tiles} - 1) * {stride} + (KW // {KW_tiles} - 1) + 1, IC // {IC_tiles}], IBUF)"
    relu_output_tile_alloc = "relu_o = alloc([N, OH, OW, OC], VMEM1, i32)"
    output_tile_load = f"o1 = load(conv_o[n, oh, ow, oc], [N // {N_tiles}, OH // {OH_tiles}, OW // {OW_tiles}, OC // {OC_tiles}], OBUF)"
    bias_element_load = f"b2 = load(b1[oc1], [1, {array_N}], PE_ARRAY)"
    weight_element_load = f"w2 = load(w1[kh1, kw1, ic1, oc1], [{array_N}, {array_M}], PE_ARRAY)"
    data_element_load = f"x2 = load(x1[n1, kh1 + {stride} * oh1, kw1 + {stride} * ow1, ic1], [1, {array_N}], PE_ARRAY)"
    output_element_load = f"o2 = load(o1[n1, oh1, ow1, oc1], [1, {array_N}], PE_ARRAY)"
    compute_1 = "o3 = mvmul_bias(x2, w2, b2, o2, PE_ARRAY)"
    intermediate_output_element_store = "store(o1[n1, oh1, ow1, oc1], o3)"
    intermediate_output_element_load = f"o4 = load(o1[n1, oh1, ow1, oc1], [{array_N}], SIMD)"
    compute_2 = "o5 = relu(o4, SIMD)"
    output_element_store = "store(relu_o[n1, oh1, ow1, oc1], o5)"
    output_tile_store = "store(o[n, oh, ow, oc], relu_o)"
    output_return = "return o"

    # Put all statements together in correct order
    header = "def conv2d_bias_relu(x: i8[N, IH, IW, IC] @ DRAM, w: i8[KH, KW, IC, OC] @ DRAM, bias: i32[OC] @ DRAM, OH: param, OW: param):\n"
    used_outer_loops = set()
    outer_loops = ""
    tabs = TAB
    is_weight_load = False
    is_bias_load = False
    for o in loop_order:
        outer_loops += tabs + outer_loop_statements[o] + "\n"
        used_outer_loops.add(o)
        tabs += TAB
        if 0 in used_outer_loops and not is_bias_load:
            outer_loops += tabs + bias_tile_load + "\n"
            is_bias_load = True
        if all([w_i in used_outer_loops for w_i in (0, 2, 3, 4)]) and not is_weight_load:
            outer_loops += tabs + weight_tile_load + "\n"
            is_weight_load = True
    outer_loops += tabs + data_tile_load + "\n"
    outer_loops += tabs + relu_output_tile_alloc + "\n"
    outer_loops += tabs + output_tile_load + "\n"

    used_inner_loops = set()
    inner_loops = ""
    is_weight_load = False
    is_bias_load = False
    for i in loop_order:
        inner_loops += tabs + inner_loop_statements[i] + "\n"
        used_inner_loops.add(i)
        tabs += TAB
        if 0 in used_inner_loops and not is_bias_load:
            inner_loops += tabs + bias_element_load + "\n"
            is_bias_load = True
        if all([w_i in used_inner_loops for w_i in (0, 2, 3, 4)]) and not is_weight_load:
            inner_loops += tabs + weight_element_load + "\n"
            is_weight_load = True
    inner_loops += tabs + data_element_load + "\n"
    inner_loops += tabs + output_element_load + "\n"
    inner_loops += tabs + compute_1 + "\n"
    inner_loops += tabs + intermediate_output_element_store + "\n"
    inner_loops += tabs + intermediate_output_element_load + "\n"
    inner_loops += tabs + compute_2 + "\n"
    inner_loops += tabs + output_element_store + "\n"

    conv2d_bias_relu_codelet = header + TAB + output_alloc + "\n" + TAB + conv_output_alloc + "\n" + outer_loops + inner_loops + (TAB * 8) + output_tile_store + "\n" + TAB + output_return + "\n"
    return conv2d_bias_relu_codelet


def generate_gemm_bias_codelet_from_config(config: dict) -> str:
    N_tiles = config["N_tiles"]
    M_tiles = config["M_tiles"]
    P_tiles = config["P_tiles"]
    array_N = config["array_N"]
    array_M = config["array_M"]
    loop_order = config["loop_order"]

    # Create outer loop statements
    p_outer_loop = f"for p in loop({P_tiles}, P // {P_tiles}):"
    n_outer_loop = f"for n in loop({N_tiles}, N // {N_tiles}):"
    m_outer_loop = f"for m in loop({M_tiles}, M // {M_tiles}):"
    outer_loop_statements = [p_outer_loop, n_outer_loop, m_outer_loop]

    # Create inner loop statements
    p_inner_loop = f"for p1 in loop(P // {P_tiles}, {array_N}):"
    n_inner_loop = f"for n1 in loop(N // {N_tiles}, {array_N}):"
    m_inner_loop = f"for m1 in loop(M // {M_tiles}):"
    inner_loop_statements = [p_inner_loop, n_inner_loop, m_inner_loop]

    # Create statements
    output_alloc = "o = alloc([M, P], DRAM, i32)"
    bias_tile_load = f"b1 = load(bias[p], [P // {P_tiles}], BBUF)"
    weight_tile_load = f"w1 = load(w[n, p], [N // {N_tiles}, P // {P_tiles}], WBUF)"
    data_tile_load = f"x1 = load(x[m, n], [M // {M_tiles}, N // {N_tiles}], IBUF)"
    output_tile_load = f"o1 = load(o[m, p], [M // {M_tiles}, P // {P_tiles}], OBUF)"
    bias_element_load = f"b2 = load(b1[p1], [1, {array_N}], PE_ARRAY)"
    weight_element_load = f"w2 = load(w1[n1, p1], [{array_N}, {array_M}], PE_ARRAY)"
    data_element_load = f"x2 = load(x1[m1, n1], [1, {array_N}], PE_ARRAY)"
    output_element_load = f"o2 = load(o1[m1, p1], [1, {array_N}], PE_ARRAY)"
    compute = "o3 = mvmul_bias(x2, w2, b2, o2, PE_ARRAY)"
    output_element_store = "store(o1[m1, p1], o3)"
    output_tile_store = "store(o[m, p], o1)"
    output_return = "return o"

    # Put all statements together in correct order
    header = "def gemm_bias(x: i8[M, N] @ DRAM, w: i8[N, P] @ DRAM, bias: i32[P] @ DRAM):\n"
    used_outer_loops = set()
    outer_loops = ""
    tabs = TAB
    is_data_load = False
    is_weight_load = False
    is_bias_load = False
    is_output_load = False
    for o in loop_order:
        outer_loops += tabs + outer_loop_statements[o] + "\n"
        used_outer_loops.add(o)
        tabs += TAB
        if 0 in used_outer_loops and not is_bias_load:
            outer_loops += tabs + bias_tile_load + "\n"
            is_bias_load = True
        if all([i_i in used_outer_loops for i_i in (1, 2)]) and not is_data_load:
            outer_loops += tabs + data_tile_load + "\n"
            is_data_load = True
        if all([w_i in used_outer_loops for w_i in (0, 1)]) and not is_weight_load:
            outer_loops += tabs + weight_tile_load + "\n"
            is_weight_load = True
        if all([o_i in used_outer_loops for o_i in (0, 2)]) and not is_output_load:
            outer_loops += tabs + output_tile_load + "\n"
            is_output_load = True

    used_inner_loops = set()
    inner_loops = ""
    is_data_load = False
    is_weight_load = False
    is_bias_load = False
    is_output_load = False
    for i in loop_order:
        inner_loops += tabs + inner_loop_statements[i] + "\n"
        used_inner_loops.add(i)
        tabs += TAB
        if 0 in used_inner_loops and not is_bias_load:
            inner_loops += tabs + bias_element_load + "\n"
            is_bias_load = True
        if all([i_i in used_inner_loops for i_i in (1, 2)]) and not is_data_load:
            inner_loops += tabs + data_element_load + "\n"
            is_data_load = True
        if all([w_i in used_inner_loops for w_i in (0, 1)]) and not is_weight_load:
            inner_loops += tabs + weight_element_load + "\n"
            is_weight_load = True
        if all([o_i in used_inner_loops for o_i in (0, 2)]) and not is_output_load:
            inner_loops += tabs + output_element_load + "\n"
            is_output_load = True

    inner_loops += tabs + compute + "\n"
    inner_loops += tabs + output_element_store + "\n"

    gemm_bias_codelet: str = header + TAB + output_alloc + "\n" + outer_loops + inner_loops + (TAB * 4) + output_tile_store + "\n" + TAB + output_return + "\n"
    return gemm_bias_codelet


def generate_relu4d_codelet_from_config(config: dict) -> str:
    N_tiles = config["N_tiles"]
    C_tiles = config["C_tiles"]
    H_tiles = config["H_tiles"]
    W_tiles = config["W_tiles"]
    array_N = config["array_N"]
    loop_order = config["loop_order"]

    # Create outer loop statements
    n_outer_loop = f"for n in loop({N_tiles}, N // {N_tiles}):"
    c_outer_loop = f"for c in loop({C_tiles}, C // {C_tiles}):"
    h_outer_loop = f"for h in loop({H_tiles}, H // {H_tiles}):"
    w_outer_loop = f"for w in loop({W_tiles}, W // {W_tiles}):"
    outer_loop_statements = [n_outer_loop, c_outer_loop, h_outer_loop, w_outer_loop]

    # Create inner loop statements
    n_inner_loop = f"for n1 in loop(N // {N_tiles}):"
    c_inner_loop = f"for c1 in loop(C // {C_tiles}, {array_N}):"
    h_inner_loop = f"for h1 in loop(H // {H_tiles}):"
    w_inner_loop = f"for w1 in loop(W // {W_tiles}):"
    inner_loop_statements = [n_inner_loop, c_inner_loop, h_inner_loop, w_inner_loop]

    # Create statements
    output_alloc = "o = alloc([N, H, W, C], DRAM, i32)"
    data_tile_load = f"x1 = load(x[n, h, w, c], [N // {N_tiles}, H // {H_tiles}, W // {W_tiles}, C // {C_tiles}], VMEM1)"
    output_tile_alloc = f"o1 = alloc([N // {N_tiles}, H // {H_tiles}, W // {W_tiles}, C // {C_tiles}], VMEM1, i32)"
    data_element_load = f"x2 = load(x1[n1, h1, w1, c1], [{array_N}], SIMD)"
    compute = "o2 = relu(x2, SIMD)"
    output_element_store = "store(o1[n1, h1, w1, c1], o2)"
    output_tile_store = "store(o[n, h, w, c], o1)"
    output_return = "return o"

    # Put all statements together in correct order
    header = "def relu4d(x: i32[N, H, W, C] @ DRAM):\n"

    outer_loops = ""
    tabs = TAB
    for o in loop_order:
        outer_loops += tabs + outer_loop_statements[o] + "\n"
        tabs += TAB
    outer_loops += tabs + output_tile_alloc + "\n"
    outer_loops += tabs + data_tile_load + "\n"

    inner_loops = ""
    for i in loop_order:
        inner_loops += tabs + inner_loop_statements[i] + "\n"
        tabs += TAB
    inner_loops += tabs + data_element_load + "\n"
    inner_loops += tabs + compute + "\n"
    inner_loops += tabs + output_element_store + "\n"

    relu_codelet: str = header + TAB + output_alloc + "\n" + outer_loops + inner_loops + (TAB * 5) + output_tile_store + "\n" + TAB + output_return + "\n"
    return relu_codelet


def generate_4d_simd_ops_codelet_from_config(config: dict) -> str:
    N_tiles = config["N_tiles"]
    C_tiles = config["C_tiles"]
    H_tiles = config["H_tiles"]
    W_tiles = config["W_tiles"]
    array_N = config["array_N"]
    loop_order = config["loop_order"]
    operations: list[ComputationGraphNode] = config["operations"]
    number_of_inputs = config["number_of_inputs"]
    inputs_vmem_1_or_2 = config["inputs_vmem_1_or_2"]
    outputs_vmem_1_or_2 = config["outputs_vmem_1_or_2"]

    input_operand_names = [f"codelet_input_{i}" for i in range(number_of_inputs)]
    operand_index_to_compute_output_operand = {i: f"compute_output_{i}" for i in range(number_of_inputs, number_of_inputs + len(operations))}
    operand_index_to_compute_output_operand.update({i: f"codelet_input_{i}" for i in range(number_of_inputs)})
    nodes_used_as_input = set()
    for op in operations:
        nodes_used_as_input.update(op.inputs)
    output_operand_names = set()
    for i, op in enumerate(operations):
        if (i + number_of_inputs) not in nodes_used_as_input: 
            output_operand_names.add(operand_index_to_compute_output_operand[number_of_inputs + i]) 
    number_of_outputs = len(output_operand_names)
    assert number_of_outputs > 0, "No outputs found for SIMD ops codelet"

    # Create outer loop statements
    n_outer_loop = f"for n in loop({N_tiles}, N // {N_tiles}):"
    c_outer_loop = f"for c in loop({C_tiles}, C // {C_tiles}):"
    h_outer_loop = f"for h in loop({H_tiles}, H // {H_tiles}):"
    w_outer_loop = f"for w in loop({W_tiles}, W // {W_tiles}):"
    outer_loop_statements = [n_outer_loop, c_outer_loop, h_outer_loop, w_outer_loop]

    # Create inner loop statements
    n_inner_loop = f"for n1 in loop(N // {N_tiles}):"
    c_inner_loop = f"for c1 in loop(C // {C_tiles}, {array_N}):"
    h_inner_loop = f"for h1 in loop(H // {H_tiles}):"
    w_inner_loop = f"for w1 in loop(W // {W_tiles}):"
    inner_loop_statements = [n_inner_loop, c_inner_loop, h_inner_loop, w_inner_loop]

    # Create statements
    output_allocs = [f"{TAB}{output_operand_name} = alloc([N, H, W, C], DRAM, i32)" for output_operand_name in output_operand_names]
    data_tile_loads = [f"{input_operand_name}_tile = load({input_operand_name}[n, h, w, c], [N // {N_tiles}, H // {H_tiles}, W // {W_tiles}, C // {C_tiles}], VMEM{inputs_vmem_1_or_2[i]})" for i, input_operand_name in enumerate(input_operand_names)]
    output_tile_allocs = [f"{operand_index_to_compute_output_operand[number_of_inputs + i]}_tile = alloc([N // {N_tiles}, H // {H_tiles}, W // {W_tiles}, C // {C_tiles}], VMEM{outputs_vmem_1_or_2[i]}, i32)" for i in range(len(operations))]
    operation_statements = []
    for i, operation in enumerate(operations):
        compute_inputs = []
        for input_index in operation.inputs:
            input_name = UniqueNameGenerator.get_unique_name(operand_index_to_compute_output_operand[input_index] + "_element")
            compute_inputs.append(input_name)
            operation_statements.append(f"{input_name} = load({operand_index_to_compute_output_operand[input_index]}_tile[n1, h1, w1, c1], [{array_N}], SIMD)")
        operation_statements.append(f"{operand_index_to_compute_output_operand[i + number_of_inputs]}_element = {operation.operation}({', '.join(compute_inputs)}, SIMD)")
        operation_statements.append(f"store({operand_index_to_compute_output_operand[i + number_of_inputs]}_tile[n1, h1, w1, c1], {operand_index_to_compute_output_operand[i + number_of_inputs]}_element)")
    output_tile_stores = [f"store({output_operand_name}[n, h, w, c], {output_operand_name}_tile)" for output_operand_name in output_operand_names]
    output_return = "return " + ", ".join(output_operand_names)

    # Put all statements together in correct order
    operation_name = "_".join([operation.operation + "4d" for operation in operations])
    header = "def " + operation_name + "(" + ", ".join(map(lambda n: f"{n}: i32[N, H, W, C] @ DRAM", input_operand_names)) + "):\n" 

    outer_loops = ""
    tabs = TAB
    for o in loop_order:
        outer_loops += tabs + outer_loop_statements[o] + "\n"
        tabs += TAB
    outer_loops += "\n".join(map(lambda s: tabs + s, output_tile_allocs)) + "\n"
    outer_loops += "\n".join(map(lambda s: tabs + s, data_tile_loads)) + "\n"

    inner_loops = ""
    for i in loop_order:
        inner_loops += tabs + inner_loop_statements[i] + "\n"
        tabs += TAB
    inner_loops += "\n".join(map(lambda s: tabs + s, operation_statements)) + "\n"

    codelet: str = header + "\n".join(output_allocs) + "\n" + outer_loops + inner_loops + "\n".join(map(lambda s: TAB * 5 + s, output_tile_stores)) + "\n" + TAB + output_return + "\n"
    return codelet


# ==================== Unit Test Shape Generation ==================== #
def get_2d_image_N() -> set[int]:
    Ns = [1]
    return set(Ns)


def get_2d_image_C() -> set[int]:
    PARAM_BUF_BW = 64
    Cs = [64, 128, 256]
    return set(Cs)


def get_2d_image_H_W() -> set[int]:
    Hs = [32, 128, 256]
    return set(Hs)


def get_2d_image_kernel_size() -> set[int]:
    kernel_sizes = [1, 3, 5, 7]
    return set(kernel_sizes)


# ==================== Codelet Search Space Creation ==================== #
class SearchSpacePoint(abc.ABC):
    @abc.abstractmethod
    def __hash__(self) -> int:
        ...
    
    @abc.abstractmethod
    def __eq__(self, other: Any) -> bool:
        ...


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


class IntegerSpacePoint(SearchSpacePoint):
    _value: int

    def __init__(self, value: int) -> None:
        super().__init__()
        self._value = value
    
    @property
    def value(self) -> int:
        return self._value
    
    def __hash__(self) -> int:
        return hash(self._value)
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, IntegerSpacePoint):
            return False
        return self._value == other._value

    def __repr__(self) -> str:
        return str(self._value)


class IntegerSpace(SearchSpace):
    _min_value: int
    _max_value: int

    def __init__(self, min_value: int, max_value: int) -> None:
        super().__init__()
        self._min_value = min_value
        self._max_value = max_value
    
    def get_space(self) -> set[IntegerSpacePoint]:
        return set(IntegerSpacePoint(i) for i in range(self._min_value, self._max_value + 1))
    
    def get_random_point(self) -> IntegerSpacePoint:
        return IntegerSpacePoint(random.randint(self._min_value, self._max_value))

    def get_cache_key(self) -> Any:
        return (self._min_value, self._max_value)


class DimensionSizeSpacePoint(IntegerSpacePoint):
    def __init__(self, dimension_size: int) -> None:
        super().__init__(dimension_size) 
    
    @property
    def dimension_size(self) -> int:
        return self.value


class DimensionSizeSpace(SearchSpace):
    _generating_functions: tuple[Callable[[], set[int]], ...]

    def __init__(self, generating_functions: tuple[Callable[[], set[int]], ...]) -> None:
        super().__init__()
        self._generating_functions = generating_functions

    def get_space(self) -> set[DimensionSizeSpacePoint]:
        space = set()
        for generating_function in self._generating_functions:
            space.update(DimensionSizeSpacePoint(i) for i in generating_function())
        return space
    
    def get_random_point(self) -> DimensionSizeSpacePoint:
        generating_function = random.choice(self._generating_functions) 
        return DimensionSizeSpacePoint(random.choice(list(generating_function())))

    def get_cache_key(self) -> Any:
        return self._generating_functions


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


class LoopOrderSpacePoint(SearchSpacePoint):
    _loop_order: tuple[int, ...]

    def __init__(self, loop_order: tuple[int, ...]) -> None:
        super().__init__()
        self._loop_order = loop_order
    
    @property
    def loop_order(self) -> tuple[int, ...]:
        return self._loop_order
    
    def __hash__(self) -> int:
        return hash(self._loop_order)
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, LoopOrderSpacePoint):
            return False
        return self._loop_order == other._loop_order

    def __repr__(self) -> str:
        return str(self._loop_order)


class LoopOrderSearchSpace(SearchSpace):
    _number_of_loops: int

    def __init__(self, number_of_loops: int) -> None:
        super().__init__()
        self._number_of_loops = number_of_loops
    
    def get_space(self) -> set[LoopOrderSpacePoint]:
        space = set()
        for permutation in itertools.permutations(range(self._number_of_loops)):
            space.add(LoopOrderSpacePoint(permutation))
        return space
    
    def get_random_point(self) -> LoopOrderSpacePoint:
        loop_order = list(range(self._number_of_loops))
        random.shuffle(loop_order)
        return LoopOrderSpacePoint(tuple(l for l in loop_order))

    def get_cache_key(self) -> Any:
        return self._number_of_loops


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


class ExhaustiveSearcher(Searcher):
    _search_space_contents: list[list[SearchSpacePoint]]
    _search_space_iterators: list[Iterator[SearchSpacePoint]]
    _current_output: Optional[list[SearchSpacePoint]]

    def __init__(self, search_spaces: list[SearchSpace]) -> None:
        super().__init__(search_spaces)
        self._search_space_contents = [list(search_space.get_space()) for search_space in self._search_spaces]
        self._search_space_iterators = [iter(search_space) for search_space in self._search_space_contents]
        self._current_output = None
    
    def reset(self):
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


class RandomSearcher(Searcher):
    _allow_repeats: bool
    _seen_points: set[tuple[SearchSpacePoint, ...]]
    _max_attempts: int

    def __init__(self, search_spaces: list[SearchSpace], allow_repeats=False, max_attempts=100) -> None:
        super().__init__(search_spaces)
        self._allow_repeats = allow_repeats
        self._seen_points = set()
        self._max_attempts = max_attempts
    
    def reset(self):
        self._seen_points = set()
    
    def get_next_search_space_point(self) -> Optional[tuple[SearchSpacePoint, ...]]:
        for _ in range(self._max_attempts):
            point = tuple(search_space.get_random_point() for search_space in self._search_spaces)
            if not self._is_point_seen(point):
                self._seen_points.add(point)
                return point
        return None
    
    def _is_point_seen(self, point: tuple[SearchSpacePoint, ...]) -> bool:
        if self._allow_repeats:
            return False
        else:
            return point in self._seen_points


# ==================== Running Codelets ==================== #
def run_layer(layer_directory: str):
    from simulator.genesys_sim.genesys import run_single_test
    config_path = f"simulator/configs/"
    test_path_components = layer_directory.split("/")
    test_path = layer_directory
    test_name = test_path_components[-1]
    mode = "perf"
    test_output = run_single_test(config_path, mode, {"name": test_name, "path": test_path})
    return test_output


def get_simulator_output_field(simulator_output, field):
    return simulator_output[2][simulator_output[1].index("totCycles")]


def get_relevant_simulator_outputs(simulator_output):
    return {
        "total_cycles": int(get_simulator_output_field(simulator_output, "totCycles")),
        "sa_cycles": int(get_simulator_output_field(simulator_output, "systotalCycles")),
        "sa_compute_cycles_per_tile": float(get_simulator_output_field(simulator_output, "sysComputeCyclesPerTile")),
        "sa_load_cycles_per_tile": float(get_simulator_output_field(simulator_output, "sysLoadCyclesPerTile")),
        "sa_store_cycles_per_tile": float(get_simulator_output_field(simulator_output, "sysStoreCyclesPerTile")),
        "sa_ibuf_util_per_tile": float(get_simulator_output_field(simulator_output, "perTileIbufUtil")),
        "sa_obuf_util_per_tile": float(get_simulator_output_field(simulator_output, "perTileObufUtil")),
        "sa_wbuf_util_per_tile": float(get_simulator_output_field(simulator_output, "perTileWbufUtil")),
        "sa_bbuf_util_per_tile": float(get_simulator_output_field(simulator_output, "perTileBbufUtil")),
        "sa_compute_util_per_tile": float(get_simulator_output_field(simulator_output, "perTileComputeUtils")),
        "simd_cycles": int(get_simulator_output_field(simulator_output, "simdtotalCycles")),
        "simd_compute_cycles_per_tile": int(get_simulator_output_field(simulator_output, "simdComputeCyclesPerTile")),
        "simd_load_cycles": float(get_simulator_output_field(simulator_output, "simdLoadCycles")),
        "simd_store_cycles": float(get_simulator_output_field(simulator_output, "simdStoreCycles")),
    }


# ==================== Codelet Generation for 2D Images ==================== #
def generate_relu_codelets_for_2d_images(num_configs: int = 100):
    LOOP_ORDER_MAP = {
        0: "N",
        1: "C",
        2: "H",
        3: "W"
    }
    N_space = DimensionSizeSpace((get_2d_image_N,))
    C_space = DimensionSizeSpace((get_2d_image_C,))
    H_W_space = DimensionSizeSpace((get_2d_image_H_W,))
    dimension_searcher = ExhaustiveSearcher((N_space, C_space, H_W_space))
    unit_test_dimensions = [p for p in dimension_searcher]

    max_dimension_sizes = [0, 0, 0]
    for unit_test_dimension in unit_test_dimensions:
        for i in range(3):
            max_dimension_sizes[i] = max(max_dimension_sizes[i], unit_test_dimension[i].dimension_size)
    
    N_tile_space = N_tile_space = TileSpace([s.value for s in N_space.get_space()])
    C_tile_space = TileSpace([s.value for s in C_space.get_space()])
    H_tile_space = TileSpace([s.value for s in H_W_space.get_space()])
    W_tile_space = TileSpace([s.value for s in H_W_space.get_space()])
    loop_order_space = LoopOrderSearchSpace(4)

    config_searcher = RandomSearcher((N_tile_space, C_tile_space, H_tile_space, W_tile_space, loop_order_space))
    output = {"data": []}
    for _ in tqdm.tqdm(range(num_configs)):
        current_config_output = {}
        N_tiles, C_tiles, H_tiles, W_tiles, loop_order = config_searcher.get_next_search_space_point()
        codelet_string = generate_relu4d_codelet_from_config({"N_tiles": N_tiles.number_of_tiles, "C_tiles": C_tiles.number_of_tiles, "H_tiles": H_tiles.number_of_tiles, "W_tiles": W_tiles.number_of_tiles, "array_N": 16, "loop_order": loop_order.loop_order})
        current_config_output["codelet"] = codelet_string
        current_config_output["N_tiles"] = N_tiles.number_of_tiles
        current_config_output["C_tiles"] = C_tiles.number_of_tiles
        current_config_output["H_tiles"] = H_tiles.number_of_tiles
        current_config_output["W_tiles"] = W_tiles.number_of_tiles
        current_config_output["loop_order"] = [LOOP_ORDER_MAP[l] for l in loop_order.loop_order]
        codelet = StealthCodelet(codelet_string)
        
        unit_test_outputs = []
        for N, C, H_W in dimension_searcher: 
            N_value = N.dimension_size
            C_value = C.dimension_size
            H_W_value = H_W.dimension_size

            if ((N_tiles.number_of_tiles * N_tiles.tile_size) > N_value) or ((C_tiles.number_of_tiles * C_tiles.tile_size) > C_value) or ((H_tiles.number_of_tiles * H_tiles.tile_size) > H_W_value) or ((W_tiles.number_of_tiles * W_tiles.tile_size) > H_W_value):
                print(f"Skipping unit test {N_value}x{C_value}x{H_W_value}x{H_W_value} because tile split is too large.")
                continue
            if (N_value % N_tiles.number_of_tiles != 0) or (C_value % C_tiles.number_of_tiles != 0) or (H_W_value % H_tiles.number_of_tiles != 0) or (H_W_value % W_tiles.number_of_tiles != 0):
                print(f"Skipping unit test {N_value}x{C_value}x{H_W_value}x{H_W_value} because tile split does not divide dimension size.")
                continue
            try:
                compile("../codelets/examples/genesys/configs/benchmark_16x16.json", codelet, {"N": N_value, "C": C_value, "H": H_W_value, "W": H_W_value})
            except Exception as e:
                print(f"Skipping unit test {N_value}x{C_value}x{H_W_value}x{H_W_value} because tiling does not satisfy constraints:")
                print(e)
                continue
            simulator_outputs = run_layer("stealth_outputs/compilation_output/test_benchmark16x16_stealth")
            relevant_simulator_outputs = get_relevant_simulator_outputs(simulator_outputs)
            relevant_simulator_outputs["input_dimensions"] = [N_value, C_value, H_W_value, H_W_value],
            unit_test_outputs.append(get_relevant_simulator_outputs(simulator_outputs))

        current_config_output["unit_test_outputs"] = unit_test_outputs
        output["data"].append(current_config_output)
    
    with open("stealth_outputs/dataset_files/relu4d.json", "w") as f:
        json.dump(output, f, indent=4)


def generate_simd_unary_element_wise_operations():
    operations: list[list[tuple[int, ComputationGraphNode]]] = []
    for op in UNARY_OPS:
        operations.append((1, [ComputationGraphNode(op, [0])]))
    return operations 


def generate_simd_binary_element_wise_operations():
    operations: list[list[tuple[int, ComputationGraphNode]]] = []
    for op in BINARY_OPS:
        operations.append((2, [ComputationGraphNode(op, [0, 1])]))
    return operations


def generate_simd_element_wise_operations(max_num_inputs=3, max_num_nodes=5, num_operations=1000):
    operations: list[list[tuple[int, ComputationGraphNode]]] = []
    # operations.extend(generate_simd_unary_element_wise_operations())
    # operations.extend(generate_simd_binary_element_wise_operations())
    num_remaining_ops = num_operations - len(operations)
    for num_inputs in range(1, max_num_inputs + 1):
        for num_nodes in range(2, max_num_nodes + 1):
            for _ in range(int(num_remaining_ops / max_num_inputs / (max_num_nodes - 1))):
                operations.append((num_inputs, generate_random_dag(num_inputs=num_inputs, num_nodes=num_nodes, only_unary_ops=num_inputs == 1)))
    while len(operations) < num_operations:
        num_inputs = random.randint(2, max_num_inputs)
        operations.append((num_inputs, generate_random_dag(num_inputs=num_inputs, num_nodes=random.randint(1, max_num_nodes))))
    return operations


def generate_simd_element_wise_operation_codelets_for_2d_images(operations: list[tuple[int, list[ComputationGraphNode]]], num_random_configs_per_exhaustive_config: int = 10, tag="simd_element_wise_operation_on_2d_images"):
    LOOP_ORDER_MAP = {
        0: "N",
        1: "C",
        2: "H",
        3: "W"
    }

    N_space = DimensionSizeSpace((get_2d_image_N,))
    C_space = DimensionSizeSpace((get_2d_image_C,))
    H_W_space = DimensionSizeSpace((get_2d_image_H_W,))
    dimension_searcher = ExhaustiveSearcher((N_space, C_space, H_W_space))
    unit_test_dimensions = [p for p in dimension_searcher]

    max_dimension_sizes = [0, 0, 0]
    for unit_test_dimension in unit_test_dimensions:
        for i in range(3):
            max_dimension_sizes[i] = max(max_dimension_sizes[i], unit_test_dimension[i].dimension_size)
    
    N_tile_space = TileSpace([s.value for s in N_space.get_space()])
    C_tile_space = TileSpace([s.value for s in C_space.get_space()])
    H_tile_space = TileSpace([s.value for s in H_W_space.get_space()])
    W_tile_space = TileSpace([s.value for s in H_W_space.get_space()])
    loop_order_space = LoopOrderSearchSpace(4) 

    exhaustive_config_searcher = ExhaustiveSearcher((N_tile_space, C_tile_space, H_tile_space, W_tile_space))

    for operation in operations:
        operation_name = "_".join([op.operation for op in operation[1]])
        if operation_name in os.listdir("stealth_outputs/partial_dataset_files"):
            continue
        operation_output = {}
        operation_output["tags"] = [tag]
        operation_output["operation"] = operation_name
        operation_output["formula"] = generate_formula(operation[1], operation[0])
        operation_output["configs"] = []
        print(exhaustive_config_searcher.get_size_of_search_space())
        for N_tiles, C_tiles, H_tiles, W_tiles in tqdm.tqdm(exhaustive_config_searcher):
            random_config_searcher = RandomSearcher((loop_order_space,))
            for _ in range(num_random_configs_per_exhaustive_config):
                (loop_order, ) = random_config_searcher.get_next_search_space_point()
                current_config_output = {}
                codelet_string = generate_4d_simd_ops_codelet_from_config(
                    {
                        "N_tiles": N_tiles.number_of_tiles,
                        "C_tiles": C_tiles.number_of_tiles, 
                        "H_tiles": H_tiles.number_of_tiles, 
                        "W_tiles": W_tiles.number_of_tiles, 
                        "array_N": 16, 
                        "loop_order": loop_order.loop_order, 
                        "number_of_inputs": operation[0],
                        "operations": operation[1], 
                        "inputs_vmem_1_or_2": [random.choice([1, 2]) for _ in range(operation[0])], 
                        "outputs_vmem_1_or_2": [random.choice([1, 2]) for _ in range(len(operation[1]))]
                      }
                    )
                print(codelet_string)
                current_config_output["codelet"] = codelet_string
                current_config_output["N_tiles"] = N_tiles.number_of_tiles
                current_config_output["C_tiles"] = C_tiles.number_of_tiles
                current_config_output["H_tiles"] = H_tiles.number_of_tiles
                current_config_output["W_tiles"] = W_tiles.number_of_tiles
                current_config_output["loop_order"] = [LOOP_ORDER_MAP[l] for l in loop_order.loop_order]
                codelet = StealthCodelet(codelet_string)
                
                unit_test_outputs = []
                for N, C, H_W in dimension_searcher: 
                    N_value = N.dimension_size
                    C_value = C.dimension_size
                    H_W_value = H_W.dimension_size

                    if ((N_tiles.number_of_tiles * N_tiles.tile_size) > N_value) or ((C_tiles.number_of_tiles * C_tiles.tile_size) > C_value) or ((H_tiles.number_of_tiles * H_tiles.tile_size) > H_W_value) or ((W_tiles.number_of_tiles * W_tiles.tile_size) > H_W_value):
                        print(f"Skipping unit test {N_value}x{C_value}x{H_W_value}x{H_W_value} because tile split is too large.")
                        continue
                    if (N_value % N_tiles.number_of_tiles != 0) or (C_value % C_tiles.number_of_tiles != 0) or (H_W_value % H_tiles.number_of_tiles != 0) or (H_W_value % W_tiles.number_of_tiles != 0):
                        print(f"Skipping unit test {N_value}x{C_value}x{H_W_value}x{H_W_value} because tile split does not divide dimension size.")
                        continue
                    try:
                        compile("../codelets/examples/genesys/configs/benchmark_16x16.json", codelet, {"N": N_value, "C": C_value, "H": H_W_value, "W": H_W_value})
                    except RuntimeError as e:
                        print(f"Skipping unit test {N_value}x{C_value}x{H_W_value}x{H_W_value} because tiling does not satisfy constraints:")
                        print(e)
                        continue
                    except Exception as e:
                        print(f"Failing unit test for the following error:")
                        raise e
                    simulator_outputs = run_layer("stealth_outputs/compilation_output/test_benchmark16x16_stealth")
                    relevant_simulator_outputs = get_relevant_simulator_outputs(simulator_outputs)
                    relevant_simulator_outputs["input_dimensions"] = [N_value, C_value, H_W_value, H_W_value],
                    unit_test_outputs.append(relevant_simulator_outputs)

                current_config_output["unit_test_outputs"] = unit_test_outputs
                if len(unit_test_outputs) > 0:
                    operation_output["configs"].append(current_config_output)
        
        if len(operation_output["configs"]) > 0:
            with open(f"stealth_outputs/partial_dataset_files/{operation_name}.json", "w") as f:
                json.dump(operation_output, f, indent=4) 
        else:
            raise RuntimeError("No valid configs found for operation. Try increasing the number of tried configs")


if __name__ == "__main__":
    # generate_relu_codelets_for_2d_images()
    # codelet_string = generate_conv2d_bias_codelet_from_config(1, {"N_tiles": 1, "IC_tiles": 4, "OC_tiles": 4, "KH_tiles": 1, "KW_tiles": 1, "OH_tiles": 2, "OW_tiles": 1, "array_N": 16, "array_M": 16, "loop_order": [0, 1, 2, 3, 4, 5, 6]})
    # codelet_string = generate_conv2d_bias_relu_codelet_from_config(2, {"N_tiles": 1, "IC_tiles": 1, "OC_tiles": 1, "KH_tiles": 1, "KW_tiles": 1, "OH_tiles": 14, "OW_tiles": 7, "array_N": 16, "array_M": 16, "loop_order": [0, 1, 2, 3, 4, 5, 6]})
    # codelet_string = generate_gemm_bias_codelet_from_config({"N_tiles": 1, "M_tiles": 1, "P_tiles": 1, "array_N": 16, "array_M": 16, "loop_order": [0, 1, 2]})
    # codelet_string = generate_relu4d_codelet_from_config({"N_tiles": 1, "C_tiles": 1, "H_tiles": 1, "W_tiles": 1, "array_N": 16, "loop_order": [0, 1, 2, 3]})
    # print(codelet_string)
    # codelet = StealthCodelet(codelet_string)
    # compile("../codelets/examples/genesys/configs/benchmark_16x16.json", codelet, {"N": 1, "C": 64, "H": 16, "IH": 230, "W": 16, "IW": 230, "KH": 7, "KW": 7, "IC": 64, "OC": 64, "OH": 112, "OW": 112, "M": 16, "P": 16})
    operations = generate_simd_element_wise_operations(num_operations=1) 
    generate_simd_element_wise_operation_codelets_for_2d_images(operations, num_random_configs_per_exhaustive_config=1)
