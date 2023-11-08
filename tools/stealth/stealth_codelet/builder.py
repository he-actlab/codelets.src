from typing import Optional
from lark import Tree, Token, ParseTree
from lark.visitors import Interpreter
from .error import CodeletError, raise_codelet_parse_error
from .expression import *
from .core import *
from stealth.stealth_codelet.converter.utils import UniqueNameGenerator


class StealthCodeletBuilder(Interpreter):
    _COMPUTE_OUTPUT_DTYPE: dict[str, str] = {
        "PE_ARRAY": "i32",
        "SIMD": "i32"
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
    _current_loop_stack: list[StealthLoop]
    _imm_unique_name_generator: UniqueNameGenerator

    _operation_name: Optional[str]
    _used_var_names: set[str]
    _operands: dict[str, StealthOperand]
    _inputs: list[StealthOperand]
    _outputs: list[StealthOperand]
    _params: list[StealthParameter]
    _immediates: dict[str, StealthExpression]
    _loop_indices: list[StealthIndex]
    _statements: list[StealthStatement]

    def __init__(self, codelet_string: str) -> None:
        super().__init__()
        self._codelet_string = codelet_string
        self._current_loop_stack = []
        self._imm_unique_name_generator = UniqueNameGenerator()

        self._operation_name = None
        self._used_var_names = set()
        self._operands = {}
        self._inputs = []
        self._outputs = []
        self._params = []
        self._immediates = {}
        self._loop_indices = []
        self._statements = []
    
    def build_codelet(self, tree: ParseTree) -> StealthCodelet:
        self.reset()
        self.visit(tree)
        return StealthCodelet(
            self._operation_name,
            self._operands.copy(),
            self._inputs.copy(),
            self._outputs.copy(),
            self._params.copy(),
            self._immediates.copy(),
            self._loop_indices.copy(),
            self._statements.copy()
        )
    
    def reset(self) -> None:
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
        new_param_dimensions = tuple(new_param_dimensions)
        
        if not isinstance(param_name, str):
            raise TypeError(f"Expected parameter name to be a string but instead got {type(param_name)}")
        if not isinstance(new_param_dimensions, tuple):
            raise TypeError(f"Expected parameter dimensions to be a tuple but instead got {type(param_dimensions)}")
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
        new_operand_dimensions = tuple(new_operand_dimensions)

        if not isinstance(operand_name, str):
            raise TypeError(f"Expected operand name to be a string but instead got {type(operand_name)}")
        if not isinstance(new_operand_dimensions, tuple):
            raise TypeError(f"Expected operand dimensions to be a tuple but instead got {type(operand_dimensions)}")
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
    
    def add_loop_index(self, loop_index_name: str, number_of_iterations: StealthExpression, stride: StealthExpression) -> StealthIndex:
        if not isinstance(loop_index_name, str):
            raise TypeError(f"Expected loop index name to be a string but instead got {type(loop_index_name)}")
        if not isinstance(number_of_iterations, StealthExpression):
            raise TypeError(f"Expected number_of_iterations to be an expression but instead got {type(number_of_iterations)}")
        if not isinstance(stride, StealthExpression):
            raise TypeError(f"Expected stride to be an expression but instead got {type(stride)}")
        
        if loop_index_name in self._used_var_names:
            raise RuntimeError(f"Loop index name {loop_index_name} was not checked for duplicates before being added")

        loop_index = StealthIndex(loop_index_name, number_of_iterations, stride)
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
        number_of_iterations_child = tree.children[1]
        stride_child = tree.children[2]
        body_child = tree.children[3]

        loop_index_variable_name: str = self.get_name(loop_index_variable_name_child, "loop index variable name")
        number_of_iterations: StealthExpression = self.get_stealth_expression_from_tree(number_of_iterations_child)
        stride: StealthExpression = self.get_stealth_expression_from_tree(stride_child) if stride_child else StealthLiteral(1)

        self.check_for_variable_name_is_not_defined(loop_index_variable_name, loop_index_variable_name_child)
        self.add_loop_index(loop_index_variable_name, number_of_iterations, stride)
        self.add_statement(StealthLoop(loop_index_variable_name, number_of_iterations, stride, []))
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
    
    def raise_codelet_parse_error(self, message: str, obj):
        raise_codelet_parse_error(message, obj, self._codelet_string) 
    
    def check_for_variable_name_defined(self, variable_name: str, tree: ParseTree) -> None:
        if variable_name not in self._used_var_names:
            self.raise_codelet_parse_error(f"A variable with name \"{variable_name}\" does not exist.", tree)

    def check_for_variable_name_is_not_defined(self, variable_name: str, tree: ParseTree) -> None:
        if variable_name in self._used_var_names:
            self.raise_codelet_parse_error(f"A variable with name \"{variable_name}\" already exists.", tree)
    
    def check_compute_arguments(self, operation_name: str, compute_arguments: list[str], location: str) -> None:
        if operation_name == "mvmul":
            if location != "PE_ARRAY":
                raise CodeletError("Matrix-vector multiplication operation \"mvmul\" is only supported on the PE_ARRAY")
            if len(compute_arguments) != 3:
                raise CodeletError(f"Expected 4 arguments for operation \"mvmul\" (input, weight, intermediate output) but instead got {len(compute_arguments)}")
            
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
 

            if compute_arguments[0] not in self._immediates:
                operand1 = self._operands[compute_arguments[0]]
                if len(operand1.shape) != 1:
                    raise CodeletError(f"Expected first argument of operation \"{operation_name}\" to be a 1D tensor but instead got a {len(operand1.shape)}D tensor")
            if compute_arguments[1] not in self._immediates:
                operand2 = self._operands[compute_arguments[1]]
                if len(operand2.shape) != 1:
                    raise CodeletError(f"Expected second argument of operation \"{operation_name}\" to be a 1D tensor but instead got a {len(operand2.shape)}D tensor")
            if compute_arguments[0] not in self._immediates and compute_arguments[1] not in self._immediates:
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
    
    def get_alloc_args_from_operation_args_child(self, operation_args_child) -> tuple[tuple[StealthExpression], str, str]:
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

        size: tuple[StealthExpression] = tuple(self.get_stealth_expression_from_tree(child) for child in size_child.children)
        location: str = self.get_name(location_child, name_type="location")
        dtype: str = self.get_name(dtype_child, name_type="dtype")

        return size, location, dtype
    
    def get_load_args_from_operation_args_child(self, operation_args_child) -> tuple[str, tuple[StealthExpression], tuple[StealthExpression], str]:
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
        source_operand_offset: tuple[StealthExpression] = tuple(self.get_stealth_expression_from_tree(child) for child in source_operand_offset_children)
        size: tuple[StealthExpression] = tuple(self.get_stealth_expression_from_tree(child) for child in size_child.children)
        location: str = self.get_name(location_child, name_type="location")

        return source_operand_name, source_operand_offset, size, location

    def get_store_args_from_operation_args_child(self, operation_args_child) -> tuple[str, tuple[StealthExpression], str]:
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
        destination_operand_offset: tuple[StealthExpression] = tuple(self.get_stealth_expression_from_tree(child) for child in destination_operand_offset_children)
        source_operand_name: str = self.get_name(source_operand_child, name_type="source operand")
        self.check_for_variable_name_defined(source_operand_name, source_operand_child)

        return destination_operand_name, destination_operand_offset, source_operand_name

    def get_compute_args_from_operation_args_child(self, operation_args_child) -> tuple[tuple[str], str]:
        assert all(isinstance(child, Tree) for child in operation_args_child.children)

        location_child = operation_args_child.children[-1]
        location = self.get_name(location_child, name_type="location")

        arguments: list[str] = []
        for child in operation_args_child.children[:-1]:
            if len(child.children) != 1:
                self.raise_codelet_parse_error(f"Expected all arguments to be call arguments but instead got {operation_args_child.children}", operation_args_child) 
            expression_child = child.children[0]
            expression = self.get_stealth_expression_from_tree(expression_child)
            if is_expression_constant(expression):
                if location == "PE_ARRAY":
                    raise CodeletError(f"Constant expressions are not supported on the PE_ARRAY")
                value = evaluate_expression(expression, {})
                immediate_name = self._imm_unique_name_generator.get_unique_name("immediate")
                self._immediates[immediate_name] = value
                arguments.append(immediate_name)
            elif isinstance(expression, StealthVariableName) and expression.name in self._operands:
                arguments.append(expression.name)
            else:
                self.raise_codelet_parse_error(f"Expected all arguments to be either a constant or an operand but instead got {expression}", expression_child)
         
        return tuple(arguments), location

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
            if compute_arguments[0] in self._operands:
                return self._operands[compute_arguments[0]].shape
            elif len(compute_arguments) > 1 and compute_arguments[1] in self._operands:
                return self._operands[compute_arguments[1]].shape
            else:
                raise RuntimeError(f"Something went wrong... Expected at least one operand to be an operand but instead got {compute_arguments}")
        else:
            raise NotImplementedError(f"{compute_operation_name} is not implemented yet")
    

def build_codelet_from_parse_tree(tree: ParseTree, codelet_string: str, verbose: bool = False) -> StealthCodelet:
    if verbose:
        print("Beginning building codelet...")
    builder = StealthCodeletBuilder(codelet_string)
    codelet: StealthCodelet = builder.build_codelet(tree)
    if verbose:
        print("Codelet built successfully!")
        print(f"Codelet:\n{codelet}")
    return codelet
