import polymath as pm
import abc
from functools import partial
from typing import Any, Callable, Optional, Union
import dataclasses
from collections import OrderedDict
from lark import ParseTree, Lark, Token, Tree, indenter
from lark.visitors import Interpreter
from onnx import helper, numpy_helper
from codelets.examples.genesys import compile_genesys, load_config, DataGen
from codelets.templates.codelet_template import CodeletTemplate, DummyOp
from codelets.adl.graph import ArchitectureGraph
from codelets.examples.genesys import DTYPE_MAP, OP_DTYPES


# ==================== Codelet Parsing ==================== #
GRAMMAR = """
start: function

function: "def " NAME "(" parameters ")" ":" suite

parameters: parameter ("," parameter)*
parameter: NAME ":" type ["@" location]

suite: _NEWLINE _INDENT stmt+ _DEDENT
stmt: "assert" expr _NEWLINE -> assert
    | (one_split | multiple_split) _NEWLINE -> split
    | [NAME "="] NAME "(" call_args ")" _NEWLINE -> call
    | "for" NAME "in" "loop" "(" expr ["," expr] ")" ":" suite -> for_loop
    | "return" NAME _NEWLINE -> return_output

one_split: split_pair "=" split_call
multiple_split: "(" split_pair ")" ("," "(" split_pair ")")+ "=" split_call ("," split_call)+
split_pair: dimension "," dimension
split_call: "split" "(" dimension ")"

call_args: call_arg ("," call_arg)*
call_arg: indexed_variable | size | NAME | INT

indexed_variable: NAME "[" expr ("," expr)* "]"

size: "[" expr ("," expr)* "]"

type: NAME "[" dimensions "]" | NAME
dtype: NAME

dimensions: dimension ("," dimension)*
dimension: NAME | INT

location: NAME

expr: "(" expr ")"
    | expr "or" expr  -> or_expr
    | expr "and" expr -> and_expr
    | expr "==" expr    -> eq_expr
    | expr "!=" expr    -> ne_expr
    | expr "<" expr     -> lt_expr
    | expr "<=" expr    -> le_expr
    | expr ">" expr     -> gt_expr
    | expr ">=" expr    -> ge_expr
    | expr "*" expr     -> mul_expr
    | expr "/" expr     -> div_expr
    | expr "//" expr    -> floordiv_expr
    | expr "+" expr     -> add_expr
    | expr "-" expr     -> sub_expr
    | NAME
    | "True"
    | "False"
    | INT

NAME: CNAME | CNAME ("." CNAME)+

_NEWLINE: ( /\\r?\\n[ \\t]*/ | COMMENT )+
COMMENT: /#[^\\n]*/

%import common.LETTER
%import common.DIGIT
%import common.INT
%import common.CNAME
%import common.WS
%import common.WS_INLINE
%declare _INDENT _DEDENT
%ignore WS
%ignore WS_INLINE
%ignore COMMENT
"""

class TreeIndenter(indenter.Indenter):
    NL_type = '_NEWLINE'
    OPEN_PAREN_types = ["LPAR"]
    CLOSE_PAREN_types = ["RPAR"]
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 4


def parse_stealth_codelet(codelet_string: str) -> ParseTree:
    parser = Lark(GRAMMAR, start='start', parser='lalr', postlex=TreeIndenter())
    tree = parser.parse(codelet_string)
    print(tree.pretty())
    return tree


# ==================== Codelet Information Collection ==================== #
class CodeletParseError(Exception):
    pass


class CodeletError(Exception):
    pass


def _raise_codelet_parse_error(message, obj, input_text):    
    def get_line(lines, index):
        return lines[index] if 0 <= index < len(lines) else None
    
    if isinstance(obj, Token):
        line = input_text.splitlines()[obj.line - 1]
        indicator = ' ' * obj.column + '^'
        full_message = (f"{message} at line {obj.line}, column {obj.column}:\n"
                        f"{obj.line:4} | {line}\n"
                        f"     | {indicator}")
    elif isinstance(obj, Tree):
        min_leaf = None
        max_leaf = None
        
        # Recursive function to find the min and max positions in a tree
        def traverse_tree(tree):
            nonlocal min_leaf, max_leaf
            for child in tree.children:
                if isinstance(child, Token):
                    if min_leaf is None:
                        min_leaf = child
                    if max_leaf is None:
                        max_leaf = child
                    
                    if child.start_pos < min_leaf.start_pos:
                        min_leaf = child
                    if child.end_pos > max_leaf.end_pos:
                        max_leaf = child
                else:
                    traverse_tree(child)
        
        traverse_tree(obj)

        lines = input_text.splitlines()
        start_line = obj.children[0].line if isinstance(obj.children[0], Token) else min_leaf.line 
        end_line = obj.children[-1].line if isinstance(obj.children[-1], Token) else max_leaf.line
        
        num_initial_chars_to_remove = None
        for i in range(start_line - 3, end_line + 2):
            line = get_line(lines, i)
            if line is not None:
                if num_initial_chars_to_remove is None:
                    num_initial_chars_to_remove = len(line) - len(line.lstrip())
                else:
                    num_initial_chars_to_remove = min(num_initial_chars_to_remove, len(line) - len(line.lstrip()))
        
        context_lines = []
        for i in range(start_line - 3, end_line + 2):
            line = get_line(lines, i)
            if line is not None:
                line_str = line[num_initial_chars_to_remove:]
                context_lines.append(f"{i + 1:4} | {line_str}")
        context = "\n".join(line for line in context_lines if line is not None)
        
        if start_line == end_line:
            full_message = f"{message} at line {start_line}:\n\n{context}"
        else:
            full_message = f"{message} between lines {start_line} and {end_line}:\n\n{context}"
    else:
        full_message = f"{message}. Unable to determine exact location."
    
    raise CodeletParseError(full_message)


class StealthExpression(abc.ABC):
    @abc.abstractmethod
    def __str__(self) -> str:
        pass


@dataclasses.dataclass
class StealthVariableName(StealthExpression):
    name: str

    def __str__(self) -> str:
        if isinstance(self.name, StealthVariableName):
            print(self.name.name)
        return self.name


@dataclasses.dataclass
class StealthLiteral(StealthExpression):
    value: Union[int, bool]

    def __str__(self) -> str:
        return str(self.value)


@dataclasses.dataclass
class StealthBinaryExpression(StealthExpression):
    lhs: StealthExpression
    rhs: StealthExpression
    operation: str

    def __str__(self) -> str:
        return f"({self.lhs} {self.operation} {self.rhs})"


@dataclasses.dataclass
class StealthUnaryExpression(StealthExpression):
    operand: StealthExpression
    operation: str

    def __str__(self) -> str:
        return f"{self.operation}{self.operand}"


def evaluate_expression(expression: StealthExpression):
    if isinstance(expression, (StealthLiteral, StealthVariableName)):
        return expression
    elif isinstance(expression, StealthBinaryExpression):
        if isinstance(expression.lhs, StealthLiteral) and isinstance(expression.rhs, StealthLiteral):
            if expression.operation == "+":
                return StealthLiteral(expression.lhs.value + expression.rhs.value)
            elif expression.operation == "-":
                return StealthLiteral(expression.lhs.value - expression.rhs.value)
            elif expression.operation == "*":
                return StealthLiteral(expression.lhs.value * expression.rhs.value)
            elif expression.operation == "/":
                return StealthLiteral(expression.lhs.value / expression.rhs.value)
            elif expression.operation == "//":
                return StealthLiteral(expression.lhs.value // expression.rhs.value)
            elif expression.operation == "==":
                return StealthLiteral(expression.lhs.value == expression.rhs.value)
            elif expression.operation == "!=":
                return StealthLiteral(expression.lhs.value != expression.rhs.value)
            elif expression.operation == "<":
                return StealthLiteral(expression.lhs.value < expression.rhs.value)
            elif expression.operation == "<=":
                return StealthLiteral(expression.lhs.value <= expression.rhs.value)
            elif expression.operation == ">":
                return StealthLiteral(expression.lhs.value > expression.rhs.value)
            elif expression.operation == ">=":
                return StealthLiteral(expression.lhs.value >= expression.rhs.value)
            elif expression.operation == "and":
                return StealthLiteral(expression.lhs.value and expression.rhs.value)
            elif expression.operation == "or":
                return StealthLiteral(expression.lhs.value or expression.rhs.value)
            else:
                raise RuntimeError(f"Unknown binary operation: {expression.operation}")
        elif isinstance(expression.lhs, StealthLiteral) and expression.lhs.value == 0:
            if expression.operation in ["/", "//", "*"]:
                return StealthLiteral(0)
            elif expression.operation == "+":
                return expression.rhs
            elif expression.operation == "-":
                return StealthUnaryExpression(expression.rhs, "-")
            else:
                return expression
        elif isinstance(expression.rhs, StealthLiteral) and expression.rhs.value == 0:
            if expression.operation in ["/", "//"]:
                raise ZeroDivisionError
            elif expression.operation == "*":
                return StealthLiteral(0)
            elif expression.operation in ["+", "-"]:
                return expression.lhs
            else:
                return expression
        else:
            return expression
    elif isinstance(expression, StealthUnaryExpression):
        if isinstance(expression.operand, StealthLiteral):
            if expression.operation == "-":
                return StealthLiteral(-expression.operand.value)
            elif expression.operation == "not":
                return StealthLiteral(not expression.operand.value)
            else:
                raise RuntimeError(f"Unknown unary operation: {expression.operation}")
        else:
            return expression
    else:
        raise TypeError(f"Unknown expression type: {type(expression)}")


@dataclasses.dataclass
class StealthOperand:
    name: str
    shape: list[StealthExpression]
    dtype: str
    location: str

    def __str__(self) -> str:
        return f"{self.name}: {self.dtype}[{', '.join(map(str, self.shape))}] @ {self.location}"


@dataclasses.dataclass
class StealthParameter:
    name: str
    shape: list[StealthExpression]
    
    def __str__(self) -> str:
        return f"{self.name}: param[{', '.join(map(str, self.shape))}]"


class StealthStatement(abc.ABC):
    pass


@dataclasses.dataclass
class StealthAllocation(StealthStatement):
    operand_name: str
    size: list[StealthExpression]
    location: str
    dtype: str


@dataclasses.dataclass
class StealthLoad(StealthStatement):
    destination_operand_name: str
    source_operand_name: str
    source_operand_offset: list[StealthExpression]
    size: list[StealthExpression]
    location: str


@dataclasses.dataclass
class StealthStore(StealthStatement):
    destination_operand_name: str
    destination_operand_offset: list[StealthExpression]
    source_operand_name: str


@dataclasses.dataclass
class StealthCompute(StealthStatement):
    destination_operand_name: str
    operation_name: str
    operands: list[Union[str, StealthExpression]]
    location: str


@dataclasses.dataclass
class StealthLoop(StealthStatement):
    loop_index_variable_name: str
    number_of_iterations: StealthExpression
    stride: StealthExpression
    body: list[StealthStatement]


@dataclasses.dataclass
class StealthIndex:
    name: str
    end: StealthExpression
    stride: StealthExpression

    def __str__(self) -> str:
        return f"{self.name}(start=0, high={self.end}, stride={self.stride})"


@dataclasses.dataclass
class ComputeOperationGroup:
    operations: list[StealthCompute]
    transfer_and_compute_operations: list[Union[StealthLoad, StealthStore, StealthCompute]]
    loop_index_variables: list[StealthIndex]

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
        if all(o.location == "SIMD" for o in self.operaitons):
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
        for transfer_and_compute_operations in self.transfer_and_compute_operations:
            for operation in transfer_and_compute_operations:
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
        "relu": "RELU",
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
        self._loop_indices = []
        self._statements = []
        self._current_loop_stack = []
    
    def __str__(self) -> str:
        operation_name: str = self._operation_name or "unknown"
        header: str = f"def {operation_name}({', '.join(list(map(str, self._inputs)) + list(map(str, self._params)))}):\n"
        return header + self._print_body(self._statements) + "\n" + f"\treturn {', '.join(map(str, self._outputs))}"
    
    def _print_body(self, statements: list[StealthStatement], indentation_level: int = 1) -> str:
        ret: str = ""
        TABS: str = "\t" * indentation_level
        for statement in statements:
            if isinstance(statement, StealthAllocation):
                ret += TABS + f"{statement.operand_name} = alloc([{', '.join(map(str, statement.size))}], {statement.location}, {statement.dtype})\n"
            elif isinstance(statement, StealthLoad):
                ret += TABS + f"{statement.destination_operand_name} = load({statement.source_operand_name}[{', '.join(map(str, statement.source_operand_offset))}], [{', '.join(map(str, statement.size))}], {statement.location})\n"
            elif isinstance(statement, StealthStore):
                ret += TABS + f"store({statement.destination_operand_name}[{', '.join(map(str, statement.destination_operand_offset))}], {statement.source_operand_name})\n"
            elif isinstance(statement, StealthCompute):
                ret += TABS + f"{statement.destination_operand_name} = {statement.operation_name}({', '.join(map(str, statement.operands))}, {statement.location})\n"
            elif isinstance(statement, StealthLoop):
                ret += TABS + f"for {statement.loop_index_variable_name} in loop({statement.number_of_iterations}, {statement.stride}):\n"
                ret += self._print_body(statement.body, indentation_level + 1)
            else:
                raise TypeError(f"Unknown statement type: {type(statement)}")
        return ret
    
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
                compute_groups[-1].append(statement)
        
        for statement in self._statements:
            helper1(statement)
        if len(compute_groups[-1]) == 0:
            compute_groups.pop()
        
        transfer_and_compute_operations: list[list[list[Union[StealthLoad, StealthStore, StealthCompute]]]] = [[[]] * len(compute_operations) for compute_operations in compute_groups]
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
                transfer_and_compute_operations[compute_group_index][compute_operation_index].append(statement)
                if isinstance(statement, StealthStore):
                    ready_to_move_on_to_next_operation = True
                if isinstance(statement, StealthLoad) and ready_to_move_on_to_next_operation:
                    compute_operation_index += 1
                    if compute_operation_index >= len(compute_groups[compute_group_index]):
                        compute_group_index += 1
                        compute_operation_index = 0
        for statement in self._statements:
            helper2(statement)
        
        loop_index_variables: list[list[StealthIndex]] = [[]] * len(compute_groups)
        for compute_group_transfer_and_compute_operations, compute_operation_loop_index_variables in zip(transfer_and_compute_operations, loop_index_variables):
            def helper3(statement: StealthStatement):
                nonlocal compute_operation_loop_index_variables
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
                            if statement.loop_index_variable_name in loop_index_variables_in_offset and statement.loop_index_variable_name not in map(lambda l: l.name, compute_operation_loop_index_variables):
                                compute_operation_loop_index_variables.append(StealthIndex(statement.loop_index_variable_name, statement.number_of_iterations, statement.stride))

                    for loop_statement in statement.body:
                        helper3(loop_statement) 
        
            for statement in self._statements:
                helper3(statement)

        # Flatten transfer and compute operations
        new_transfer_and_compute_operations = []
        for compute_group_transfer_and_compute_operations in transfer_and_compute_operations:
            new_transfer_and_compute_operations.append([])
            for compute_operation_transfer_and_compute_operations in compute_group_transfer_and_compute_operations:
                new_transfer_and_compute_operations[-1].extend(compute_operation_transfer_and_compute_operations)
        
        return [ComputeOperationGroup(compute_group, compute_group_transfer_and_compute_operations, compute_group_loop_index_variables) for compute_group, compute_group_transfer_and_compute_operations, compute_group_loop_index_variables in zip(compute_groups, new_transfer_and_compute_operations, loop_index_variables)]
    
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
        output_operand_name_child = tree.children[0]
        output_operand_name: str = self.get_name(output_operand_name_child, "output operand name")
        self.check_for_variable_name_defined(output_operand_name, output_operand_name_child)
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
        elif operation_name in ["add", "sub", "mul", "div", "max", "min"]:
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
            self.raise_codelet_parse_error(f"Expected 3 arguments for alloc operation but instead got {len(operation_args_child.children)}", tree)
        assert all(isinstance(child, Tree) for child in operation_args_child.children)
        if any(child.data != "call_arg" and len(child.children) == 1 for child in operation_args_child.children):
            self.raise_codelet_parse_error(f"Expected all arguments to be call arguments but instead got {operation_args_child.children}", tree)

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
            self.raise_codelet_parse_error(f"Expected 3 arguments for load operation but instead got {len(operation_args_child.children)}", tree)
        assert all(isinstance(child, Tree) for child in operation_args_child.children)
        if any(child.data != "call_arg" or len(child.children) != 1 for child in operation_args_child.children):
            self.raise_codelet_parse_error(f"Expected all arguments to be call arguments but instead got {operation_args_child.children}", tree)
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
            self.raise_codelet_parse_error(f"Expected 2 arguments for store operation but instead got {len(operation_args_child.children)}", tree)
        assert all(isinstance(child, Tree) for child in operation_args_child.children)
        if any(child.data != "call_arg" or len(child.children) != 1 for child in operation_args_child.children):
            self.raise_codelet_parse_error(f"Expected all arguments to be call arguments but instead got {operation_args_child.children}", tree)
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
                self.raise_codelet_parse_error(f"Expected all arguments to be call arguments but instead got {operation_args_child.children}", tree) 
            expression_child = child.children[0]
            expression = self.get_stealth_expression_from_tree(expression_child)
            if self.is_expression_constant(expression):
                arguments.append(expression)
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
                                        "relu", "leaky_relu", "sigmoid", "tanh",
                                        "exp", "sqrt", "inv_sqrt", "log2",
                                        "equal", "neq", "gt", "gte", "lt", "lte"]:
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
            if isinstance(evaluate_expression(statement.number_of_iterations), StealthLiteral):
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
        setattr(self, "_number_of_stores_to_output", 0)
        for statement in self._statements:
            self._check_codelet_has_only_one_store_to_output(statement)
        if self._number_of_stores_to_output == 0:
            raise CodeletError(f"Codelet {self._operation_name} has no store to the output operand")
        delattr(self, "_number_of_stores_to_output")

    def _check_codelet_has_only_one_store_to_output(self, statement: StealthStatement):
        if isinstance(statement, StealthLoop):
            for loop_statement in statement.body:
                self._check_codelet_has_only_one_store_to_output(loop_statement)
        elif isinstance(statement, StealthStore):
            if statement.destination_operand_name in map(lambda o: o.name, self._outputs):
                if self._number_of_stores_to_output == 1:
                    raise CodeletError(f"Codelet {self._operation_name} has more than one store to the output operand (second store is {statement})")
                else:
                    self._number_of_stores_to_output += 1

    
def collect_codelet_information_from_parse_tree(tree: ParseTree, codelet_string: str) -> CodeletInformationCollector:
    collector = CodeletInformationCollector(codelet_string)
    collector.visit(tree)
    collector.check_codelet()
    print(collector)
    return collector


# ==================== Codelet Template Creator ==================== #
# class UniqueNameGenerator:
#     _NAME_ID: dict[str, int] = defaultdict(lambda: 0)

#     @staticmethod
#     def get_unique_name(name: str) -> str:
#         name_id = UniqueNameGenerator._NAME_ID[name]
#         UniqueNameGenerator._NAME_ID[name] += 1
#         return f"{name}_{name_id}"


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
        if len(self._staged_tile_loads) != len(self._staged_inputs):
            raise RuntimeError(f"Expected {len(self._staged_tile_loads)} loads but instead got {len(self._staged_inputs)}")
        if len(self._staged_element_loads) != len(self._staged_inputs):
            raise RuntimeError(f"Expected {len(self._staged_element_loads)} loads but instead got {len(self._staged_inputs)}")
        if len(self._staged_tile_stores) != len(self._staged_outputs):
            raise RuntimeError(f"Expected {len(self._staged_tile_stores)} stores but instead got {len(self._staged_outputs)}")
        if len(self._staged_element_stores) != len(self._staged_outputs):
            raise RuntimeError(f"Expected {len(self._staged_element_stores)} stores but instead got {len(self._staged_outputs)}")
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
        cdlt.compute(self.compute.operation, list(map(lambda t: t[0][t[1]], self.compute.inputs)), list(map(lambda t: t[0][t[1]], self.compute.outputs)), self.compute.location)
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
            def dummy_op_creation_helper(collector_operands, node_operands):
                for stealth_codelet_operand, node_operand in zip(collector_operands, node_operands):
                    for stealth_codelet_operand_dimension, node_operand_dimension in zip(stealth_codelet_operand.shape, node_operand.shape):
                        dimension_name = input_output_dimension_to_name(stealth_codelet_operand_dimension)
                        if dimension_name not in dimension_dummy_ops:
                            dimension_dummy_ops[dimension_name] = cdlt.dummy_op(dimension_name, node_operand_dimension)
            dummy_op_creation_helper(collector.get_input_operands(), cdlt.node.inputs)
            dummy_op_creation_helper(collector.get_output_operands(), cdlt.node.outputs)

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
            
            # Create intermediate operands for complicated codelets
            compute_operation_groups: list[ComputeOperationGroup] = collector.get_compute_operation_groups()
            # compute_operation_group_output_operand_information: list[list] = []
            # most_recent_template_operand_information = None
            # for compute_operation_group in compute_operation_groups:
            #     compute_operation_group_output_operand_information.append([])
            #     for transfer_or_compute_operation in compute_operation_group.transfer_and_compute_operations:
            #         if isinstance(transfer_or_compute_operation, StealthCompute):
            #             compute_operation_output_operand: StealthOperand = collector._operands[transfer_or_compute_operation.destination_operand_name]
            #             output_shape: list[StealthExpression] = compute_operation_output_operand.shape
            #             if any(not isinstance(d, (StealthVariableName, StealthLiteral)) for d in output_shape):
            #                 raise RuntimeError(f"Expected all output operand dimensions to be either StealthVariableName or StealthLiteral but instead got {output_shape}")
            #             output_shape = [dimension_dummy_ops[input_output_dimension_to_name(dimension)] for dimension in output_shape]
            #             template_operand_information = tuple(compute_operation_output_operand.name, output_shape, None)
            #             most_recent_template_operand_information = template_operand_information
            #             compute_operation_group_output_operand_information[-1].append(template_operand_information)
            #         elif isinstance(transfer_or_compute_operation, StealthStore):
            #             if most_recent_template_operand_information[0] == transfer_or_compute_operation.source_operand_name:
            #                 most_recent_template_operand_information[-1] = collector._operands[transfer_or_compute_operation.destination_operand_name].location
            # compute_operation_group_output_operands: list = []
            # operand_name_that_stores_to_output: str = collector.get_compute_output_operand_that_stores_to_output_operand().name
            # for template_operand_information in compute_operation_group_output_operand_information:
            #     if operand_name_that_stores_to_output == template_operand_information[0]:
            #         compute_operation_group_output_operands.append(output_operands[0])  # TODO: If there are more than one outputs, this doesn't work
            #     else:
            #         temp_operand = cdlt.create_operand_template(template_operand_information[0], OP_DTYPES, template_operand_information[1], default_dtype=acc_dtype)
            #         temp_operand.start_location = template_operand_information[2]
            #         cdlt.add_temp_operand(temp_operand)
            #         compute_operation_group_output_operands.append(temp_operand)
            # assert len(compute_operation_group_output_operands) == sum(map(lambda g: len(g.operations), compute_operation_groups)), f"The list of output operands for each compute operation does not match the length of the total number of compute operations"
            
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
            
            def simd_start_config(_cdlt, immediates: list[tuple[str, int]]) -> None:
                immediate_dummy_ops = []
                for immediate_name, immediate_value in immediates:
                    immediate_dummy_ops.append(_cdlt.dummy_op(immediate_name, immediate_value))
                    temp_operand = _cdlt.create_temp_operand([SIMD_SIZE], "IMM", name=immediate_name)
                    name_to_operand[str(immediate_value)] = temp_operand
                _cdlt.configure("start", "simd")
                for immediate_dummy_op in immediate_dummy_ops:
                    _cdlt.configure("start", immediate_value=immediate_dummy_op)
                    
            def simd_end_config(_cdlt) -> None:
                _cdlt.configure("end", "simd")

            compute_operation_index: int = 0 
            for compute_operation_group in compute_operation_groups:
                if compute_operation_group.is_systolic_array_operation:
                    config_functions = (systolic_array_start_config, systolic_array_end_config)
                elif compute_operation_group.is_simd_operation:
                    config_functions = (partial(simd_start_config, immediates=[]), simd_end_config)
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
                        # name_to_operand[transfer_or_compute_operation.destination_operand_name] = compute_operation_group_output_operands[compute_operation_index]
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
        self._codelet_template = create_codelet_template_from_codelet_information_collector(self._information_collector)
 

# ==================== PolyMath Node Generation ==================== #
def create_dummy_polymath_node_from_codelet(codelet: StealthCodelet) -> pm.Graph:
    with pm.Node(name=codelet._information_collector.get_operation_name()) as graph:
        parameters = {}
        for operand in codelet._information_collector._inputs + codelet._information_collector._outputs:
            for dimension in operand.shape:
                if isinstance(dimension, StealthVariableName) and dimension.name not in parameters:
                    parameters[dimension.name] = pm.parameter(name=dimension.name)
        inputs = {}
        for operand in codelet._information_collector._inputs:
            inputs[operand.name] = pm.input(name=operand.name, shape=tuple(parameters[s.name] if isinstance(s, StealthVariableName) else s.value for s in operand.shape))
        outputs = {}
        for output_operand in codelet._information_collector._outputs:
            outputs[output_operand.name] = pm.output(name=output_operand.name, shape=tuple(parameters[s.name] if isinstance(s, StealthVariableName) else s.value for s in output_operand.shape))
    return graph


# ==================== Compilation ==================== #
def compile(config_path: str, layer: StealthCodelet) -> None:
    graph = create_dummy_polymath_node_from_codelet(layer)
    cfg = load_config(config_path)
    program = compile_genesys(
        model_name=layer._information_collector.get_operation_name(),
        graph=graph,
        genesys_cfg=cfg,
        custom_codelets={layer._information_collector.get_operation_name(): layer._codelet_template},
        print_config=False,
    )
    program.compile(finalize=True)
    print(program.codelets)

    sys_array_size = cfg['ARRAY_M']
    dgen = DataGen(program,
                    single_codelets=False,
                    shared_datagen=False,
                    dir_ext=f"benchmark{sys_array_size}x{sys_array_size}",
                    identifier="stealth",
                    generate_data=False,
                    verbose=False,
                    out_path=f"/stealth_compilation_output",
                    store_whole_program=False)
    dgen.generate()


if __name__ == "__main__":
    codelet_string = """def conv2d_bias(x: i8[N, IH, IW, C] @ DRAM, w: i8[KH, KW, IC, OC] @ DRAM, bias: i32[OC] @ DRAM, sx: param, sy: param, pad: param, OH: param, OW: param):
    o = alloc([N, OH, OW, OC], DRAM, i32)
    for oc in loop(2, OC // 2):
        b1 = load(bias[oc], [OC // 2], BBUF)
        for n in loop(1, N // 1):
            for ic in loop(4, IC // 4):
                for kh in loop(1, KH // 1):
                    for kw in loop(1, KW // 1):
                        w1 = load(w[kh, kw, ic, oc], [KH // 1, KW // 1, IC // 4, OC // 2], WBUF)
                        for oh in loop(14, OH // 14):
                            for ow in loop(7, OW // 7):
                                x1 = load(x[n, kh + 2 * oh, kw + 2 * ow, ic],
                                          [N // 1, OH // 14, OW // 7, OC // 2], IBUF)
                                o1 = load(o[n, oh, ow, oc], [N // 1, OH // 14, OW // 7, OC // 2], OBUF)
                                for oc1 in loop(OC // 2, 16):
                                    b2 = load(b1[oc1], [1, 16], PE_ARRAY)
                                    for n1 in loop(N // 1):
                                        for ic1 in loop(IC // 4, 16):
                                            for kh1 in loop(KH // 1):
                                                for kw1 in loop(KW // 1):
                                                    w2 = load(w1[kh1, kw1, ic1, oc1],
                                                              [16, 16],
                                                              PE_ARRAY)
                                                    for oh1 in loop(OH // 14):
                                                        for ow1 in loop(OW // 7):
                                                            x2 = load(
                                                                x1[n1, kh1 + 2 * oh1, kw1 + 2 * ow1, ic1],
                                                                [1, 16], PE_ARRAY)
                                                            o2 = load(o1[n1, oh1, ow1, oc1], [16, 1],
                                                                      PE_ARRAY)
                                                            # We need to load the pre-existing obuf data, then generate new data
                                                            o3 = mvmul_bias(x2, w2, b2, o2, PE_ARRAY)
                                                            store(o1[n1, oh1, ow1, oc1], o3)
                                store(o[n, oh, ow, oc], o1)
    return o
"""
    codelet = StealthCodelet(codelet_string)
    # print(codelet._information_collector.get_compute_operation_groups()[0])
    compile("../codelets/examples/genesys/configs/benchmark_16x16.json", codelet)
