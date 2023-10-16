import abc
import dataclasses
from typing import Union
from .expression.core import StealthExpression


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
    end: StealthExpression
    stride: StealthExpression
    body: list[StealthStatement]


@dataclasses.dataclass
class StealthIndex:
    name: str
    number_of_iterations: StealthExpression
    stride: StealthExpression

    def __str__(self) -> str:
        return f"{self.name}(number_of_iterations={self.number_of_iterations}, stride={self.stride})"


class StealthCodelet:
    _operation_name: str
    _operands: dict[str, StealthOperand]
    _inputs: list[StealthOperand]
    _outputs: list[StealthOperand]
    _params: list[StealthParameter]
    _immediates: dict[str, StealthExpression]
    _loop_indices: list[StealthIndex]
    _statements: list[StealthStatement]

    def __init__(self, operation_name: str, operands: dict[str, StealthOperand], inputs: list[StealthOperand], outputs: list[StealthOperand], params: list[StealthParameter], immediates: dict[str, StealthExpression], loop_indices: list[StealthIndex], statements: list[StealthStatement]) -> None:
        self._operation_name = operation_name
        self._operands = operands
        self._inputs = inputs
        self._outputs = outputs
        self._params = params
        self._immediates = immediates
        self._loop_indices = loop_indices
        self._statements = statements
    
    @property
    def operation_name(self) -> str:
        return self._operation_name
    
    @property
    def inputs(self) -> list[StealthOperand]:
        return self._inputs.copy()
    
    @property
    def outputs(self) -> list[StealthOperand]:
        return self._outputs.copy()
    
    @property
    def params(self) -> list[StealthParameter]:
        return self._params.copy()
    
    def get_operand(self, operand_name: str) -> StealthOperand:
        return self._operands[operand_name]
    
    def get_immediate(self, immediate_name: str) -> StealthExpression:
        return self._immediates[immediate_name]

    def __str__(self) -> str:
        header: str = f"def {self._operation_name}({', '.join(list(map(str, self._inputs)) + list(map(str, self._params)))}):\n"
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
                ret += TABS + f"for {statement.loop_index_variable_name} in loop({statement.end}, {statement.stride}):\n"
                ret += self._print_body(statement.body, indentation_level + 1)
            else:
                raise TypeError(f"Unknown statement type: {type(statement)}")
        return ret
