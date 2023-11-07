from stealth.stealth_codelet.core import StealthAllocation, StealthCompute, StealthIndex, StealthOperand, StealthParameter, StealthStatement
from .core import *
from .expression import evaluate_expression
from .visitor import StealthCodeletTransformer


class VariableSubstituter(StealthCodeletTransformer):
    _name_to_value_map: dict[str, int]

    def __init__(self, name_to_value_map: dict[str, int]):
        self._name_to_value_map = name_to_value_map.copy()

    def transform(self, codelet: StealthCodelet) -> StealthCodelet:
        new_operands: dict[str, StealthOperand] = {}
        for name, operand in codelet._operands.items():
            new_name, new_operand = self.transform_operand(name, operand)
            new_operands[new_name] = new_operand
        new_inputs: list[StealthOperand] = [self.transform_operand(operand.name, operand)[1] for operand in codelet._inputs]
        new_outputs: list[StealthOperand] = [self.transform_operand(operand.name, operand)[1] for operand in codelet._outputs]
        new_parameters: list[StealthParameter] = list(filter(lambda p: p.name not in self._name_to_value_map, codelet._input_params))
        new_immediates: dict[str, int] = {}
        for name, value in codelet._immediates.items():
            new_name, new_value = self.transform_immediate(name, value)
            new_immediates[new_name] = new_value
        new_loop_indices: list[StealthIndex] = [self.transform_loop_index(index) for index in codelet._loop_indices]
        new_statements: list[StealthStatement] = [self.transform_statement(statement) for statement in codelet._statements]
        return StealthCodelet(
            codelet.operation_name,
            new_operands,
            new_inputs,
            new_outputs,
            new_parameters,
            new_immediates,
            new_loop_indices,
            new_statements
        )
    
    def transform_operand(self, name: str, operand: StealthOperand) -> tuple[str, StealthOperand]:
        return name, StealthOperand(operand.name, tuple(evaluate_expression(dim, self._name_to_value_map) for dim in operand.shape), operand.dtype, operand.location)
    
    def transform_loop_index(self, index: StealthIndex) -> StealthIndex:
        return StealthIndex(index.name, evaluate_expression(index.number_of_iterations, self._name_to_value_map), evaluate_expression(index.stride, self._name_to_value_map))
    
    def transform_allocation(self, statement: StealthAllocation) -> StealthAllocation:
        return StealthAllocation(statement.operand_name, tuple(evaluate_expression(dim, self._name_to_value_map) for dim in statement.size), statement.location, statement.dtype)

    def transform_load(self, statement: StealthLoad) -> StealthLoad:
        return StealthLoad(statement.destination_operand_name, statement.source_operand_name, [evaluate_expression(dim, self._name_to_value_map) for dim in statement.source_operand_offset], [evaluate_expression(dim, self._name_to_value_map) for dim in statement.size], statement.location)
    
    def transform_store(self, statement: StealthStore) -> StealthStore:
        return StealthStore(statement.destination_operand_name, tuple(evaluate_expression(dim, self._name_to_value_map) for dim in statement.destination_operand_offset), statement.source_operand_name)
    
    def transform_loop(self, statement: StealthLoop) -> StealthLoop:
        return StealthLoop(
            statement.loop_index_variable_name,
            evaluate_expression(statement.number_of_iterations, self._name_to_value_map),
            evaluate_expression(statement.stride, self._name_to_value_map),
            [self.transform_statement(loop_statement) for loop_statement in statement.body],
        )
    
    def transform_compute(self, statement: StealthCompute) -> StealthCompute:
        return StealthCompute(
            statement.destination_operand_name,
            statement.operation_name,
            tuple(evaluate_expression(expression, self._name_to_value_map) if isinstance(expression, StealthExpression) else expression for expression in statement.operands),
            statement.location
        )

def substitute_variables(codelet: StealthCodelet, name_to_value_map: dict[str, int]) -> StealthCodelet:
    return VariableSubstituter(name_to_value_map).transform(codelet)
