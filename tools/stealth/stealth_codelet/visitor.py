from .core import *


class StealthCodeletVisitor:
    def visit(self, codelet: StealthCodelet) -> None:
        for operand in codelet._operands.values():
            self.visit_operand(operand)
        for input_operand in codelet._inputs:
            self.visit_operand(input_operand)
        for output_operand in codelet._outputs:
            self.visit_operand(output_operand)
        for parameter in codelet._input_params:
            self.visit_parameter(parameter)
        for immediate_name, immediate_value in codelet._immediates.items():
            self.visit_immediate(immediate_name, immediate_value)
        for loop_index in codelet._loop_indices:
            self.visit_loop_index(loop_index)
        for statement in codelet._statements:
            self.visit_statement(statement)

    def visit_operand(self, operand: StealthOperand) -> None:
        pass

    def visit_parameter(self, parameter: StealthParameter) -> None:
        pass

    def visit_immediate(self, name: str, value: int) -> None:
        pass

    def visit_loop_index(self, index: StealthIndex) -> None:
        pass

    def visit_statement(self, statement: StealthStatement) -> None:
        if isinstance(statement, StealthAllocation):
            self.visit_allocation(statement)
        elif isinstance(statement, StealthLoad):
            self.visit_load(statement)
        elif isinstance(statement, StealthStore):
            self.visit_store(statement)
        elif isinstance(statement, StealthLoop):
            self.visit_loop(statement)
        elif isinstance(statement, StealthCompute):
            self.visit_compute(statement)
        else:
            raise RuntimeError(f"Unknown statement type: {statement}")
    
    def visit_allocation(self, statement: StealthAllocation) -> None:
        pass

    def visit_load(self, statement: StealthLoad) -> None:
        pass

    def visit_store(self, statement: StealthStore) -> None:
        pass

    def visit_loop(self, statement: StealthLoop) -> None:
        for loop_statement in statement.body:
            self.visit_statement(loop_statement)
    
    def visit_compute(self, statement: StealthCompute) -> None:
        pass


class StealthCodeletTransformer:
    def transform(self, codelet: StealthCodelet) -> StealthCodelet:
        new_operands: dict[str, StealthOperand] = {}
        for name, operand in codelet._operands.items():
            new_name, new_operand = self.transform_operand(name, operand)
            new_operands[new_name] = new_operand
        new_inputs: list[StealthOperand] = [self.transform_operand(operand.name, operand)[1] for operand in codelet._inputs]
        new_outputs: list[StealthOperand] = [self.transform_operand(operand.name, operand)[1] for operand in codelet._outputs]
        new_parameters: list[StealthParameter] = [self.transform_parameter(parameter) for parameter in codelet._input_params]
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
        return name, StealthOperand(operand.name, operand.shape, operand.dtype, operand.location)

    def transform_parameter(self, parameter: StealthParameter) -> StealthParameter:
        return StealthParameter(parameter.name, parameter.shape)

    def transform_immediate(self, name: str, value: int) -> tuple[str, int]:
        return name, value

    def transform_loop_index(self, index: StealthIndex) -> StealthIndex:
        return StealthIndex(index.name, index.number_of_iterations, index.stride)

    def transform_statement(self, statement: StealthStatement) -> StealthStatement:
        if isinstance(statement, StealthAllocation):
            return self.transform_allocation(statement)
        elif isinstance(statement, StealthLoad):
            return self.transform_load(statement)
        elif isinstance(statement, StealthStore):
            return self.transform_store(statement)
        elif isinstance(statement, StealthLoop):
            return self.transform_loop(statement)
        elif isinstance(statement, StealthCompute):
            return self.transform_compute(statement)
        else:
            raise RuntimeError(f"Unknown statement type: {statement}")
    
    def transform_allocation(self, statement: StealthAllocation) -> StealthAllocation:
        return StealthAllocation(statement.operand_name, statement.size, statement.location, statement.dtype)

    def transform_load(self, statement: StealthLoad) -> StealthLoad:
        return StealthLoad(statement.destination_operand_name, statement.source_operand_name, statement.source_operand_offset, statement.size)

    def transform_store(self, statement: StealthStore) -> StealthStore:
        return StealthStore(statement.destination_operand_name, statement.destination_operand_offset, statement.source_operand_name)

    def transform_loop(self, statement: StealthLoop) -> StealthLoop:
        return StealthLoop(
            statement.loop_index_variable_name,
            statement.number_of_iterations,
            statement.stride,
            [self.transform_statement(loop_statement) for loop_statement in statement.body],
        )
    
    def transform_compute(self, statement: StealthCompute) -> StealthCompute:
        return StealthCompute(
            statement.destination_operand_name,
            statement.operation_name,
            statement.operands,
            statement.location
        )
