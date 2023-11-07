import dataclasses
from typing import Optional
from stealth.stealth_codelet.core import StealthCodelet, StealthLoad
from ..core import *
from ..expression import get_loop_index_variable_names_in_expression
from .utils import int_to_name
from ..visitor import StealthCodeletVisitor


@dataclasses.dataclass
class ComputeOperationGroup:
    top_level_allocations: tuple[StealthAllocation]
    transfer_and_compute_operations: list[Union[StealthLoad, StealthStore, StealthCompute]]
    loop_index_variables: list[StealthIndex]
    immediates: list[str]

    @property
    def operations(self) -> list[StealthCompute]:
        return [o for o in self.transfer_and_compute_operations if isinstance(o, StealthCompute)]

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
        ret += "\tTop Level Allocations:\n"
        for allocation in self.top_level_allocations:
            ret += f"\t\t{allocation}\n"
        ret += "\tTransfer and Compute Operations:\n"
        for operation in self.transfer_and_compute_operations:
            ret += f"\t\t{operation}\n"
        ret += "\tLoop Index Variables:\n"
        for loop_index_variable in self.loop_index_variables:
            ret += f"\t\t{loop_index_variable}\n"
        return ret


class ComputeOperationGrouper(StealthCodeletVisitor):
    _grouped_compute_operations: list[list[StealthCompute]]

    def __init__(self) -> None:
        super().__init__()
        self._grouped_compute_operations = [[]]
    
    @property
    def grouped_compute_operations(self) -> list[list[StealthCompute]]:
        return list(filter(lambda l: len(l) != 0, self._grouped_compute_operations))
    
    def _push_new_compute_operation_group(self) -> None:
        if len(self._grouped_compute_operations[-1]) != 0:
            self._grouped_compute_operations.append([])
    
    def _add_compute_operation(self, compute_operation: StealthCompute) -> None:
        self._grouped_compute_operations[-1].append(compute_operation)
    
    def visit_loop(self, statement: StealthLoop) -> None:
        self._push_new_compute_operation_group()
        super().visit_loop(statement)

    def visit_compute(self, statement: StealthCompute) -> None:
        if statement.location == "PE_ARRAY":
            self._push_new_compute_operation_group()
        self._add_compute_operation(statement)


def group_compute_operations(codelet: StealthCodelet) -> list[list[StealthCompute]]:
    grouper = ComputeOperationGrouper()
    grouper.visit(codelet)
    grouped_compute_operations: list[list[StealthCompute]] = grouper.grouped_compute_operations
    return grouped_compute_operations


class LinkedOperandNameCollector(StealthCodeletVisitor):
    _operand_names: set[str]

    def __init__(self, initial_operand_names: set[str]) -> None:
        super().__init__()
        self._operand_names = initial_operand_names.copy()
    
    @property
    def operand_names(self) -> set[str]:
        return self._operand_names
    
    def visit_load(self, statement: StealthLoad) -> None:
        if statement.destination_operand_name in self._operand_names:
            self._operand_names.add(statement.source_operand_name)
    
    def visit_store(self, statement: StealthStore) -> None:
        if statement.source_operand_name in self._operand_names:
            self._operand_names.add(statement.destination_operand_name)
    
    def visit_compute(self, statement: StealthCompute) -> None:
        for operand_name in statement.operands:
            if operand_name in self._operand_names:
                self._operand_names.add(statement.destination_operand_name)


def collect_all_operand_names_used_by_compute_operation_sequence(grouped_compute_operations: list[StealthCompute], codelet: StealthCodelet, max_iters: int = 100) -> set[str]:
    operand_names: set[str] = set()
    prev_operand_names: Optional[set[str]] = None
    number_of_iters: int = 0
    while (prev_operand_names is None or len(operand_names.difference(prev_operand_names)) != 0) and number_of_iters <= max_iters:
        prev_operand_names = operand_names.copy()
        for compute_operation in grouped_compute_operations:
            operand_names.update(compute_operation.operands)
            operand_names.add(compute_operation.destination_operand_name)
            collector = LinkedOperandNameCollector(operand_names)
            collector.visit(codelet)
            operand_names.update(collector.operand_names)
        number_of_iters += 1
    return operand_names


class TopLevelAllocationCollector(StealthCodeletVisitor):
    _operand_names: set[str]
    _top_level_allocations: list[StealthAllocation]

    def __init__(self, operand_names: set[str]) -> None:
        super().__init__()
        self._operand_names = operand_names.copy()
        self._top_level_allocations = []
    
    @property
    def top_level_allocations(self) -> tuple[StealthAllocation]:
        return tuple(self._top_level_allocations)

    def visit(self, codelet: StealthCodelet) -> None:
        for statement in codelet._statements:
            if isinstance(statement, StealthAllocation):
                self._top_level_allocations.append(statement)


def collect_top_level_allocations(codelet: StealthCodelet, operand_names: set[str]) -> list[StealthAllocation]:
    collector = TopLevelAllocationCollector(operand_names)
    collector.visit(codelet)
    return collector.top_level_allocations


class TransferAndComputeOperationCollector(StealthCodeletVisitor):
    _operand_names: set[str]
    _compute_operations: list[StealthCompute]
    _transfer_and_compute_operations: list[Union[StealthLoad, StealthCompute, StealthStore]]

    def __init__(self, operand_names: set[str], compute_operations: list[StealthCompute]) -> None:
        super().__init__()
        self._operand_names = operand_names.copy()
        self._compute_operations = compute_operations.copy()
        self._transfer_and_compute_operations = []
    
    @property
    def transfer_and_compute_operations(self) -> list[Union[StealthLoad, StealthCompute, StealthStore]]:
        return self._transfer_and_compute_operations.copy()
    
    def visit_load(self, statement: StealthLoad) -> None:
        if statement.source_operand_name in self._operand_names and statement.destination_operand_name in self._operand_names:
            self._transfer_and_compute_operations.append(statement) 

    def visit_store(self, statement: StealthStore) -> None:
        if statement.source_operand_name in self._operand_names and statement.destination_operand_name in self._operand_names:
            self._transfer_and_compute_operations.append(statement)

    def visit_compute(self, statement: StealthCompute) -> None:
        if statement in self._compute_operations:
            self._transfer_and_compute_operations.append(statement) 


def collect_transfer_and_compute_operations(codelet: StealthCodelet, operand_names: set[str], compute_operations: list[StealthCompute]) -> list[Union[StealthLoad, StealthCompute, StealthStore]]:
    collector = TransferAndComputeOperationCollector(operand_names, compute_operations)
    collector.visit(codelet)
    return collector.transfer_and_compute_operations


class LoopIndexNameCollector(StealthCodeletVisitor):
    _loop_index_names: set[str]

    def __init__(self) -> None:
        super().__init__()
        self._loop_index_names = set()

    @property
    def loop_index_names(self) -> set[str]:
        return self._loop_index_names.copy()
    
    def visit_loop_index(self, index: StealthIndex) -> None:
        self._loop_index_names.add(index.name)


def collect_loop_index_names(codelet: StealthCodelet) -> set[str]:
    collector = LoopIndexNameCollector()
    collector.visit(codelet)
    return collector.loop_index_names


class LoopIndicesOrderCollector(StealthCodeletVisitor):
    _loop_index_names: set[str]
    _loop_indices: list[StealthIndex]

    def __init__(self, loop_index_names: set[str]) -> None:
        super().__init__()
        self._loop_index_names = loop_index_names.copy()
        self._loop_indices = []

    @property
    def loop_indices(self) -> list[StealthIndex]:
        return self._loop_indices.copy()
    
    def visit_loop_index(self, index: StealthIndex) -> None:
        if index.name in self._loop_index_names:
            self._loop_indices.append(index)


def collect_loop_indices(codelet: StealthCodelet, loop_index_names: set[str]) -> list[StealthIndex]:
    collector = LoopIndicesOrderCollector(loop_index_names)
    collector.visit(codelet)
    return collector.loop_indices


class ImmediateCollector(StealthCodeletVisitor):
    _compute_operations: list[StealthCompute]
    _codelet_immediates: Optional[set[str]]
    _immediates: set[str]

    def __init__(self, compute_operations: list[StealthCompute]) -> None:
        super().__init__()
        self._compute_operations = compute_operations.copy()
        self._immediates = set()
        self._codelet_immediates = None
    
    @property
    def immediates(self) -> set[str]:
        return self._immediates.copy()

    def visit(self, codelet: StealthCodelet) -> None:
        self._codelet_immediates = set(codelet._immediates.keys())
        return super().visit(codelet)
    
    def visit_compute(self, statement: StealthCompute) -> None:
        if statement in self._compute_operations:
            for operand_name in statement.operands:
                if operand_name in self._codelet_immediates:
                    self._immediates.add(operand_name)


def collect_immediates(codelet: StealthCodelet, compute_operations: list[StealthCompute]) -> set[str]:
    collector = ImmediateCollector(compute_operations)
    collector.visit(codelet)
    return collector.immediates


def collect_loop_index_names_from_transfer_and_compute_operations(transfer_and_compute_operations: list[Union[StealthLoad, StealthCompute, StealthStore]], all_loop_index_names: set[str]) -> set[str]:
    loop_index_names: set[str] = set()
    for transfer_and_compute_operation in transfer_and_compute_operations:
        if isinstance(transfer_and_compute_operation, StealthLoad):
            for offset in transfer_and_compute_operation.source_operand_offset:
                loop_index_names.update(get_loop_index_variable_names_in_expression(offset, all_loop_index_names))
        elif isinstance(transfer_and_compute_operation, StealthStore):
            for offset in transfer_and_compute_operation.destination_operand_offset:
                loop_index_names.update(get_loop_index_variable_names_in_expression(offset, all_loop_index_names))
    return loop_index_names 


def build_compute_operation_groups(codelet: StealthCodelet) -> list[ComputeOperationGroup]:
    grouped_compute_operations: list[list[StealthCompute]] = group_compute_operations(codelet)
    operand_names_for_groups: list[set[str]] = [collect_all_operand_names_used_by_compute_operation_sequence(group, codelet) for group in grouped_compute_operations]
    top_level_allocations_for_groups: list[tuple[StealthAllocation]] = [collect_top_level_allocations(codelet, operand_names) for operand_names in operand_names_for_groups]
    transfer_and_compute_operations_for_groups: list[list[Union[StealthLoad, StealthCompute, StealthStore]]] = [collect_transfer_and_compute_operations(codelet, operand_names, compute_operations) for operand_names, compute_operations in zip(operand_names_for_groups, grouped_compute_operations)]
    all_loop_index_names: set[str] = collect_loop_index_names(codelet)
    loop_index_names_for_groups: list[set[str]] = [collect_loop_index_names_from_transfer_and_compute_operations(transfer_and_compute_operations, all_loop_index_names) for transfer_and_compute_operations in transfer_and_compute_operations_for_groups]
    loop_indices_for_groups: list[list[StealthIndex]] = [collect_loop_indices(codelet, loop_index_names) for loop_index_names in loop_index_names_for_groups]
    immediates: list[set[str]] = [collect_immediates(codelet, compute_operations) for compute_operations in grouped_compute_operations]
    compute_operation_groups: list[ComputeOperationGroup] = [ComputeOperationGroup(compute_operation_group_top_level_allocations, compute_operation_group_transfer_and_compute_operations, compute_group_operation_loop_index_variables, compute_group_operation_immediates) for compute_operation_group_top_level_allocations, compute_operation_group_transfer_and_compute_operations, compute_group_operation_loop_index_variables, compute_group_operation_immediates in zip(top_level_allocations_for_groups, transfer_and_compute_operations_for_groups, loop_indices_for_groups, immediates)]
    return compute_operation_groups
