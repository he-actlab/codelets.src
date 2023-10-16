from typing import Optional

from tools.stealth.stealth_codelet.core import StealthAllocation

from ...stealth_codelet.core import *
from ..expression import evaluate_expression, is_expression_constant, StealthExpression, StealthLiteral
from ..visitor import StealthCodeletVisitor


class AllVariablesSetChecker(StealthCodeletVisitor):
    _is_all_variables_set: Optional[bool]
    _loop_index_names: Optional[set[str]]

    def __init__(self) -> None:
        self._is_all_variables_set = None
        self._loop_index_names = None
    
    @property
    def is_all_variables_set(self) -> bool:
        assert self._is_all_variables_set is not None
        return self._is_all_variables_set
    
    def visit(self, codelet: StealthCodelet) -> None:
        self._is_all_variables_set = True
        self._loop_index_names = set(lambda index: index.name, codelet._loop_indices)
        return super().visit(codelet)
    
    def visit_operand(self, operand: StealthOperand) -> None:
        self._check_shape(operand.shape)
    
    def visit_allocation(self, statement: StealthAllocation) -> None:
        self._check_shape(statement.size)
    
    def visit_load(self, statement: StealthLoad) -> None:
        self._check_offsets(statement.source_operand_offset)
        self._check_shape(statement.size)
    
    def visit_store(self, statement: StealthStore) -> None:
        self._check_offsets(statement.destination_operand_offset)
    
    def _check_offsets(self, offsets: list[StealthExpression]) -> None:
        for offset in offsets:
            if not is_expression_constant(offset, self._loop_index_names):
                self._is_all_variables_set = False

    def _check_shape(self, shape: list[StealthExpression]) -> None:
        for dim in shape:
            if not isinstance(evaluate_expression(dim, {}), StealthLiteral):
                self._is_all_variables_set = False


def check_all_variables_set(codelet: StealthCodelet) -> None:
    visitor = AllVariablesSetChecker()
    visitor.visit(codelet)
    if not visitor.is_all_variables_set:
        raise RuntimeError("Not all variables are set in codelet")
