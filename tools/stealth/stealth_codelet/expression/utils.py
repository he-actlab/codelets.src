from tools.stealth.stealth_codelet.expression.core import StealthVariableName
from .core import *
from .visitor import ExpressionVisitor


def is_expression_constant(expression: StealthExpression, constant_variable_names: set[str] = set()) -> bool:
        if isinstance(expression, StealthLiteral):
            return True
        elif isinstance(expression, StealthVariableName):
            return expression.name in constant_variable_names 
        elif isinstance(expression, StealthUnaryExpression):
            return is_expression_constant(expression.operand)
        elif isinstance(expression, StealthBinaryExpression):
            return is_expression_constant(expression.lhs) and is_expression_constant(expression.rhs)
        else:
            raise RuntimeError(f"Unknown expression type: {type(expression)}")
 

class LoopIndexVariableNameGetter(ExpressionVisitor):
    _all_loop_index_variable_names: set[str]
    _loop_index_variable_names_in_expression: set[str] 

    def __init__(self, all_loop_index_variable_names: set[str]) -> None:
        super().__init__()
        self._all_loop_index_variable_names = all_loop_index_variable_names.copy()
    
    @property
    def loop_index_variable_names_in_expression(self) -> set[str]:
        return self._loop_index_variable_names_in_expression.copy()
    
    def reset(self) -> None:
        self._loop_index_variable_names_in_expression = set()
    
    def visit_variable_name(self, expression: StealthVariableName) -> None:
        if expression.name in self._all_loop_index_variable_names:
            self._loop_index_variable_names_in_expression.add(expression.name)
    

def get_loop_index_variable_names_in_expression(expression: StealthExpression, all_loop_index_variable_names: set[str]) -> set[str]:
    getter = LoopIndexVariableNameGetter(all_loop_index_variable_names)
    getter(expression)
    return getter.loop_index_variable_names_in_expression.copy()
