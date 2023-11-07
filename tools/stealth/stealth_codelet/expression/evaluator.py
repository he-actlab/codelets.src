from stealth.stealth_codelet.expression.core import StealthBinaryExpression, StealthExpression, StealthUnaryExpression
from .core import *
from .visitor import ExpressionTransformer


class ExpressionEvaluator(ExpressionTransformer):
    _name_to_value_mapping: dict[str, int]

    def __init__(self, name_to_value_mapping: dict[str, int]) -> None:
        super().__init__()
        self._name_to_value_mapping = name_to_value_mapping.copy()
    
    def transform_variable_name(self, variable_name: StealthVariableName) -> StealthExpression:
        if variable_name.name in self._name_to_value_mapping:
            return StealthLiteral(self._name_to_value_mapping[variable_name.name])
        else:
            return super().transform_variable_name(variable_name)
    
    def transform_binary_expression(self, expression: StealthBinaryExpression) -> StealthExpression:
        lhs: StealthExpression = self.transform(expression.lhs)
        rhs: StealthExpression = self.transform(expression.rhs)
        if isinstance(lhs, StealthLiteral) and isinstance(rhs, StealthLiteral):
            if expression.operation == "+":
                return StealthLiteral(lhs.value + rhs.value)
            elif expression.operation == "-":
                return StealthLiteral(lhs.value - rhs.value)
            elif expression.operation == "*":
                return StealthLiteral(lhs.value * rhs.value)
            elif expression.operation == "/":
                return StealthLiteral(lhs.value / rhs.value)
            elif expression.operation == "//":
                return StealthLiteral(lhs.value // rhs.value)
            elif expression.operation == "==":
                return StealthLiteral(lhs.value == rhs.value)
            elif expression.operation == "!=":
                return StealthLiteral(lhs.value != rhs.value)
            elif expression.operation == "<":
                return StealthLiteral(lhs.value < rhs.value)
            elif expression.operation == "<=":
                return StealthLiteral(lhs.value <= rhs.value)
            elif expression.operation == ">":
                return StealthLiteral(lhs.value > rhs.value)
            elif expression.operation == ">=":
                return StealthLiteral(lhs.value >= rhs.value)
            elif expression.operation == "and":
                return StealthLiteral(lhs.value and rhs.value)
            elif expression.operation == "or":
                return StealthLiteral(lhs.value or rhs.value)
            else:
                raise RuntimeError(f"Unknown binary operation: {expression.operation}")
        elif isinstance(lhs, StealthLiteral) and lhs.value == 0:
            if expression.operation in ["/", "//", "*"]:
                return StealthLiteral(0)
            elif expression.operation == "+":
                return rhs
            elif expression.operation == "-":
                return StealthUnaryExpression(rhs, "-")
            else:
                return StealthBinaryExpression(lhs, rhs, expression.operation) 
        elif isinstance(rhs, StealthLiteral) and rhs.value == 0:
            if expression.operation in ["/", "//"]:
                raise ZeroDivisionError
            elif expression.operation == "*":
                return StealthLiteral(0)
            elif expression.operation in ["+", "-"]:
                return lhs
            else:
                return StealthBinaryExpression(lhs, rhs, expression.operation)
        # elif isinstance(lhs, StealthLiteral) and lhs.value == 1:
        #     if expression.operation  == "*":
        #         return rhs
        #     else:
        #         return StealthBinaryExpression(lhs, rhs, expression.operation)
        # elif isinstance(rhs, StealthLiteral) and rhs.value == 1:
        #     if expression.operation in ["*", "//", "/"]:
        #         return lhs
        #     else:
        #         return StealthBinaryExpression(lhs, rhs, expression.operation)
        else:
            return StealthBinaryExpression(lhs, rhs, expression.operation)
    
    def transform_unary_expression(self, expression: StealthUnaryExpression) -> StealthExpression:
        operand: StealthExpression = self.transform(expression.operand)
        if isinstance(operand, StealthLiteral):
            if expression.operation == "-":
                return StealthLiteral(-operand.value)
            elif expression.operation == "not":
                return StealthLiteral(not operand.value)
            else:
                raise RuntimeError(f"Unknown unary operation: {expression.operation}")
        else:
            return StealthUnaryExpression(operand, expression.operation) 


def evaluate_expression(expression: StealthExpression, name_to_value_mapping: dict[str, int]) -> StealthExpression:
    return ExpressionEvaluator(name_to_value_mapping).transform(expression)
