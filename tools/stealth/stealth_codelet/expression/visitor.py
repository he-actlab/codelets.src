from .core import *


class ExpressionVisitor:
    def visit(self, expression: StealthExpression) -> None:
        if isinstance(expression, StealthLiteral):
            self.visit_literal(expression)
        elif isinstance(expression, StealthVariableName):
            self.visit_variable_name(expression)
        elif isinstance(expression, StealthBinaryExpression):
            self.visit_binary_expression(expression)
        elif isinstance(expression, StealthUnaryExpression):
            self.visit_unary_expression(expression)
        else:
            raise RuntimeError(f"Unknown expression type: {expression}")
    
    def visit_literal(self, expression: StealthLiteral) -> None:
        pass

    def visit_variable_name(self, expression: StealthVariableName) -> None:
        pass

    def visit_binary_expression(self, expression: StealthBinaryExpression) -> None:
        self.visit(expression.lhs)
        self.visit(expression.rhs)

    def visit_unary_expression(self, expression: StealthUnaryExpression) -> None:
        self.visit(expression.operand)


class ExpressionTransformer:
    def transform(self, expression: StealthExpression) -> StealthExpression:
        if isinstance(expression, StealthLiteral):
            return self.transform_literal(expression)
        elif isinstance(expression, StealthVariableName):
            return self.transform_variable_name(expression)
        elif isinstance(expression, StealthBinaryExpression):
            return self.transform_binary_expression(expression)
        elif isinstance(expression, StealthUnaryExpression):
            return self.transform_unary_expression(expression)
        else:
            raise RuntimeError(f"Unknown expression type: {expression}")
    
    def transform_literal(self, expression: StealthLiteral) -> StealthExpression:
        return StealthLiteral(expression.value)

    def transform_variable_name(self, expression: StealthVariableName) -> StealthExpression:
        return StealthVariableName(expression.name)

    def transform_binary_expression(self, expression: StealthBinaryExpression) -> StealthExpression:
        lhs = self.transform(expression.lhs)
        rhs = self.transform(expression.rhs)
        return StealthBinaryExpression(lhs, expression.operation, rhs)

    def transform_unary_expression(self, expression: StealthUnaryExpression) -> StealthExpression:
        operand = self.transform(expression.operand)
        return StealthUnaryExpression(operand, expression.operation)
