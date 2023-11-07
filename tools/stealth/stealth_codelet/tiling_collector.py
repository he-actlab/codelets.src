from .core import StealthLoop, StealthCodelet
from .visitor import StealthCodeletVisitor
from .expression import evaluate_expression, StealthExpression, StealthLiteral, StealthVariableName, StealthBinaryExpression


class TilingCollector(StealthCodeletVisitor):
    _cache: dict[int, dict[str, int]] = {}

    _tiling: dict[str, int]

    def __init__(self) -> None:
        super().__init__()
        self._tiling = {}
    
    def visit(self, codelet: StealthCodelet) -> None:
        key: int = id(codelet)
        if key in TilingCollector._cache:
            self._tiling = TilingCollector._cache[key]
        else:
            super().visit(codelet)
            TilingCollector._cache[key] = self._tiling
    
    def visit_loop(self, statement: StealthLoop) -> None:
        number_of_iterations: StealthExpression = evaluate_expression(statement.number_of_iterations, {})
        stride: StealthExpression = evaluate_expression(statement.stride, {})
        if isinstance(number_of_iterations, StealthLiteral) and isinstance(stride, StealthBinaryExpression) and isinstance(stride.lhs, StealthVariableName) and isinstance(stride.rhs, StealthLiteral):
            self._tiling[stride.lhs.name] = number_of_iterations.value 
        return super().visit_loop(statement)


def collect_tiling(codelet: StealthCodelet) -> dict[str, int]:
    tiling_collector = TilingCollector()
    tiling_collector.visit(codelet)
    return tiling_collector._tiling
