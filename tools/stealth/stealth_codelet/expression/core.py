import abc
import dataclasses
from typing import Union


class StealthExpression(abc.ABC):
    @abc.abstractmethod
    def __str__(self) -> str:
        pass


@dataclasses.dataclass(frozen=True)
class StealthVariableName(StealthExpression):
    name: str

    def __str__(self) -> str:
        if isinstance(self.name, StealthVariableName):
            return self.name.name
        return self.name


@dataclasses.dataclass(frozen=True)
class StealthLiteral(StealthExpression):
    value: Union[int, bool]

    def __str__(self) -> str:
        return str(self.value)


@dataclasses.dataclass(frozen=True)
class StealthBinaryExpression(StealthExpression):
    lhs: StealthExpression
    rhs: StealthExpression
    operation: str

    def __str__(self) -> str:
        return f"({self.lhs} {self.operation} {self.rhs})"


@dataclasses.dataclass(frozen=True)
class StealthUnaryExpression(StealthExpression):
    operand: StealthExpression
    operation: str

    def __str__(self) -> str:
        return f"{self.operation}{self.operand}"
