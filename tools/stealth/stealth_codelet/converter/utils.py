from ..core import *
from ..expression import *
from tools.stealth.utils import int_to_name, UniqueNameGenerator


def input_output_dimension_to_name(dimension: StealthExpression) -> str:
    if isinstance(dimension, StealthVariableName):
        return dimension.name
    elif isinstance(dimension, StealthLiteral):
        return int_to_name(dimension.value)
    else:
        raise TypeError(f"Unknown stealth codelet operand dimension type: {type(dimension)}")
