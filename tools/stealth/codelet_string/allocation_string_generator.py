from typing import Union
from .utils import sequence_to_string


def generate_allocation(operand_name: str, size: tuple[Union[str, int], ...], location: str, dtype: str) -> str:
    return f"{operand_name} = alloc([{sequence_to_string(size)}], {location}, {dtype})"
