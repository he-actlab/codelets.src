from typing import Union
from .utils import sequence_to_string


def generate_load(source_operand_name: str, source_operand_offset: tuple[Union[str, int], ...], destination_operand_name: str, size: tuple[Union[str, int], ...], location: str) -> str:
    return f"{destination_operand_name} = load({source_operand_name}[{sequence_to_string(source_operand_offset)}], [{sequence_to_string(size)}], {location})"
