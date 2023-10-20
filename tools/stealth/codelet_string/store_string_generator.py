from typing import Union
from .utils import sequence_to_string


def generate_store(source_operand_name: str, destination_operand_name: str, destination_operand_offset: tuple[Union[str, int], ...]) -> str:
    return f"store({destination_operand_name}[{sequence_to_string(destination_operand_offset)}], {source_operand_name})"
