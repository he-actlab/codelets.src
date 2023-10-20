from typing import Union
from .utils import sequence_to_string


def _generate_header(operation_name: str, arguments: tuple[str, ...]) -> str:
    return f"def {operation_name}({sequence_to_string(arguments)}):"


def generate_header(operation_name: str, inputs: tuple[tuple[str, str, tuple[Union[str, int], ...], str], ...], params: tuple[Union[tuple[str, tuple[Union[str, int], ...]], str], ...]) -> str:
    input_sequence: tuple[str, ...] = tuple(f"{input_name}: {input_dtype}[{sequence_to_string(input_shape)}] @ {input_location}" for input_name, input_dtype, input_shape, input_location in inputs)
    param_sequence: tuple[str, ...] = tuple(f"{params[0]}: param[{sequence_to_string(param[1])}]" if isinstance(param, tuple) else f"{param}: param" for param in params)
    return _generate_header(operation_name, input_sequence + param_sequence)
