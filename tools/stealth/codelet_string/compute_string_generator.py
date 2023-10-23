from typing import Union
from .utils import sequence_to_string


def _generate_compute(destination_operand_name: str, operation_name: str, arguments: tuple[Union[str, int], ...], compute_unit: str) -> str:
    return f"{destination_operand_name} = {operation_name}({sequence_to_string(arguments)}, {compute_unit})"


def _generate_simd_compute(destination_operand_name: str, operation_name: str, arguments: tuple[Union[str, int], ...]) -> str:
    return _generate_compute(destination_operand_name, operation_name, arguments, "SIMD")


def _generate_pe_array_compute(destination_operand_name: str, operation_name: str, arguments: tuple[Union[str, int], ...]) -> str:
    return _generate_compute(destination_operand_name, operation_name, arguments, "PE_ARRAY")


def generate_mvmul(destination_operand_name: str, arguments: tuple[str, ...]) -> str:
    if len(arguments) == 3:
        return _generate_pe_array_compute(destination_operand_name, "mvmul", arguments)
    elif len(arguments) == 4:
        return _generate_simd_compute(destination_operand_name, "mvmul_bias", arguments)
    else:
        raise ValueError("Invalid number of arguments for mvmul")


def generate_add(destination_operand_name: str, arguments: tuple[Union[str, int], ...]) -> str:
    if len(arguments) != 2:
        raise ValueError("Invalid number of arguments for add")
    return _generate_simd_compute(destination_operand_name, "add", arguments)


def generate_sub(destination_operand_name: str, arguments: tuple[Union[str, int], ...]) -> str:
    if len(arguments) != 2:
        raise ValueError("Invalid number of arguments for sub")
    return _generate_simd_compute(destination_operand_name, "sub", arguments)


def generate_mul(destination_operand_name: str, arguments: tuple[Union[str, int], ...]) -> str:
    if len(arguments) != 2:
        raise ValueError("Invalid number of arguments for mul")
    return _generate_simd_compute(destination_operand_name, "mul", arguments)


def generate_div(destination_operand_name: str, arguments: tuple[Union[str, int], ...]) -> str:
    if len(arguments) != 2:
        raise ValueError("Invalid number of arguments for div")
    return _generate_simd_compute(destination_operand_name, "div", arguments)


def generate_max(destination_operand_name: str, arguments: tuple[Union[str, int], ...]) -> str:
    if len(arguments) != 2:
        raise ValueError("Invalid number of arguments for max")
    return _generate_simd_compute(destination_operand_name, "max", arguments)


def generate_min(destination_operand_name: str, arguments: tuple[Union[str, int], ...]) -> str:
    if len(arguments) != 2:
        raise ValueError("Invalid number of arguments for min")
    return _generate_simd_compute(destination_operand_name, "min", arguments)


def generate_pow(destination_operand_name: str, arguments: tuple[Union[str, int], ...]) -> str:
    if len(arguments) != 2:
        raise ValueError("Invalid number of arguments for pow")
    return _generate_simd_compute(destination_operand_name, "pow", arguments)


def generate_relu(destination_operand_name: str, argument: Union[str, int]) -> str:
    return _generate_simd_compute(destination_operand_name, "relu", (argument,))


def generate_sigmoid(destination_operand_name: str, argument: Union[str, int]) -> str:
    return _generate_simd_compute(destination_operand_name, "sigmoid", (argument,))


def generate_tanh(destination_operand_name: str, argument: Union[str, int]) -> str:
    return _generate_simd_compute(destination_operand_name, "tanh", (argument,))


def generate_sqrt(destination_operand_name: str, argument: Union[str, int]) -> str:
    return _generate_simd_compute(destination_operand_name, "sqrt", (argument,))
