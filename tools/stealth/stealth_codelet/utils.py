from typing import Any, Callable


def get_compute_operation_output_shape(compute_operation_name: str, compute_arguments: list[Any], operands: dict[str, Any], compute_unit: str, shape_func: Callable[[Any], Any]):
    if compute_operation_name in ["mvmul", "mvmul_bias"]:
        return shape_func(operands[compute_arguments[-1]])
    elif compute_operation_name in ["add", "sub", "mul", "div", "max", "min",
                                    "rshift", "lshift",
                                    "relu", "leaky_relu", "sigmoid", "tanh",
                                    "exp", "sqrt", "inv_sqrt", "log2"]:
        if compute_arguments[0] in operands:
            return shape_func(operands[compute_arguments[0]])
        elif len(compute_arguments) > 1 and compute_arguments[1] in operands:
            return shape_func(operands[compute_arguments[1]])
        else:
            raise RuntimeError(f"Something went wrong... Expected at least one operand to be an operand but instead got {compute_arguments}")
    else:
        raise NotImplementedError(f"{compute_operation_name} is not implemented yet")
