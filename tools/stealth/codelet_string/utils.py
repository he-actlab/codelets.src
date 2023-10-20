from typing import Any, Union


TAB = "    "


def sequence_to_string(sequence: Union[list[Any], tuple[Any, ...]]) -> str:
    return ", ".join(str(s) for s in sequence)


def order_sequence(sequence: tuple[Any], order_sequence: tuple[int, ...]) -> tuple[Any, ...]:
    return tuple(sequence[order] for order in order_sequence)
