from typing import Any, Optional, Union
from tools.stealth.utils import UniqueNameGenerator, int_to_name


TAB = "    "


def sequence_to_string(sequence: Union[list[Any], tuple[Any, ...]]) -> str:
    return ", ".join(str(s) for s in sequence)


def order_sequence(sequence: tuple[Any], order_sequence: tuple[int, ...]) -> tuple[Any, ...]:
    return tuple(sequence[order] for order in order_sequence)


def index_with_tuple(sequence: Union[list[Any], tuple[Any]], index: tuple[int, ...]) -> tuple[Any]:
    return tuple(sequence[i] for i in index)


def check_operand_names(*operand_names: Optional[str]) -> None:
    for operand_name in operand_names:
        if operand_name is not None:
            assert isinstance(operand_name, str)
    for i in range(len(operand_names)):
        for j in range(i + 1, len(operand_names)):
            if operand_names[i] is not None and operand_names[j] is not None:
                assert operand_names[i] != operand_names[j]


def check_operand_shapes(*operand_dimensions: Optional[tuple[Union[str, int], ...]]) -> None:
    for operand_dimension in operand_dimensions:
        if operand_dimension is None:
            continue
        assert isinstance(operand_dimension, tuple)
        for dim in operand_dimension:
            assert isinstance(dim, (str, int))
            if isinstance(dim, int):
                assert dim > 0
            elif isinstance(dim, str):
                assert dim.isupper()


def check_shape_be_broadcast_to_shape(shape_to_be_broadcast: tuple[Union[str, int], ...], reference_shape: tuple[Union[str, int], ...]) -> None:
    assert len(shape_to_be_broadcast) <= len(reference_shape)
    shape_to_be_broadcast = reference_shape[:len(reference_shape) - len(shape_to_be_broadcast)] + shape_to_be_broadcast
    assert len(shape_to_be_broadcast) == len(reference_shape)
    for shape_to_be_broadcast_dim, reference_shape_dim in zip(shape_to_be_broadcast, reference_shape):
        if isinstance(shape_to_be_broadcast_dim, int):
            assert isinstance(reference_shape_dim, int)
            assert shape_to_be_broadcast_dim == reference_shape_dim
        elif isinstance(shape_to_be_broadcast_dim, str):
            assert isinstance(reference_shape_dim, str)
            assert shape_to_be_broadcast_dim == reference_shape_dim


def check_simd_width(simd_width: int) -> None:
    assert isinstance(simd_width, int)
    assert simd_width > 0


def check_pe_array_dimensions(pe_array_width: int, pe_array_height: int) -> None:
    assert isinstance(pe_array_width, int)
    assert isinstance(pe_array_height, int)
    assert pe_array_width > 0
    assert pe_array_height > 0


def check_tiling(tiling: tuple[int, ...], number_of_dimensions: int) -> None:
    assert len(tiling) == number_of_dimensions
    assert all(tile_size > 0 for tile_size in tiling)


def create_default_tiling(dimensions: tuple[Union[str, int], ...]) -> tuple[int, ...]:
    return tuple(1 for _ in range(len(dimensions)))


def create_tile_size_strings(dimensions: tuple[Union[str, int], ...], tiling: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(f"{dim_size} // {number_of_tiles}" for dim_size, number_of_tiles in zip(dimensions, tiling))


def check_loop_order(loop_order: tuple[int, ...], number_of_dimensions: int) -> None:
    assert len(loop_order) == number_of_dimensions
    assert set(loop_order) == set(range(number_of_dimensions))


def create_default_loop_order(dimensions: tuple[Union[str, int], ...]) -> tuple[int, ...]:
    return tuple(range(len(dimensions)))


def check_vmem(vmem: Optional[str]) -> None:
    assert vmem in ("VMEM1", "VMEM2")


def name_operand_tile(operand_name: str) -> str:
    return operand_name + "_tile"


def name_operand_element(operand_name: str) -> str:
    return operand_name + "_element"


def create_intermediate_names_for_loop_indices(dimensions: tuple[Union[str, int], ...]) -> tuple[str, ...]:
    return tuple(int_to_name(dim).lower() if isinstance(dim, int) else dim.lower() for dim in dimensions)


def create_outer_loop_index_names(intermediate_names_for_loop_indices: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(loop_index + "_tile_index" for loop_index in intermediate_names_for_loop_indices)


def create_inner_loop_index_names(intermediate_names_for_loop_indices: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(loop_index + "_element_index" for loop_index in intermediate_names_for_loop_indices)


def create_operation_name(operation_name: str, *operand_shapes: tuple[Union[str, int], ...]) -> str:
    if all(len(shape_1) == len(shape_2) for shape_1, shape_2 in zip(operand_shapes, operand_shapes[1:])):
        return operation_name + str(len(operand_shapes[0])) + "d"
    else:
        return operation_name + "_".join(str(len(shape)) + "d" for shape in operand_shapes)


def format_line(number_of_tabs: int, line: str) -> str:
    return TAB * number_of_tabs + line + "\n"
