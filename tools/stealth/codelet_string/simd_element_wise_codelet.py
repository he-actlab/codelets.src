from typing import Any, Callable, Optional, Union
from .header_string_generator import generate_header
from .allocation_string_generator import generate_allocation
from .loop_string_generator import generate_inner_for_loop, generate_outer_for_loop
from .load_string_generator import generate_load
from .store_string_generator import generate_store
from .return_string_generator import generate_return
from .compute_string_generator import generate_relu, generate_sigmoid, generate_tanh
from .compute_string_generator import generate_add, generate_sub, generate_mul, generate_div
from .utils import TAB, order_sequence
from tools.stealth.utils import UniqueNameGenerator, int_to_name


def _check_tiling(tiling: tuple[int, ...], number_of_dimensions: int) -> None:
    assert len(tiling) == number_of_dimensions
    assert all(tile_size > 0 for tile_size in tiling)


def _create_default_tiling(dimensions: tuple[Union[str, int], ...]) -> tuple[int, ...]:
    return tuple(1 for _ in range(len(dimensions)))


def _create_tile_size_strings(dimensions: tuple[Union[str, int], ...], tiling: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(f"{dim_size} // {number_of_tiles}" for dim_size, number_of_tiles in zip(dimensions, tiling))


def _check_loop_order(loop_order: tuple[int, ...], number_of_dimensions: int) -> None:
    assert len(loop_order) == number_of_dimensions
    assert set(loop_order) == set(range(number_of_dimensions))


def _create_default_loop_order(dimensions: tuple[Union[str, int], ...]) -> tuple[int, ...]:
    return tuple(range(len(dimensions)))


def _check_vmem(vmem: Optional[str]) -> None:
    assert vmem in ("VMEM1", "VMEM2")


def _name_operand_tile(operand_name: str) -> str:
    return operand_name + "_tile"


def _name_operand_element(operand_name: str) -> str:
    return operand_name + "_element"


def _create_intermediate_names_for_loop_indices(dimensions: tuple[Union[str, int], ...]) -> tuple[str, ...]:
    return tuple(int_to_name(dim).lower() if isinstance(dim, int) else dim.lower() for dim in dimensions)


def _create_outer_loop_index_names(intermediate_names_for_loop_indices: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(loop_index + "_tile_index" for loop_index in intermediate_names_for_loop_indices)


def _create_inner_loop_index_names(intermediate_names_for_loop_indices: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(loop_index + "_element_index" for loop_index in intermediate_names_for_loop_indices)


def _format_line(number_of_tabs: int, line: str) -> str:
    return TAB * number_of_tabs + line + "\n"


def _generate_simd_unary_element_wise_codelet(codelet_name: str, input_operand: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], compute_operation_generator_function: Callable[[str, Any], str], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    input_operand_name, input_operand_shape = input_operand
    output_operand_name, output_operand_shape = output_operand
    assert input_operand_name != output_operand_name
    assert len(input_operand_shape) == len(output_operand_shape)
    assert all(in_dim == out_dim for in_dim, out_dim in zip(input_operand_shape, output_operand_shape))
    assert all(dim.isupper() for dim in input_operand_shape)
    assert all(dim.isupper() for dim in output_operand_shape)

    if dimension_tiling is None:
        dimension_tiling = _create_default_tiling(input_operand_shape)
    _check_tiling(dimension_tiling, len(input_operand_shape))
    dimension_tile_sizes: tuple[int, ...] = _create_tile_size_strings(input_operand_shape, dimension_tiling) 
    if loop_order is None:
        loop_order = _create_default_loop_order(input_operand_shape)
    _check_loop_order(loop_order, len(input_operand_shape))
    if input_vmem is None:
        input_vmem = "VMEM1"
    _check_vmem(input_vmem)
    if output_vmem is None:
        output_vmem = "VMEM2"
    _check_vmem(output_vmem)

    input_tile_name: str = _name_operand_tile(input_operand_name)
    output_tile_name: str = _name_operand_tile(output_operand_name)
    input_element_name: str = _name_operand_element(input_operand_name)
    output_element_name: str = _name_operand_element(output_operand_name)

    intermediate_loop_indices: tuple[str, ...] = _create_intermediate_names_for_loop_indices(input_operand_shape) 
    outer_loop_indices: tuple[str, ...] = _create_outer_loop_index_names(intermediate_loop_indices)
    inner_loop_indices: tuple[str, ...] = _create_inner_loop_index_names(intermediate_loop_indices)

    dimension_in_order: tuple[Union[str, int], ...] = order_sequence(input_operand_shape, loop_order)
    dimension_tiling_in_order: tuple[int, ...] = order_sequence(dimension_tiling, loop_order)
    outer_loop_indices_in_order: tuple[str, ...] = order_sequence(outer_loop_indices, loop_order)
    inner_loop_indices_in_order: tuple[str, ...] = order_sequence(inner_loop_indices, loop_order)
    inner_loop_stride_in_order: tuple[Union[str, int], ...] = tuple(simd_width if loop_number == len(input_operand_shape) - 1 else 1 for loop_number in loop_order)

    header_string: str = generate_header(codelet_name, ((input_operand_name, "i32", input_operand_shape, "DRAM"),), ())
    output_alloc: str = generate_allocation(output_operand_name, output_operand_shape, "DRAM", "i32")
    outer_loop_strings: tuple[str, ...] = tuple(generate_outer_for_loop(loop_index, number_of_tiles, dim_size) for loop_index, dim_size, number_of_tiles in zip(outer_loop_indices_in_order, dimension_in_order, dimension_tiling_in_order))
    output_tile_alloc: str = generate_allocation(output_tile_name, dimension_tile_sizes, output_vmem, "i32")
    input_tile_load: str = generate_load(input_operand_name, outer_loop_indices, input_tile_name, dimension_tile_sizes, input_vmem) 
    inner_loop_strings: tuple[str, ...] = tuple(generate_inner_for_loop(loop_index, dim_size, number_of_tiles, loop_stride) for loop_index, dim_size, number_of_tiles, loop_stride in zip(inner_loop_indices_in_order, dimension_in_order, dimension_tiling_in_order, inner_loop_stride_in_order))
    input_element_load: str = generate_load(input_tile_name, inner_loop_indices, input_element_name, [simd_width], "SIMD")
    compute: str = compute_operation_generator_function(output_element_name, input_element_name)
    output_element_store: str = generate_store(output_element_name, output_tile_name, inner_loop_indices)
    output_tile_store: str = generate_store(output_tile_name, output_operand_name, outer_loop_indices)
    return_statement: str = generate_return((output_operand_name,))

    number_of_tabs: int = 0
    codelet_string: str = _format_line(number_of_tabs, header_string) 
    number_of_tabs += 1
    codelet_string += _format_line(number_of_tabs, output_alloc)
    for outer_loop_string in outer_loop_strings:
        codelet_string += _format_line(number_of_tabs, outer_loop_string)
        number_of_tabs += 1
    codelet_string += _format_line(number_of_tabs, output_tile_alloc)
    codelet_string += _format_line(number_of_tabs, input_tile_load)
    for inner_loop_string in inner_loop_strings:
        codelet_string += _format_line(number_of_tabs, inner_loop_string)
        number_of_tabs += 1
    codelet_string += _format_line(number_of_tabs, input_element_load)
    codelet_string += _format_line(number_of_tabs, compute)
    codelet_string += _format_line(number_of_tabs, output_element_store)
    number_of_tabs -= len(inner_loop_strings)
    codelet_string += _format_line(number_of_tabs, output_tile_store)
    number_of_tabs -= len(outer_loop_strings)
    codelet_string += _format_line(number_of_tabs, return_statement)
    return codelet_string


def generate_simd_element_wise_relu_codelet(input_operand: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_unary_element_wise_codelet(f"relu{len(input_operand[1])}d", input_operand, output_operand, generate_relu, simd_width, dimension_tiling, loop_order, input_vmem, output_vmem)


def generate_simd_element_wise_sigmoid_codelet(input_operand: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_unary_element_wise_codelet(f"sigmoid{len(input_operand[1])}d", input_operand, output_operand, generate_sigmoid, simd_width, dimension_tiling, loop_order, input_vmem, output_vmem)


def generate_simd_element_wise_tanh_codelet(input_operand: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_unary_element_wise_codelet(f"tanh{len(input_operand[1])}d", input_operand, output_operand, generate_tanh, simd_width, dimension_tiling, loop_order, input_vmem, output_vmem)


def _generate_simd_binary_element_wise_codelet(codelet_name: str, input_operand_1: tuple[str, tuple[Union[str, int], ...]], input_operand_2: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], compute_operation_generator_function: Callable[[str, Any], str], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_1_vmem: Optional[str] = None, input_2_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    input_1_operand_name, input_1_operand_shape = input_operand_1
    input_2_operand_name, input_2_operand_shape = input_operand_2
    output_operand_name, output_operand_shape = output_operand
    assert input_1_operand_name != output_operand_name
    assert input_2_operand_name != output_operand_name
    assert len(input_1_operand_shape) == len(input_2_operand_shape) == len(output_operand_shape)
    assert all(in_1_dim == in_2_dim == out_dim for in_1_dim, in_2_dim, out_dim in zip(input_1_operand_shape, input_2_operand_shape, output_operand_shape))

    if dimension_tiling is None:
        dimension_tiling = _create_default_tiling(input_1_operand_shape)
    _check_tiling(dimension_tiling, len(input_1_operand_shape))
    dimension_tile_sizes: tuple[int, ...] = _create_tile_size_strings(input_1_operand_shape, dimension_tiling)
    if loop_order is None:
        loop_order = _create_default_loop_order(input_1_operand_shape)
    _check_loop_order(loop_order, len(input_1_operand_shape))
    if input_1_vmem is None:
        input_1_vmem = "VMEM1"
    _check_vmem(input_1_vmem)
    if input_2_vmem is None:
        input_2_vmem = "VMEM2"
    _check_vmem(input_2_vmem)
    if output_vmem is None:
        output_vmem = "VMEM1"
    _check_vmem(output_vmem)

    input_1_tile_name: str = _name_operand_tile(input_1_operand_name)
    input_2_tile_name: str = _name_operand_tile(input_2_operand_name)
    output_tile_name: str = _name_operand_tile(output_operand_name)
    input_1_element_name: str = _name_operand_element(input_1_operand_name)
    input_2_element_name: str = _name_operand_element(input_2_operand_name)
    output_element_name: str = _name_operand_element(output_operand_name)

    intermediate_loop_indices: tuple[str, ...] = _create_intermediate_names_for_loop_indices(input_1_operand_shape)
    outer_loop_indices: tuple[str, ...] = _create_outer_loop_index_names(intermediate_loop_indices)
    inner_loop_indices: tuple[str, ...] = _create_inner_loop_index_names(intermediate_loop_indices)

    dimension_in_order: tuple[Union[str, int], ...] = order_sequence(input_1_operand_shape, loop_order)
    dimension_tiling_in_order: tuple[int, ...] = order_sequence(dimension_tiling, loop_order)
    outer_loop_indices_in_order: tuple[str, ...] = order_sequence(outer_loop_indices, loop_order)
    inner_loop_indices_in_order: tuple[str, ...] = order_sequence(inner_loop_indices, loop_order)
    inner_loop_stride_in_order: tuple[Union[str, int], ...] = tuple(simd_width if loop_number == len(input_1_operand_shape) - 1 else 1 for loop_number in loop_order)

    header_string: str = generate_header(codelet_name, ((input_1_operand_name, "i32", input_1_operand_shape, "DRAM"), (input_2_operand_name, "i32", input_2_operand_shape, "DRAM")), ())
    output_alloc: str = generate_allocation(output_operand_name, output_operand_shape, "DRAM", "i32")
    outer_loop_strings: tuple[str, ...] = tuple(generate_outer_for_loop(loop_index, number_of_tiles, dim_size) for loop_index, dim_size, number_of_tiles in zip(outer_loop_indices_in_order, dimension_in_order, dimension_tiling_in_order))
    output_tile_alloc: str = generate_allocation(output_tile_name, dimension_tile_sizes, output_vmem, "i32")
    input_1_tile_load: str = generate_load(input_1_operand_name, outer_loop_indices, input_1_tile_name, dimension_tile_sizes, input_1_vmem)
    input_2_tile_load: str = generate_load(input_2_operand_name, outer_loop_indices, input_2_tile_name, dimension_tile_sizes, input_2_vmem)
    inner_loop_strings: tuple[str, ...] = tuple(generate_inner_for_loop(loop_index, dim_size, number_of_tiles, loop_stride) for loop_index, dim_size, number_of_tiles, loop_stride in zip(inner_loop_indices_in_order, dimension_in_order, dimension_tiling_in_order, inner_loop_stride_in_order))
    input_1_element_load: str = generate_load(input_1_tile_name, inner_loop_indices, input_1_element_name, [simd_width], "SIMD")
    input_2_element_load: str = generate_load(input_2_tile_name, inner_loop_indices, input_2_element_name, [simd_width], "SIMD")
    compute: str = compute_operation_generator_function(output_element_name, (input_1_element_name, input_2_element_name))
    output_element_store: str = generate_store(output_element_name, output_tile_name, inner_loop_indices)
    output_tile_store: str = generate_store(output_tile_name, output_operand_name, outer_loop_indices)
    return_statement: str = generate_return((output_operand_name,))

    number_of_tabs: int = 0
    codelet_string: str = _format_line(number_of_tabs, header_string)
    number_of_tabs += 1
    codelet_string += _format_line(number_of_tabs, output_alloc)
    for outer_loop_string in outer_loop_strings:
        codelet_string += _format_line(number_of_tabs, outer_loop_string)
        number_of_tabs += 1
    codelet_string += _format_line(number_of_tabs, output_tile_alloc)
    codelet_string += _format_line(number_of_tabs, input_1_tile_load)
    codelet_string += _format_line(number_of_tabs, input_2_tile_load)
    for inner_loop_string in inner_loop_strings:
        codelet_string += _format_line(number_of_tabs, inner_loop_string)
        number_of_tabs += 1
    codelet_string += _format_line(number_of_tabs, input_1_element_load)
    codelet_string += _format_line(number_of_tabs, input_2_element_load)
    codelet_string += _format_line(number_of_tabs, compute)
    codelet_string += _format_line(number_of_tabs, output_element_store)
    number_of_tabs -= len(inner_loop_strings)
    codelet_string += _format_line(number_of_tabs, output_tile_store)
    number_of_tabs -= len(outer_loop_strings)
    codelet_string += _format_line(number_of_tabs, return_statement)
    return codelet_string


def generate_simd_element_wise_add_codelet(input_operand_1: tuple[str, tuple[Union[str, int], ...]], input_operand_2: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_1_vmem: Optional[str] = None, input_2_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_binary_element_wise_codelet(f"add{len(input_operand_1[1])}d", input_operand_1, input_operand_2, output_operand, generate_add, simd_width, dimension_tiling, loop_order, input_1_vmem, input_2_vmem, output_vmem)


def generate_simd_element_wise_sub_codelet(input_operand_1: tuple[str, tuple[Union[str, int], ...]], input_operand_2: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_1_vmem: Optional[str] = None, input_2_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_binary_element_wise_codelet(f"sub{len(input_operand_1[1])}d", input_operand_1, input_operand_2, output_operand, generate_sub, simd_width, dimension_tiling, loop_order, input_1_vmem, input_2_vmem, output_vmem)


def generate_simd_element_wise_mul_codelet(input_operand_1: tuple[str, tuple[Union[str, int], ...]], input_operand_2: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_1_vmem: Optional[str] = None, input_2_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_binary_element_wise_codelet(f"mul{len(input_operand_1[1])}d", input_operand_1, input_operand_2, output_operand, generate_mul, simd_width, dimension_tiling, loop_order, input_1_vmem, input_2_vmem, output_vmem)


def generate_simd_element_wise_div_codelet(input_operand_1: tuple[str, tuple[Union[str, int], ...]], input_operand_2: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_1_vmem: Optional[str] = None, input_2_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_binary_element_wise_codelet(f"div{len(input_operand_1[1])}d", input_operand_1, input_operand_2, output_operand, generate_div, simd_width, dimension_tiling, loop_order, input_1_vmem, input_2_vmem, output_vmem)
