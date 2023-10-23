from typing import Any, Callable, Optional, Union
from .header_string_generator import generate_header
from .allocation_string_generator import generate_allocation
from .loop_string_generator import generate_inner_for_loop, generate_outer_for_loop, create_loops_with_dependent_statements
from .load_string_generator import generate_load
from .store_string_generator import generate_store
from .return_string_generator import generate_return
from .compute_string_generator import generate_relu, generate_sigmoid, generate_tanh, generate_sqrt
from .compute_string_generator import generate_add, generate_sub, generate_mul, generate_div, generate_max, generate_min, generate_pow
from .utils import order_sequence, index_with_tuple, name_operand_tile, name_operand_element, \
    create_intermediate_names_for_loop_indices, create_outer_loop_index_names, \
    create_inner_loop_index_names, format_line, create_tile_size_strings, \
    create_default_tiling, create_default_loop_order, check_operand_names, check_operand_shapes, \
    check_shape_be_broadcast_to_shape, check_simd_width, check_tiling, check_loop_order, check_vmem, \
    create_operation_name


def _generate_simd_unary_element_wise_tensor_codelet(codelet_name: str, input_operand: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], compute_operation_generator_function: Callable[[str, Any], str], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    input_operand_name, input_operand_shape = input_operand
    output_operand_name, output_operand_shape = output_operand
    check_operand_names(input_operand_name, output_operand_name)
    check_operand_shapes(input_operand_shape, output_operand_shape)
    assert len(input_operand_shape) != 0, f"This function is not designed to handle scalars."
    assert len(input_operand_shape) == len(output_operand_shape)
    assert all(in_dim == out_dim for in_dim, out_dim in zip(input_operand_shape, output_operand_shape))

    if dimension_tiling is None:
        dimension_tiling = create_default_tiling(input_operand_shape)
    check_tiling(dimension_tiling, len(input_operand_shape))
    dimension_tile_sizes: tuple[int, ...] = create_tile_size_strings(input_operand_shape, dimension_tiling) 
    if loop_order is None:
        loop_order = create_default_loop_order(input_operand_shape)
    check_loop_order(loop_order, len(input_operand_shape))
    if input_vmem is None:
        input_vmem = "VMEM1"
    check_vmem(input_vmem)
    if output_vmem is None:
        output_vmem = "VMEM2"
    check_vmem(output_vmem)
    check_simd_width(simd_width)

    input_tile_name: str = name_operand_tile(input_operand_name)
    output_tile_name: str = name_operand_tile(output_operand_name)
    input_element_name: str = name_operand_element(input_operand_name)
    output_element_name: str = name_operand_element(output_operand_name)

    intermediate_names_for_loop_indices: tuple[str, ...] = create_intermediate_names_for_loop_indices(input_operand_shape) 
    outer_loop_indices: tuple[str, ...] = create_outer_loop_index_names(intermediate_names_for_loop_indices)
    inner_loop_indices: tuple[str, ...] = create_inner_loop_index_names(intermediate_names_for_loop_indices)

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
    input_element_load: str = generate_load(input_tile_name, inner_loop_indices, input_element_name, (simd_width,), "SIMD")
    compute: str = compute_operation_generator_function(output_element_name, input_element_name)
    output_element_store: str = generate_store(output_element_name, output_tile_name, inner_loop_indices)
    output_tile_store: str = generate_store(output_tile_name, output_operand_name, outer_loop_indices)
    return_statement: str = generate_return((output_operand_name,))

    number_of_tabs: int = 0
    codelet_string: str = format_line(number_of_tabs, header_string) 
    number_of_tabs += 1
    codelet_string += format_line(number_of_tabs, output_alloc)
    for outer_loop_string in outer_loop_strings:
        codelet_string += format_line(number_of_tabs, outer_loop_string)
        number_of_tabs += 1
    codelet_string += format_line(number_of_tabs, output_tile_alloc)
    codelet_string += format_line(number_of_tabs, input_tile_load)
    for inner_loop_string in inner_loop_strings:
        codelet_string += format_line(number_of_tabs, inner_loop_string)
        number_of_tabs += 1
    codelet_string += format_line(number_of_tabs, input_element_load)
    codelet_string += format_line(number_of_tabs, compute)
    codelet_string += format_line(number_of_tabs, output_element_store)
    number_of_tabs -= len(inner_loop_strings)
    codelet_string += format_line(number_of_tabs, output_tile_store)
    number_of_tabs -= len(outer_loop_strings)
    codelet_string += format_line(number_of_tabs, return_statement)
    return codelet_string


def generate_simd_element_wise_relu_codelet(input_operand: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_unary_element_wise_tensor_codelet(f"relu{len(input_operand[1])}d", input_operand, output_operand, generate_relu, simd_width, dimension_tiling, loop_order, input_vmem, output_vmem)


def generate_simd_element_wise_sigmoid_codelet(input_operand: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_unary_element_wise_tensor_codelet(f"sigmoid{len(input_operand[1])}d", input_operand, output_operand, generate_sigmoid, simd_width, dimension_tiling, loop_order, input_vmem, output_vmem)


def generate_simd_element_wise_tanh_codelet(input_operand: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_unary_element_wise_tensor_codelet(f"tanh{len(input_operand[1])}d", input_operand, output_operand, generate_tanh, simd_width, dimension_tiling, loop_order, input_vmem, output_vmem)


def generate_simd_element_wise_sqrt_codelet(input_operand: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_unary_element_wise_tensor_codelet(f"sqrt{len(input_operand[1])}d", input_operand, output_operand, generate_sqrt, simd_width, dimension_tiling, loop_order, input_vmem, output_vmem)


def _generate_simd_binary_element_wise_tensor_codelet(codelet_name: str, input_operand_1: tuple[str, tuple[Union[str, int], ...]], input_operand_2: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], compute_operation_generator_function: Callable[[str, Any], str], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_1_vmem: Optional[str] = None, input_2_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    input_1_operand_name, input_1_operand_shape = input_operand_1
    input_2_operand_name, input_2_operand_shape = input_operand_2
    output_operand_name, output_operand_shape = output_operand
    check_operand_names(input_1_operand_name, input_2_operand_name, output_operand_name)
    check_operand_shapes(input_1_operand_shape, input_2_operand_shape, output_operand_shape)
    codelet_name = create_operation_name(codelet_name, input_1_operand_shape, input_2_operand_shape)
    assert len(input_1_operand_shape) != 0, f"This function is not designed to handle scalars."
    assert len(input_2_operand_shape) != 0, f"This function is not designed to handle scalars."
    if len(input_1_operand_shape) <= len(input_2_operand_shape):
        check_shape_be_broadcast_to_shape(input_1_operand_shape, input_2_operand_shape)
        assert len(input_2_operand_shape) == len(output_operand_shape)
        assert all(in_2_dim == out_dim for in_2_dim, out_dim in zip(input_2_operand_shape, output_operand_shape))
        iterable_dimensions = input_2_operand_shape
    else:
        check_shape_be_broadcast_to_shape(input_2_operand_shape, input_1_operand_shape)
        assert len(input_1_operand_shape) == len(output_operand_shape)
        assert all(in_1_dim == out_dim for in_1_dim, out_dim in zip(input_1_operand_shape, output_operand_shape))
        iterable_dimensions = input_1_operand_shape
    input_1_operand_size_and_offset_indices: tuple[int, ...] = tuple(i for i in range(len(iterable_dimensions) - len(input_1_operand_shape), len(iterable_dimensions)))
    input_2_operand_size_and_offset_indices: tuple[int, ...] = tuple(i for i in range(len(iterable_dimensions) - len(input_2_operand_shape), len(iterable_dimensions)))
    input_1_operand_loop_dependencies: set[int] = set(input_1_operand_size_and_offset_indices) 
    input_2_operand_loop_dependencies: set[int] = set(input_2_operand_size_and_offset_indices)

    if dimension_tiling is None:
        dimension_tiling = create_default_tiling(iterable_dimensions)
    check_tiling(dimension_tiling, len(iterable_dimensions))
    dimension_tile_sizes: tuple[int, ...] = create_tile_size_strings(iterable_dimensions, dimension_tiling)
    if loop_order is None:
        loop_order = create_default_loop_order(iterable_dimensions)
    check_loop_order(loop_order, len(iterable_dimensions))
    if input_1_vmem is None:
        input_1_vmem = "VMEM1"
    check_vmem(input_1_vmem)
    if input_2_vmem is None:
        input_2_vmem = "VMEM2"
    check_vmem(input_2_vmem)
    if output_vmem is None:
        output_vmem = "VMEM1"
    check_vmem(output_vmem)
    check_simd_width(simd_width)

    input_1_tile_name: str = name_operand_tile(input_1_operand_name)
    input_2_tile_name: str = name_operand_tile(input_2_operand_name)
    output_tile_name: str = name_operand_tile(output_operand_name)
    input_1_element_name: str = name_operand_element(input_1_operand_name)
    input_2_element_name: str = name_operand_element(input_2_operand_name)
    output_element_name: str = name_operand_element(output_operand_name)

    intermediate_names_for_loop_indices: tuple[str, ...] = create_intermediate_names_for_loop_indices(iterable_dimensions)
    outer_loop_indices: tuple[str, ...] = create_outer_loop_index_names(intermediate_names_for_loop_indices)
    inner_loop_indices: tuple[str, ...] = create_inner_loop_index_names(intermediate_names_for_loop_indices)

    dimension_in_order: tuple[Union[str, int], ...] = order_sequence(iterable_dimensions, loop_order)
    dimension_tiling_in_order: tuple[int, ...] = order_sequence(dimension_tiling, loop_order)
    outer_loop_indices_in_order: tuple[str, ...] = order_sequence(outer_loop_indices, loop_order)
    inner_loop_indices_in_order: tuple[str, ...] = order_sequence(inner_loop_indices, loop_order)
    inner_loop_stride_in_order: tuple[Union[str, int], ...] = tuple(simd_width if loop_number == (len(iterable_dimensions) - 1) else 1 for loop_number in loop_order)

    header_string: str = generate_header(codelet_name, ((input_1_operand_name, "i32", input_1_operand_shape, "DRAM"), (input_2_operand_name, "i32", input_2_operand_shape, "DRAM")), ())
    output_alloc: str = generate_allocation(output_operand_name, output_operand_shape, "DRAM", "i32")
    outer_loop_strings: tuple[str, ...] = tuple(generate_outer_for_loop(loop_index, number_of_tiles, dim_size) for loop_index, dim_size, number_of_tiles in zip(outer_loop_indices_in_order, dimension_in_order, dimension_tiling_in_order))
    output_tile_alloc: str = generate_allocation(output_tile_name, dimension_tile_sizes, output_vmem, "i32")
    input_1_tile_load: str = generate_load(input_1_operand_name, index_with_tuple(outer_loop_indices, input_1_operand_size_and_offset_indices), input_1_tile_name, index_with_tuple(dimension_tile_sizes, input_1_operand_size_and_offset_indices), input_1_vmem)
    input_2_tile_load: str = generate_load(input_2_operand_name, index_with_tuple(outer_loop_indices, input_2_operand_size_and_offset_indices), input_2_tile_name, index_with_tuple(dimension_tile_sizes, input_2_operand_size_and_offset_indices), input_2_vmem)
    inner_loop_strings: tuple[str, ...] = tuple(generate_inner_for_loop(loop_index, dim_size, number_of_tiles, loop_stride) for loop_index, dim_size, number_of_tiles, loop_stride in zip(inner_loop_indices_in_order, dimension_in_order, dimension_tiling_in_order, inner_loop_stride_in_order))
    input_1_element_load: str = generate_load(input_1_tile_name, index_with_tuple(inner_loop_indices, input_1_operand_size_and_offset_indices), input_1_element_name, (simd_width,), "SIMD")
    input_2_element_load: str = generate_load(input_2_tile_name, index_with_tuple(inner_loop_indices, input_2_operand_size_and_offset_indices), input_2_element_name, (simd_width,), "SIMD")
    compute: str = compute_operation_generator_function(output_element_name, (input_1_element_name, input_2_element_name))
    output_element_store: str = generate_store(output_element_name, output_tile_name, inner_loop_indices)
    output_tile_store: str = generate_store(output_tile_name, output_operand_name, outer_loop_indices)
    return_statement: str = generate_return((output_operand_name,))

    number_of_tabs: int = 0
    codelet_string: str = format_line(number_of_tabs, header_string)
    number_of_tabs += 1
    codelet_string += format_line(number_of_tabs, output_alloc)
    codelet_string += create_loops_with_dependent_statements(
        outer_loop_strings, loop_order, 
        (
            (input_1_tile_load, input_1_operand_loop_dependencies), 
            (input_2_tile_load, input_2_operand_loop_dependencies), 
        ),
        current_number_of_tabs=number_of_tabs
    )
    number_of_tabs += len(outer_loop_strings)
    codelet_string += format_line(number_of_tabs, output_tile_alloc)
    codelet_string += create_loops_with_dependent_statements(
        inner_loop_strings, loop_order,
        (
            (input_1_element_load, input_1_operand_loop_dependencies),
            (input_2_element_load, input_2_operand_loop_dependencies),
        ),
        current_number_of_tabs=number_of_tabs
    ) 
    number_of_tabs += len(inner_loop_strings)
    codelet_string += format_line(number_of_tabs, compute)
    codelet_string += format_line(number_of_tabs, output_element_store)
    number_of_tabs -= len(inner_loop_strings)
    codelet_string += format_line(number_of_tabs, output_tile_store)
    number_of_tabs -= len(outer_loop_strings)
    codelet_string += format_line(number_of_tabs, return_statement)
    return codelet_string


def generate_simd_element_wise_add_codelet(input_operand_1: tuple[str, tuple[Union[str, int], ...]], input_operand_2: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_1_vmem: Optional[str] = None, input_2_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_binary_element_wise_tensor_codelet("add", input_operand_1, input_operand_2, output_operand, generate_add, simd_width, dimension_tiling, loop_order, input_1_vmem, input_2_vmem, output_vmem)


def generate_simd_element_wise_sub_codelet(input_operand_1: tuple[str, tuple[Union[str, int], ...]], input_operand_2: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_1_vmem: Optional[str] = None, input_2_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_binary_element_wise_tensor_codelet("sub", input_operand_1, input_operand_2, output_operand, generate_sub, simd_width, dimension_tiling, loop_order, input_1_vmem, input_2_vmem, output_vmem)


def generate_simd_element_wise_mul_codelet(input_operand_1: tuple[str, tuple[Union[str, int], ...]], input_operand_2: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_1_vmem: Optional[str] = None, input_2_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_binary_element_wise_tensor_codelet("mul", input_operand_1, input_operand_2, output_operand, generate_mul, simd_width, dimension_tiling, loop_order, input_1_vmem, input_2_vmem, output_vmem)


def generate_simd_element_wise_div_codelet(input_operand_1: tuple[str, tuple[Union[str, int], ...]], input_operand_2: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_1_vmem: Optional[str] = None, input_2_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_binary_element_wise_tensor_codelet("div", input_operand_1, input_operand_2, output_operand, generate_div, simd_width, dimension_tiling, loop_order, input_1_vmem, input_2_vmem, output_vmem)


def generate_simd_element_wise_max_codelet(input_operand_1: tuple[str, tuple[Union[str, int], ...]], input_operand_2: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_1_vmem: Optional[str] = None, input_2_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_binary_element_wise_tensor_codelet("max", input_operand_1, input_operand_2, output_operand, generate_max, simd_width, dimension_tiling, loop_order, input_1_vmem, input_2_vmem, output_vmem)


def generate_simd_element_wise_min_codelet(input_operand_1: tuple[str, tuple[Union[str, int], ...]], input_operand_2: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_1_vmem: Optional[str] = None, input_2_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_binary_element_wise_tensor_codelet("min", input_operand_1, input_operand_2, output_operand, generate_min, simd_width, dimension_tiling, loop_order, input_1_vmem, input_2_vmem, output_vmem)


def generate_simd_element_wise_pow_codelet(input_operand_1: tuple[str, tuple[Union[str, int], ...]], input_operand_2: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_1_vmem: Optional[str] = None, input_2_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    return _generate_simd_binary_element_wise_tensor_codelet("pow", input_operand_1, input_operand_2, output_operand, generate_pow, simd_width, dimension_tiling, loop_order, input_1_vmem, input_2_vmem, output_vmem)
