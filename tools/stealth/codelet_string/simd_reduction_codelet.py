from typing import Any, Callable, Optional, Union
from .header_string_generator import generate_header
from .allocation_string_generator import generate_allocation
from .loop_string_generator import generate_inner_for_loop, generate_outer_for_loop, create_loops_with_dependent_statements
from .load_string_generator import generate_load
from .store_string_generator import generate_store
from .return_string_generator import generate_return
from .compute_string_generator import generate_add, generate_sub, generate_mul, generate_div, generate_max, generate_min, generate_pow
from .utils import order_sequence, index_with_tuple, name_operand_tile, name_operand_element, \
    create_intermediate_names_for_loop_indices, create_outer_loop_index_names, \
    create_inner_loop_index_names, format_line, create_tile_size_strings, \
    create_default_tiling, create_default_loop_order, check_operand_names, check_operand_shapes, \
    check_shape_be_broadcast_to_shape, check_simd_width, check_tiling, check_loop_order, check_vmem, \
    create_operation_name, \
    floating_point_to_fixed_point


def _generate_indices_for_reduced_operand(indices: tuple[Union[str, int], ...], reduced_axis: int) -> tuple[Union[str, int], ...]:
    return tuple(index if i != reduced_axis else 0 for i, index in enumerate(indices))


def _generate_size_for_reduced_operand(size: tuple[Union[str, int], ...], reduced_axis: int) -> tuple[Union[str, int], ...]:
    return tuple(size[i] if i != reduced_axis else 1 for i in range(len(size)))


def _generate_simd_reduce_mean_codelet(codelet_name: str, input_operand: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], reduced_axis: tuple[int, ...], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_vmem: Optional[str] = None, partial_sum_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    input_operand_name, input_operand_shape = input_operand
    output_operand_name, output_operand_shape = output_operand
    check_operand_names(input_operand_name, output_operand_name)
    check_operand_shapes(input_operand_shape, output_operand_shape)
    codelet_name = create_operation_name(codelet_name, input_operand_shape)
    assert len(input_operand_shape) != 0, f"This function is not designed to handle scalars."
    assert len(input_operand_shape) == len(output_operand_shape), f"The input and output operands must have the same number of dimensions."
    for i, (input_dim, output_dim) in enumerate(zip(input_operand_shape, output_operand_shape)):
        if i in reduced_axis:
            assert output_dim == 1, f"The output dimension along the reduced axis must be 1."
        else:
            assert input_dim == output_dim, f"The input and output dimensions must be the same except along the reduced axis."
    iterable_dimensions = input_operand_shape
    reduced_dimension_names: tuple[str, ...] = index_with_tuple(input_operand_shape, reduced_axis)
    iterable_dimensions_for_mul: tuple[str, ...] = tuple(filter(lambda d: d not in reduced_dimension_names, iterable_dimensions))

    input_operand_size_and_offset_indices: tuple[int, ...] = tuple(i for i in range(len(iterable_dimensions)))
    input_operand_loop_dependencies: set[int] = set(input_operand_size_and_offset_indices) 
    output_operand_loop_dependencies: set[int] = set(filter(lambda i: i not in reduced_axis, input_operand_size_and_offset_indices))

    if dimension_tiling is None:
        dimension_tiling = create_default_tiling(iterable_dimensions)
    check_tiling(dimension_tiling, len(iterable_dimensions))
    dimension_tile_sizes: tuple[int, ...] = create_tile_size_strings(iterable_dimensions, dimension_tiling)
    if loop_order is None:
        loop_order = create_default_loop_order(iterable_dimensions)
    check_loop_order(loop_order, len(iterable_dimensions))
    if input_vmem is None:
        input_vmem = "VMEM1"
    check_vmem(input_vmem)
    if partial_sum_vmem is None:
        partial_sum_vmem = "VMEM1"
    check_vmem(partial_sum_vmem)
    if output_vmem is None:
        output_vmem = "VMEM1"
    check_vmem(output_vmem)
    check_simd_width(simd_width)

    input_tile_name: str = name_operand_tile(input_operand_name)
    partial_sum_tile_name: str = name_operand_tile(output_operand_name + "_sum")
    input_element_name: str = name_operand_element(input_operand_name)
    partial_sum_element_name: str = name_operand_element(output_operand_name + "partial_sum")
    compute_add_output_element_name: str = name_operand_element(output_operand_name + "_add")
    output_tile_name: str = name_operand_tile(output_operand_name)
    output_element_name: str = name_operand_element(output_operand_name)
    compute_mul_output_element_name: str = name_operand_element(output_operand_name + "_mul")

    intermediate_names_for_sum_loop_indices: tuple[str, ...] = tuple(map(lambda name: f"{name}_sum", create_intermediate_names_for_loop_indices(iterable_dimensions)))
    intermediate_names_for_mul_loop_indices: tuple[str, ...] = tuple(map(lambda name: f"{name}_mul", create_intermediate_names_for_loop_indices(iterable_dimensions)))
    sum_outer_loop_indices: tuple[str, ...] = create_outer_loop_index_names(intermediate_names_for_sum_loop_indices)
    sum_inner_loop_indices: tuple[str, ...] = create_inner_loop_index_names(intermediate_names_for_sum_loop_indices)
    mul_outer_loop_indices: tuple[str, ...] = create_outer_loop_index_names(intermediate_names_for_mul_loop_indices)
    mul_inner_loop_indices: tuple[str, ...] = create_inner_loop_index_names(intermediate_names_for_mul_loop_indices)

    sum_output_operand_offset_indices: tuple[Union[str, int], ...] = _generate_indices_for_reduced_operand(sum_outer_loop_indices, reduced_axis)
    sum_output_tile_offset_indices: tuple[Union[str, int], ...] = _generate_indices_for_reduced_operand(sum_inner_loop_indices, reduced_axis)
    mul_output_operand_offset_indices: tuple[Union[str, int], ...] = _generate_indices_for_reduced_operand(mul_outer_loop_indices, reduced_axis)
    mul_output_tile_offset_indices: tuple[Union[str, int], ...] = _generate_indices_for_reduced_operand(mul_inner_loop_indices, reduced_axis)

    output_tile_size: tuple[Union[str, int], ...] = _generate_size_for_reduced_operand(dimension_tile_sizes, reduced_axis)

    dimension_in_order: tuple[Union[str, int], ...] = order_sequence(iterable_dimensions, loop_order)
    dimension_tiling_in_order: tuple[int, ...] = order_sequence(dimension_tiling, loop_order)
    sum_outer_loop_indices_in_order: tuple[str, ...] = order_sequence(sum_outer_loop_indices, loop_order)
    sum_inner_loop_indices_in_order: tuple[str, ...] = order_sequence(sum_inner_loop_indices, loop_order)
    sum_inner_loop_stride_in_order: tuple[Union[str, int], ...] = tuple(simd_width if loop_number == (len(iterable_dimensions) - 1) else 1 for loop_number in loop_order)

    header_string: str = generate_header(codelet_name, ((input_operand_name, "i32", input_operand_shape, "DRAM"),), ())
    output_alloc: str = generate_allocation(output_operand_name, output_operand_shape, "DRAM", "i32")
    sum_outer_loop_strings: tuple[str, ...] = tuple(generate_outer_for_loop(loop_index, number_of_tiles, dim_size) for loop_index, dim_size, number_of_tiles in zip(sum_outer_loop_indices_in_order, dimension_in_order, dimension_tiling_in_order))
    input_tile_load = generate_load(input_operand_name, sum_outer_loop_indices, input_tile_name, dimension_tile_sizes, input_vmem)
    partial_sum_tile_load = generate_load(output_operand_name, sum_output_operand_offset_indices, partial_sum_tile_name, output_tile_size, partial_sum_vmem)
    sum_inner_loop_strings: tuple[str, ...] = tuple(generate_inner_for_loop(loop_index, dim_size, number_of_tiles, loop_stride) for loop_index, dim_size, number_of_tiles, loop_stride in zip(sum_inner_loop_indices_in_order, dimension_in_order, dimension_tiling_in_order, sum_inner_loop_stride_in_order))
    input_element_load: str = generate_load(input_tile_name, sum_inner_loop_indices, input_element_name, (simd_width,), "SIMD")
    partial_sum_element_load: str = generate_load(partial_sum_tile_name, sum_output_tile_offset_indices, compute_add_output_element_name, (simd_width,), "SIMD")
    add_compute: str = generate_add(partial_sum_element_name, (input_element_name, output_element_name))
    partial_sum_element_store = generate_store(partial_sum_element_name, partial_sum_tile_name, sum_output_tile_offset_indices)
    partial_sum_tile_store = generate_store(partial_sum_tile_name, output_operand_name, sum_output_operand_offset_indices)
    mul_outer_loop_strings: tuple[str, ...] = tuple(generate_outer_for_loop(loop_index, number_of_tiles, dim_size) for loop_index, dim_size, number_of_tiles in zip(mul_outer_loop_indices, dimension_in_order, dimension_tiling_in_order))
    output_tile_load: str = generate_load(output_operand_name, mul_output_operand_offset_indices, output_tile_name, output_tile_size, output_vmem)
    mul_inner_loop_strings: tuple[str, ...] = tuple(generate_inner_for_loop(loop_index, dim_size, number_of_tiles, loop_stride) for loop_index, dim_size, number_of_tiles, loop_stride in zip(mul_inner_loop_indices, dimension_in_order, dimension_tiling_in_order, sum_inner_loop_stride_in_order))
    output_element_load: str = generate_load(output_tile_name, mul_output_tile_offset_indices, compute_add_output_element_name, (simd_width,), "SIMD")
    mul_compute: str = generate_div(compute_mul_output_element_name, (compute_add_output_element_name, f"{' * '.join(reduced_dimension_names)} * 2 ** 16"))
    output_element_store: str = generate_store(compute_mul_output_element_name, output_tile_name, mul_output_tile_offset_indices)
    output_tile_store: str = generate_store(output_tile_name, output_operand_name, mul_output_operand_offset_indices)
    return_statement: str = generate_return((output_operand_name,))

    number_of_tabs: int = 0
    codelet_string: str = format_line(number_of_tabs, header_string)
    number_of_tabs += 1
    codelet_string += format_line(number_of_tabs, output_alloc)
    codelet_string += create_loops_with_dependent_statements(
        sum_outer_loop_strings, loop_order, 
        (
            (input_tile_load, input_operand_loop_dependencies),
            (partial_sum_tile_load, output_operand_loop_dependencies), 
        ),
        current_number_of_tabs=number_of_tabs
    )
    number_of_tabs += len(sum_outer_loop_strings)
    codelet_string += create_loops_with_dependent_statements(
        sum_inner_loop_strings, loop_order,
        (
            (input_element_load, input_operand_loop_dependencies),
            (partial_sum_element_load, input_operand_loop_dependencies),
        ),
        current_number_of_tabs=number_of_tabs
    ) 
    number_of_tabs += len(sum_inner_loop_strings)
    codelet_string += format_line(number_of_tabs, add_compute)
    codelet_string += format_line(number_of_tabs, partial_sum_element_store)
    number_of_tabs -= len(sum_inner_loop_strings)
    codelet_string += format_line(number_of_tabs, partial_sum_tile_store)
    number_of_tabs -= len(sum_outer_loop_strings)

    # codelet_string += create_loops_with_dependent_statements(
    #     mul_outer_loop_strings, loop_order, 
    #     (
    #         (input_tile_load, input_operand_loop_dependencies),
    #         (partial_sum_tile_load, output_operand_loop_dependencies), 
    #     ),
    #     current_number_of_tabs=number_of_tabs
    # )
    # number_of_tabs += len(sum_outer_loop_strings)
    # codelet_string += create_loops_with_dependent_statements(
    #     sum_inner_loop_strings, loop_order,
    #     (
    #         (input_element_load, input_operand_loop_dependencies),
    #         (partial_sum_element_load, input_operand_loop_dependencies),
    #     ),
    #     current_number_of_tabs=number_of_tabs
    # ) 
    # number_of_tabs += len(sum_inner_loop_strings)
    # codelet_string += format_line(number_of_tabs, add_compute)
    # codelet_string += format_line(number_of_tabs, partial_sum_element_store)
    # number_of_tabs -= len(sum_inner_loop_strings)
    # codelet_string += format_line(number_of_tabs, partial_sum_tile_store)
    # number_of_tabs -= len(sum_outer_loop_strings)
    # codelet_string += format_line(number_of_tabs, return_statement)
    return codelet_string


def generate_simd_reduce_mean_codelet(input_operand: tuple[str, tuple[Union[str, int], ...]], output_operand: tuple[str, tuple[Union[str, int], ...]], reduction_axis: tuple[int, ...], simd_width: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None, input_vmem: Optional[str] = None, partial_sum_vmem: Optional[str] = None, output_vmem: Optional[str] = None) -> str:
    assert len(input_operand[1]) - 1 not in reduction_axis, f"Cannot reduce the outermost dimension."
    return _generate_simd_reduce_mean_codelet("reduce_mean", input_operand, output_operand, reduction_axis, simd_width, dimension_tiling, loop_order, input_vmem, partial_sum_vmem, output_vmem)
