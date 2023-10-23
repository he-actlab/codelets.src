from typing import Any, Callable, Optional, Union
from .header_string_generator import generate_header
from .allocation_string_generator import generate_allocation
from .loop_string_generator import generate_inner_for_loop, generate_outer_for_loop, create_loops_with_dependent_statements
from .load_string_generator import generate_load
from .store_string_generator import generate_store
from .return_string_generator import generate_return
from .compute_string_generator import generate_mvmul
from .utils import order_sequence, index_with_tuple, name_operand_tile, name_operand_element, \
    create_intermediate_names_for_loop_indices, create_outer_loop_index_names, \
    create_inner_loop_index_names, format_line, create_tile_size_strings, \
    create_default_tiling, create_default_loop_order, check_operand_names, check_operand_shapes, \
    check_shape_be_broadcast_to_shape, check_pe_array_dimensions, check_tiling, check_loop_order, \
    create_operation_name


def _generate_systolic_array_matmul_codelet(codelet_name: str, input_1_operand: tuple[str, tuple[Union[str, int], ...]], input_2_operand: tuple[str, tuple[Union[str, int], ...]], bias_operand: Optional[tuple[str, tuple[Union[str, int], ...]]], output_operand: tuple[str, tuple[Union[str, int], ...]], pe_array_width: int, pe_array_height: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None) -> str:
    input_1_operand_name, input_1_operand_shape = input_1_operand
    input_2_operand_name, input_2_operand_shape = input_2_operand
    if bias_operand is not None:
        bias_operand_name, bias_operand_shape = bias_operand
    else:
        bias_operand_name, bias_operand_shape = None, None
    output_operand_name, output_operand_shape = output_operand
    check_operand_names(input_1_operand_name, input_2_operand_name, bias_operand_name, output_operand_name)
    check_operand_shapes(input_1_operand_shape, input_2_operand_shape, bias_operand_shape, output_operand_shape)
    # TODO: Support 1D tensors as well.
    assert len(input_1_operand_shape) >= 2
    assert len(input_2_operand_shape) >= 2
    assert len(output_operand_shape) >= 2
    if bias_operand_shape is not None:
        assert len(bias_operand_shape) == 1
 
    input_1_operand_stack_shape: tuple[Union[str, int], ...] = input_1_operand_shape[:-2]
    input_1_operand_matrix_shape: tuple[Union[str, int], tuple[Union[str, int]]] = input_1_operand_shape[-2:]
    input_2_operand_stack_shape: tuple[Union[str, int], ...] = input_2_operand_shape[:-2]
    input_2_operand_matrix_shape: tuple[Union[str, int], tuple[Union[str, int]]] = input_2_operand_shape[-2:]
    output_operand_stack_shape: tuple[Union[str, int], ...] = output_operand_shape[:-2]
    output_operand_matrix_shape: tuple[Union[str, int], tuple[Union[str, int]]] = output_operand_shape[-2:]
    assert input_1_operand_matrix_shape[1] == input_2_operand_matrix_shape[0]
    assert input_1_operand_matrix_shape[0] == output_operand_matrix_shape[0]
    assert input_2_operand_matrix_shape[1] == output_operand_matrix_shape[1]
    if bias_operand_shape is not None:
        assert bias_operand_shape[0] == output_operand_matrix_shape[1]
    codelet_name = create_operation_name(codelet_name if bias_operand_name is None else codelet_name + "_bias", input_1_operand_shape, input_2_operand_shape)

    if len(input_1_operand_stack_shape) <= len(input_2_operand_stack_shape):
        check_shape_be_broadcast_to_shape(input_1_operand_stack_shape, input_2_operand_stack_shape)
        assert len(input_2_operand_stack_shape) == len(output_operand_stack_shape)
        assert all(in_2_dim == out_dim for in_2_dim, out_dim in zip(input_2_operand_stack_shape, output_operand_stack_shape))
        iterable_dimensions = input_2_operand_stack_shape
    else:
        check_shape_be_broadcast_to_shape(input_2_operand_stack_shape, input_1_operand_stack_shape)
        assert len(input_1_operand_stack_shape) == len(output_operand_stack_shape)
        assert all(in_1_dim == out_dim for in_1_dim, out_dim in zip(input_1_operand_stack_shape, output_operand_stack_shape))
        iterable_dimensions = input_1_operand_stack_shape
    input_1_operand_size_and_offset_indices: tuple[int, ...] = tuple(i for i in range(len(iterable_dimensions) - len(input_1_operand_stack_shape), len(iterable_dimensions))) + (len(iterable_dimensions), len(iterable_dimensions) + 1)
    input_2_operand_size_and_offset_indices: tuple[int, ...] = tuple(i for i in range(len(iterable_dimensions) - len(input_2_operand_stack_shape), len(iterable_dimensions))) + (len(iterable_dimensions) + 1, len(iterable_dimensions) + 2)
    output_operand_size_and_offset_indices: tuple[int, ...] = tuple(i for i in range(len(iterable_dimensions))) + (len(iterable_dimensions), len(iterable_dimensions) + 2)
    input_1_operand_loop_dependencies: set[int] = set(input_1_operand_size_and_offset_indices) 
    input_2_operand_loop_dependencies: set[int] = set(input_2_operand_size_and_offset_indices)
    output_operand_loop_dependencies: set[int] = set(output_operand_size_and_offset_indices)
    iterable_dimensions += input_1_operand_matrix_shape + (input_2_operand_matrix_shape[1],)

    if dimension_tiling is None:
        dimension_tiling = create_default_tiling(iterable_dimensions)
    check_tiling(dimension_tiling, len(iterable_dimensions))
    dimension_tile_sizes: tuple[int, ...] = create_tile_size_strings(iterable_dimensions, dimension_tiling)
    if loop_order is None:
        loop_order = create_default_loop_order(iterable_dimensions)
    check_loop_order(loop_order, len(iterable_dimensions))
    check_pe_array_dimensions(pe_array_width, pe_array_height)

    input_1_tile_name: str = name_operand_tile(input_1_operand_name)
    input_2_tile_name: str = name_operand_tile(input_2_operand_name)
    bias_tile_name: Optional[str] = name_operand_tile(bias_operand_name) if bias_operand_name is not None else None
    output_tile_name: str = name_operand_tile(output_operand_name)
    input_1_element_name: str = name_operand_element(input_1_operand_name)
    input_2_element_name: str = name_operand_element(input_2_operand_name)
    bias_element_name: Optional[str] = name_operand_element(bias_operand_name) if bias_operand_name is not None else None
    output_element_name: str = name_operand_element(output_operand_name)
    compute_output_element_name: str = name_operand_element("compute_output")

    intermediate_names_for_loop_indices: tuple[str, ...] = create_intermediate_names_for_loop_indices(iterable_dimensions)
    outer_loop_indices: tuple[str, ...] = create_outer_loop_index_names(intermediate_names_for_loop_indices)
    inner_loop_indices: tuple[str, ...] = create_inner_loop_index_names(intermediate_names_for_loop_indices)

    dimension_in_order: tuple[Union[str, int], ...] = order_sequence(iterable_dimensions, loop_order)
    dimension_tiling_in_order: tuple[int, ...] = order_sequence(dimension_tiling, loop_order)
    outer_loop_indices_in_order: tuple[str, ...] = order_sequence(outer_loop_indices, loop_order)
    inner_loop_indices_in_order: tuple[str, ...] = order_sequence(inner_loop_indices, loop_order)
    inner_loop_stride_in_order: list[Union[str, int]] = []
    for i in range(len(dimension_in_order)):
        if i == (len(iterable_dimensions) - 1):
            inner_loop_stride_in_order.append(pe_array_width)
        elif i == (len(iterable_dimensions) - 2):
            inner_loop_stride_in_order.append(pe_array_height)
        else:
            inner_loop_stride_in_order.append(1)
    inner_loop_stride_in_order = tuple(inner_loop_stride_in_order)

    codelet_inputs: tuple[tuple[str, str, tuple[Union[str, int], ...], str], ...] = tuple(
        filter(
            lambda t: t[0] is not None, 
            ((input_1_operand_name, "i8", input_1_operand_shape, "DRAM"), (input_2_operand_name, "i8", input_2_operand_shape, "DRAM"), (bias_operand_name, "i32", bias_operand_shape, "DRAM"),)
        )
    )
    header_string: str = generate_header(codelet_name, codelet_inputs, ())
    output_alloc: str = generate_allocation(output_operand_name, output_operand_shape, "DRAM", "i32")
    outer_loop_strings: tuple[str, ...] = tuple(generate_outer_for_loop(loop_index, number_of_tiles, dim_size) for loop_index, dim_size, number_of_tiles in zip(outer_loop_indices_in_order, dimension_in_order, dimension_tiling_in_order))
    input_1_tile_load: str = generate_load(input_1_operand_name, index_with_tuple(outer_loop_indices, input_1_operand_size_and_offset_indices), input_1_tile_name, index_with_tuple(dimension_tile_sizes, input_1_operand_size_and_offset_indices), "IBUF")
    input_2_tile_load: str = generate_load(input_2_operand_name, index_with_tuple(outer_loop_indices, input_2_operand_size_and_offset_indices), input_2_tile_name, index_with_tuple(dimension_tile_sizes, input_2_operand_size_and_offset_indices), "WBUF")
    bias_tile_load: Optional[str] = None
    if bias_operand_name is not None:
        bias_tile_load = generate_load(bias_operand_name, (outer_loop_indices[-1],), bias_tile_name, (dimension_tile_sizes[-1],), "BBUF")
    output_tile_load: str = generate_load(output_operand_name, index_with_tuple(outer_loop_indices, output_operand_size_and_offset_indices), output_tile_name, index_with_tuple(dimension_tile_sizes, output_operand_size_and_offset_indices), "OBUF")
    inner_loop_strings: tuple[str, ...] = tuple(generate_inner_for_loop(loop_index, dim_size, number_of_tiles, loop_stride) for loop_index, dim_size, number_of_tiles, loop_stride in zip(inner_loop_indices_in_order, dimension_in_order, dimension_tiling_in_order, inner_loop_stride_in_order))
    input_1_element_load: str = generate_load(input_1_tile_name, index_with_tuple(inner_loop_indices, input_1_operand_size_and_offset_indices), input_1_element_name, (1, pe_array_width), "PE_ARRAY")
    input_2_element_load: str = generate_load(input_2_tile_name, index_with_tuple(inner_loop_indices, input_2_operand_size_and_offset_indices), input_2_element_name, (pe_array_width, pe_array_height), "PE_ARRAY")
    bias_element_load: Optional[str] = None
    if bias_operand_name is not None:
        bias_element_load = generate_load(bias_tile_name, (inner_loop_indices[-1],), bias_element_name, (1, pe_array_height), "PE_ARRAY")
    output_element_load: str = generate_load(output_tile_name, index_with_tuple(inner_loop_indices, output_operand_size_and_offset_indices), output_element_name, (1, pe_array_height), "PE_ARRAY")
    compute_arguments: tuple[str, ...] = tuple(
        filter(
            lambda s: s is not None,
            (input_1_element_name, input_2_element_name, bias_element_name, output_element_name)
        )
    )
    compute: str = generate_mvmul(compute_output_element_name, compute_arguments)
    output_element_store: str = generate_store(compute_output_element_name, output_tile_name, index_with_tuple(inner_loop_indices, output_operand_size_and_offset_indices))
    output_tile_store: str = generate_store(output_tile_name, output_operand_name, index_with_tuple(outer_loop_indices, output_operand_size_and_offset_indices))
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
            (bias_tile_load, set((output_operand_size_and_offset_indices[-1],))),
            (output_tile_load, output_operand_loop_dependencies),
        ),
        current_number_of_tabs=number_of_tabs
    )
    number_of_tabs += len(outer_loop_strings)
    codelet_string += create_loops_with_dependent_statements(
        inner_loop_strings, loop_order,
        (
            (input_1_element_load, input_1_operand_loop_dependencies),
            (input_2_element_load, input_2_operand_loop_dependencies),
            (bias_element_load, set((output_operand_size_and_offset_indices[-1],))),
            (output_element_load, output_operand_loop_dependencies),
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


def generate_systolic_array_matmul_codelet(input_1_operand: tuple[str, tuple[Union[str, int], ...]], input_2_operand: tuple[str, tuple[Union[str, int], ...]], bias_operand: Optional[tuple[str, tuple[Union[str, int], ...]]], output_operand: tuple[str, tuple[Union[str, int], ...]], pe_array_width: int, pe_array_height: int, dimension_tiling: Optional[tuple[int, ...]] = None, loop_order: Optional[tuple[int, ...]] = None) -> str:
    return _generate_systolic_array_matmul_codelet("matmul", input_1_operand, input_2_operand, bias_operand, output_operand, pe_array_width, pe_array_height, dimension_tiling, loop_order)
