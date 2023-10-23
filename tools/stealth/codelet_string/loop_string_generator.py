from typing import Optional, Union
from .utils import format_line


def _generate_for_loop(loop_index_name: str, number_of_iterations: str, stride: str) -> str:
    return f"for {loop_index_name} in loop({number_of_iterations}, {stride}):"


def generate_outer_for_loop(loop_index_name: str, number_of_tiles: int, dimension_size: Union[str, int]) -> str:
    if isinstance(dimension_size, int):
        dimension_size = str(dimension_size)
    return _generate_for_loop(loop_index_name, str(number_of_tiles), f"{dimension_size} // {number_of_tiles}") 


def generate_inner_for_loop(loop_index_name: str, dimension_size: Union[str, int], number_of_tiles: int, stride: Union[str, int]) -> str:
    if isinstance(dimension_size, int):
        dimension_size = str(dimension_size)
    if isinstance(stride, int):
        stride = str(stride)
    return _generate_for_loop(loop_index_name, f"({dimension_size} // {number_of_tiles}) // {stride}", stride)


def create_loops_with_dependent_statements(loop_strings_in_order: tuple[str], loop_order: tuple[int], statements_with_dependencies: tuple[tuple[Optional[str], set[int]]], current_number_of_tabs: int = 0) -> str:
    ret: str = ""
    loop_numbers_iterated: set[int] = set()
    indices_of_statements_placed: set[int] = set()
    for loop_number, loop_string in zip(loop_order, loop_strings_in_order):
        ret += format_line(current_number_of_tabs, loop_string)
        current_number_of_tabs += 1
        loop_numbers_iterated.add(loop_number)
        for statement_index, (statement, dependencies) in enumerate(statements_with_dependencies):
            if statement is None:
                continue
            if statement_index not in indices_of_statements_placed and loop_numbers_iterated.union(dependencies) == loop_numbers_iterated:
                ret += format_line(current_number_of_tabs, statement)
                indices_of_statements_placed.add(statement_index)
    return ret
