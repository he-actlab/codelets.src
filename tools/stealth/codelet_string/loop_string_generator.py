from typing import Optional, Union


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
