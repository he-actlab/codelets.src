import argparse
import json
import string
import os
import shutil
from typing import Callable, Optional
import numpy as np
import tqdm
import functools
from threading import Thread, Lock
from queue import Queue
from stealth.codelet_string import *
from stealth.compile import compile
from stealth.stealth_codelet import *
from stealth.parse import parse_stealth_codelet
from stealth.unit_test import *
from stealth.run import run_layer, get_relevant_simulator_outputs
from stealth.search_space import *
from stealth.searcher import *
from stealth.utils import repeat
from codelets.examples.genesys import load_config


def codelet_string_to_codelet(codelet_string: str, verbose: bool = False) -> StealthCodelet:
    parsed_codelet = parse_stealth_codelet(codelet_string, verbose=verbose)
    return build_codelet_from_parse_tree(parsed_codelet, codelet_string, verbose=verbose)


def get_shape_from_number_of_dimensions(number_of_dimensions: int) -> tuple[str, ...]:
    assert number_of_dimensions > 0, "Number of dimensions must be positive."

    NUMBER_OF_DIMENSIONS_TO_SHAPE = {
        3: ("N", "L", "H"),
        4: ("N", "H", "W", "C"),
    }

    if number_of_dimensions in NUMBER_OF_DIMENSIONS_TO_SHAPE:
        return NUMBER_OF_DIMENSIONS_TO_SHAPE[number_of_dimensions]
    elif number_of_dimensions > 26:
        raise ValueError("Number of dimensions is too large.")
    else:
        return tuple(string.ascii_uppercase[:number_of_dimensions])
    

def get_int8_tensor(shape: tuple[int, ...]) -> np.ndarray:
    return np.random.randint(-2**7, 2**7 - 1, shape, dtype=np.int8)


def get_int32_tensor(shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) == 0:
        return np.random.randint(-2**31, 2**31 - 1, dtype=np.int32)
    else:
        return np.random.randint(-2**31, 2**31 - 1, shape, dtype=np.int32)


def get_random_vmem() -> str:
    return f"VMEM{np.random.randint(1, 2)}"


def get_tiling(search_space_point: tuple[SearchSpacePoint, ...], iterable_dimensions: tuple[int, ...]) -> tuple[int, ...]:
    tiling_points: tuple[SearchSpacePoint, ...] = search_space_point[:len(iterable_dimensions)]
    assert all(isinstance(tile_space_point, TileSpacePoint) for tile_space_point in tiling_points)
    dimension_tiling: tuple[int, ...] = tuple(tile_space_point.number_of_tiles for tile_space_point in tiling_points)
    return dimension_tiling


def get_unit_test_output(unit_test: UnitTest, stealth_codelet: StealthCodelet, operand_dim_sizes: dict[str, int], config_path: str, config: dict[str, Union[int, str, bool]], array_n: int, array_m: int, thread_id: int, lock: Lock, verify_output: bool = False) -> dict:
    input_dimensions: dict[str, tuple[int, ...]] = {input_name: input_shape for input_name, input_shape in zip(map(lambda o: o.name, stealth_codelet.inputs), map(lambda n: tuple(n.shape), unit_test.inputs))}
    checker_error_message = get_codelet_check_error_message(stealth_codelet, operand_dim_sizes, config)
    if checker_error_message is None:
        with lock:
            compile(config_path, stealth_codelet, operand_dim_sizes, thread_id=thread_id)
        if verify_output:
            unit_test.run(substitute_variables(stealth_codelet, operand_dim_sizes))
        simulator_outputs = run_layer(f"stealth_outputs/compilation_output/test_benchmark{array_n}x{array_m}_{thread_id}_stealth")
        relevant_simulator_outputs = get_relevant_simulator_outputs(simulator_outputs)
        relevant_simulator_outputs["input_dimensions"] = input_dimensions 
        return relevant_simulator_outputs
    else:
        return {"error": checker_error_message, "input_dimensions": input_dimensions}


def get_unit_test_outputs(unit_tests: tuple[UnitTest, ...], codelet_string: str, stealth_codelet: StealthCodelet, operand_shapes: tuple[tuple[str, ...], ...], dimension_tiling: tuple[int, ...], loop_order: tuple[int, ...], iterable_dimensions: tuple[str, ...], config_path: str, config: dict[str, Union[int, str, bool]], array_n: int, array_m: int, thread_id: int, lock: Lock, output_queue: Queue):
    current_codelet_output = {}
    current_codelet_output["codelet"] = codelet_string
    for dim_name, number_of_tiles in zip(iterable_dimensions, dimension_tiling):
        current_codelet_output[f"{dim_name}_tiles"] = number_of_tiles
    current_codelet_output["loop_order"] = [iterable_dimensions[l] for l in loop_order]
    
    unit_test_outputs = []
    for unit_test in unit_tests:
        operand_dim_sizes = {}
        for operand_shape, operand_shape_dim_names in zip(map(lambda n: tuple(n.shape), unit_test.inputs), operand_shapes):
            for dim_name, dim_size in zip(operand_shape_dim_names, operand_shape):
                operand_dim_sizes[dim_name] = dim_size
        unit_test_outputs.append(get_unit_test_output(unit_test, stealth_codelet, operand_dim_sizes, config_path, config, array_n, array_m, thread_id, lock))
        
    current_codelet_output["unit_test_outputs"] = unit_test_outputs
    if len(unit_test_outputs) > 0:
        output_queue.put(current_codelet_output)
        

def generate_simd_unary_element_wise_tensor_codelet(start_index: int, end_index: int, thread_id: int, shared_data: dict[str, Any], output_queue: Queue, lock: Lock, **kwargs: Any) -> None:
    assert "codelet_string_generation_function" in kwargs, "Missing codelet_string_generation_function argument."
    codelet_string_generation_function: Callable = kwargs["codelet_string_generation_function"]
    
    assert "unit_tests" in kwargs, "Missing unit_tests argument."
    unit_tests: tuple[UnitTest, ...] = kwargs["unit_tests"]

    assert "config_path" in kwargs, "Missing config_path argument."
    config_path: str = kwargs["config_path"]

    config = load_config(config_path)

    assert "operand_shape" in kwargs, "Missing operand_shape argument."
    operand_shape: tuple[str, ...] = kwargs["operand_shape"]

    assert "array_n" in kwargs, "Missing array_n argument."
    assert "array_m" in kwargs, "Missing array_m argument."
    array_n: int = kwargs["array_n"]
    array_m: int = kwargs["array_m"]

    for _ in range(end_index - start_index):
        with lock:
            search_space_point = shared_data["searcher"].get_next_search_space_point()
        
        dimension_tiling: tuple[int, ...] = get_tiling(search_space_point, operand_shape)

        loop_order_point = search_space_point[-1]
        assert isinstance(loop_order_point, LoopOrderSpacePoint)
        loop_order: tuple[int, ...] = loop_order_point.loop_order

        input_vmem = get_random_vmem()
        output_vmem = get_random_vmem()

        codelet_string: str = codelet_string_generation_function(("input_activation", operand_shape), ("output_activation", operand_shape), array_n, dimension_tiling, loop_order, input_vmem, output_vmem)
        stealth_codelet: StealthCodelet = codelet_string_to_codelet(codelet_string) 
        get_unit_test_outputs(unit_tests, codelet_string, stealth_codelet, (operand_shape,), dimension_tiling, loop_order, operand_shape, config_path, config, array_n, array_m, thread_id, lock, output_queue)
        

def generate_simd_binary_element_wise_codelet(start_index: int, end_index: int, thread_id: int, shared_data: dict[str, Any], output_queue: Queue, lock: Lock, **kwargs: Any) -> None:
    assert "codelet_string_generation_function" in kwargs, "Missing codelet_string_generation_function argument."
    codelet_string_generation_function: Callable = kwargs["codelet_string_generation_function"]
    
    assert "unit_tests" in kwargs, "Missing unit_tests argument."
    unit_tests: tuple[UnitTest, ...] = kwargs["unit_tests"]

    assert "config_path" in kwargs, "Missing config_path argument."
    config_path: str = kwargs["config_path"]

    config = load_config(config_path)

    scalar: Optional[int] = kwargs.get("scalar", None)
    is_scalar_first: bool = kwargs.get("is_scalar_first", False)

    assert "operand_shapes" in kwargs, "Missing operand_shapes argument."
    operand_shapes: tuple[tuple[str, ...], ...] = kwargs["operand_shapes"]
    if scalar is None:
        assert len(operand_shapes) == 2
    else:
        assert len(operand_shapes) == 1
    iterable_dimensions = max(operand_shapes, key=lambda s: len(s))

    assert "array_n" in kwargs, "Missing array_n argument."
    assert "array_m" in kwargs, "Missing array_m argument."
    array_n: int = kwargs["array_n"]
    array_m: int = kwargs["array_m"]

    for _ in range(end_index - start_index):
        with lock:
            search_space_point = shared_data["searcher"].get_next_search_space_point()

        dimension_tiling = get_tiling(search_space_point, iterable_dimensions)

        loop_order_point = search_space_point[-1]
        assert isinstance(loop_order_point, LoopOrderSpacePoint)
        loop_order: tuple[int, ...] = loop_order_point.loop_order

        input_1_vmem = get_random_vmem()
        input_2_vmem = get_random_vmem()
        output_vmem = get_random_vmem()
        
        if scalar is not None:
            if is_scalar_first:
                codelet_string = codelet_string_generation_function(scalar, ("input_activation", operand_shapes[0]), ("output_activation", iterable_dimensions), array_n, dimension_tiling, loop_order, input_1_vmem, output_vmem)
            else:
                codelet_string = codelet_string_generation_function(("input_activation", operand_shapes[0]), scalar, ("output_activation", iterable_dimensions), array_n, dimension_tiling, loop_order, input_1_vmem, output_vmem)
        else:
            codelet_string = codelet_string_generation_function(("input_activation_1", operand_shapes[0]), ("input_activation_2", operand_shapes[1]), ("output_activation", iterable_dimensions), array_n, dimension_tiling, loop_order, input_1_vmem, input_2_vmem, output_vmem)
        stealth_codelet: StealthCodelet = codelet_string_to_codelet(codelet_string) 
        get_unit_test_outputs(unit_tests, codelet_string, stealth_codelet, operand_shapes, dimension_tiling, loop_order, iterable_dimensions, config_path, config, array_n, array_m, thread_id, lock, output_queue)


def generate_systolic_array_codelet(start_index: int, end_index: int, thread_id: int, shared_data: dict[str, Any], output_queue: Queue, lock: Lock, **kwargs: Any) -> None:
    assert "codelet_string_generation_function" in kwargs, "Missing codelet_string_generation_function argument."
    codelet_string_generation_function: Callable = kwargs["codelet_string_generation_function"]
    
    assert "unit_tests" in kwargs, "Missing unit_tests argument."
    unit_tests: tuple[UnitTest, ...] = kwargs["unit_tests"]

    assert "config_path" in kwargs, "Missing config_path argument."
    config_path: str = kwargs["config_path"]

    config = load_config(config_path)

    assert "operand_shapes" in kwargs, "Missing operand_shapes argument."
    operand_shapes: tuple[tuple[str, ...], ...] = kwargs["operand_shapes"]
    assert len(operand_shapes) == 2 or len(operand_shapes) == 3
    if len(operand_shapes[1]) < len(operand_shapes[0]):
        outer_dims = operand_shapes[0][:-2]
    else:
        outer_dims = operand_shapes[1][:-2]
    inner_dims: tuple[str, ...] = operand_shapes[0][-2:] + operand_shapes[1][-1:] 
    iterable_dimensions: tuple[str, ...] = outer_dims + inner_dims
    if len(operand_shapes) == 3:
        assert len(operand_shapes[2]) == 1    

    assert "array_n" in kwargs, "Missing array_n argument."
    assert "array_m" in kwargs, "Missing array_m argument."
    array_n: int = kwargs["array_n"]
    array_m: int = kwargs["array_m"]

    for _ in range(end_index - start_index):
        with lock:
            search_space_point = shared_data["searcher"].get_next_search_space_point()

        dimension_tiling = get_tiling(search_space_point, iterable_dimensions)

        loop_order_point = search_space_point[-1]
        assert isinstance(loop_order_point, LoopOrderSpacePoint)
        loop_order: tuple[int, ...] = loop_order_point.loop_order 

        codelet_string: str = codelet_string_generation_function(("input_activation", operand_shapes[0]), ("weight", operand_shapes[1]), ("bias", operand_shapes[2]) if len(operand_shapes) == 3 else None, ("output_activation", outer_dims + (inner_dims[0], inner_dims[2])), array_n, array_m, dimension_tiling, loop_order)
        stealth_codelet: StealthCodelet = codelet_string_to_codelet(codelet_string) 
        get_unit_test_outputs(unit_tests, codelet_string, stealth_codelet, operand_shapes, dimension_tiling, loop_order, iterable_dimensions, config_path, config, array_n, array_m, thread_id, lock, output_queue)


def run_codelet_operation_generation(thread_function: Callable, shared_data: dict[str, Any], num_points: Optional[int], num_jobs: int, **kwargs: Any):
    os.makedirs("stealth_outputs/tiling_info", exist_ok=True)
    
    lock = Lock()
    output_queue = Queue()

    points_per_thread = num_points // num_jobs
    threads: list[Thread] = []
    for job in range(num_jobs):
        start_index = job * points_per_thread
        end_index = start_index + points_per_thread
        if job == num_jobs - 1:
            end_index = num_points
        
        thread = Thread(target=thread_function, args=(start_index, end_index, job, shared_data, output_queue, lock), kwargs=kwargs)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    operation_output = []
    while not output_queue.empty():
        operation_output.append(output_queue.get())

    return operation_output


def generate_simd_relu4d_dataset(operation_id: int, config_path: str, num_points: Optional[int], num_jobs: int) -> str:
    assert operation_id == 0

    config = load_config(config_path)

    unit_test_input_shapes: tuple[tuple[int, ...], ...] = (
        (1, 224, 224, config["ARRAY_N"] * 4),
        (1, 49, 49, config["ARRAY_N"] * 2)
    )
    unit_tests = []
    for unit_test_input_shape in unit_test_input_shapes:
        input_tensor = get_int32_tensor(unit_test_input_shape)
        output_tensor = np.maximum(input_tensor, 0) 
        unit_tests.append(UnitTest((input_tensor,), (output_tensor,)))
    unit_tests = tuple(unit_tests)

    operand_shape: tuple[str, ...] = get_shape_from_number_of_dimensions(4)

    config = load_config(config_path)
    array_n = config["ARRAY_N"]
    array_m = config["ARRAY_M"]

    all_unit_test_dim_sizes = [[] for _ in range(len(unit_tests[0].inputs[0].shape))]
    for unit_test in unit_tests:
        for i, dim in enumerate(unit_test.inputs[0].shape):
            all_unit_test_dim_sizes[i].append(dim)
    tile_spaces: tuple[TileSpace, ...] = tuple(TileSpace(dim_sizes) for dim_sizes in all_unit_test_dim_sizes)
    loop_order_space: LoopOrderSpace = LoopOrderSpace(len(operand_shape))

    if num_points is None:
        searcher = ExhaustiveSearcher(tile_spaces + (loop_order_space,))
        num_points = searcher.get_size_of_search_space()
    else:
        searcher = RandomSearcher(tile_spaces + (loop_order_space,))
    
    shared_data: dict[str, Any] = {
        "searcher": searcher
    }

    operation_output = run_codelet_operation_generation(generate_simd_unary_element_wise_tensor_codelet, shared_data, num_points, num_jobs, codelet_string_generation_function=generate_simd_element_wise_relu_codelet, unit_tests=unit_tests, config_path=config_path, operand_shape=operand_shape, array_n=array_n, array_m=array_m)
    if len(operation_output) > 0:
        with open(f"relu4d.json", "w") as f:
            json.dump({"data": operation_output}, f, indent=4) 
    else:
        raise RuntimeError("No valid configs found for operation. Try increasing the number of tried configs")

    return "relu4d.json"


def organize_unit_test_input_shapes_for_element_wise_operations(input_shapes: tuple[tuple[tuple[int, ...], str], ...], input_dimensionality: tuple[int, ...]) -> tuple[tuple[tuple[int, ...], str]]:
    ret = []
    for input_shape in input_shapes:
        new_inputs = []
        for input_tensor_number_of_dimensions in input_dimensionality:
            assert len(input_shape[0]) >= input_tensor_number_of_dimensions, f"Input shape {input_shape[0]} does not have enough dimensions for {input_tensor_number_of_dimensions} input tensors."
            if input_tensor_number_of_dimensions == 0:
                new_inputs.append(((), input_shape[1]))
            else:
                new_inputs.append((input_shape[0][len(input_shape[0]) - input_tensor_number_of_dimensions:], input_shape[1]))
        ret.append(tuple(new_inputs))
    return tuple(ret)


def generate_profiled_simple_simd_ops_dataset(operation_id: int, config_path: str, num_points: Optional[int], num_jobs: int) -> str:
    config = load_config(config_path)

    simd_4d_contrived_input_shapes = (
        simd_1_128_128_2048(),
        simd_1_256_256_1024(),
        simd_1_512_512_512(),
        simd_1_1024_1024_256(),
        simd_1_2048_2048_64(),
    )
    unary_4d_contrived_input_shapes = organize_unit_test_input_shapes_for_element_wise_operations(simd_4d_contrived_input_shapes, (4,)) 
    binary_4d_4d_contrived_input_shapes = organize_unit_test_input_shapes_for_element_wise_operations(simd_4d_contrived_input_shapes, (4, 4)) 

    simd_3d_contrived_input_shapes = (
        simd_1_128_2048(),
        simd_1_256_1024(),
        simd_1_512_512(),
        simd_1_1024_256(),
        simd_1_2048_128(),
    )
    unary_3d_contrived_input_shapes = organize_unit_test_input_shapes_for_element_wise_operations(simd_3d_contrived_input_shapes, (3,))
    binary_3d_3d_contrived_input_shapes = organize_unit_test_input_shapes_for_element_wise_operations(simd_3d_contrived_input_shapes, (3, 3))
    binary_3d_1d_contrived_input_shapes = organize_unit_test_input_shapes_for_element_wise_operations(simd_3d_contrived_input_shapes, (3, 1))
    binary_3d_0d_contrived_input_shapes = organize_unit_test_input_shapes_for_element_wise_operations(simd_3d_contrived_input_shapes, (3, 0))

    shape_4d = get_shape_from_number_of_dimensions(4)
    shape_3d = get_shape_from_number_of_dimensions(3)

    OPERATIONS: list[tuple[str, Callable, tuple[str, ...], tuple[tuple[str, ...], ...], Callable, tuple[tuple[tuple[tuple[int, ...], str]]]]] = [
        # (
        #     "matmul_bias2d2d",
        #     functools.partial(generate_systolic_array_codelet, codelet_string_generation_function=generate_systolic_array_matmul_codelet, operand_shapes=(("M", "N"), ("N", "P"), ("P",))),
        #     ("M", "N", "P"),
        #     (("M", "N"), ("N", "P"), ("P",)),
        #     lambda x: np.matmul(x[0], x[1]) + x[2],
        #     (
        #         (((64, 64), "i8"), ((64, 64), "i8"), ((64,), "i32")),
        #         (((256, 256), "i8"), ((256, 256), "i8"), ((256,), "i32")),
        #         (((64, 256), "i8"), ((256, 256), "i8"), ((256,), "i32")),
        #         (((1, 1024), "i8"), ((1024, 1024), "i8"), ((1024,), "i32")),
        #         (((128, 512), "i8"), ((512, 64), "i8"), ((64,), "i32")),
        #     )
        # ),
        (
            "add4d4d",
            functools.partial(generate_simd_binary_element_wise_codelet, codelet_string_generation_function=generate_simd_element_wise_add_codelet, operand_shapes=repeat(shape_4d, 2)),
            shape_4d,
            repeat(shape_4d, 2),
            lambda x: x[0] + x[1],
            binary_4d_4d_contrived_input_shapes
        ),
        (
            "add3d3d",
            functools.partial(generate_simd_binary_element_wise_codelet, codelet_string_generation_function=generate_simd_element_wise_add_codelet, operand_shapes=repeat(shape_3d, 2)),
            shape_3d,
            repeat(shape_3d, 2),
            lambda x: x[0] + x[1],
            binary_3d_3d_contrived_input_shapes
        ),
        (
            "add3d1d",
            functools.partial(generate_simd_binary_element_wise_codelet, codelet_string_generation_function=generate_simd_element_wise_add_codelet, operand_shapes=(shape_3d, shape_3d[len(shape_3d) - 1:])),
            shape_3d,
            (shape_3d, shape_3d[len(shape_3d) - 1:]),
            lambda x: x[0] + x[1],
            binary_3d_1d_contrived_input_shapes
        ),
        (
            "add3d_scalar8",
            functools.partial(generate_simd_binary_element_wise_codelet, codelet_string_generation_function=generate_simd_element_wise_add_scalar_codelet, operand_shapes=(shape_3d,), scalar=8, is_scalar_first=False),
            shape_3d,
            (shape_3d,),
            lambda x: x[0] + 8,
            unary_3d_contrived_input_shapes
        ),
        (
            "sub3d3d",
            functools.partial(generate_simd_binary_element_wise_codelet, codelet_string_generation_function=generate_simd_element_wise_sub_codelet, operand_shapes=repeat(shape_3d, 2)),
            shape_3d,
            repeat(shape_3d, 2),
            lambda x: x[0] - x[1],
            binary_3d_3d_contrived_input_shapes
        ),
        (
            "sub0d3d",
            functools.partial(generate_simd_binary_element_wise_codelet, codelet_string_generation_function=generate_simd_element_wise_sub_scalar_codelet, operand_shapes=(shape_3d,), scalar=8, is_scalar_first=True),
            shape_3d,
            (shape_3d,),
            lambda x: 8 - x[0],
            unary_3d_contrived_input_shapes 
        ),
        (
            "mul3d3d",
            functools.partial(generate_simd_binary_element_wise_codelet, codelet_string_generation_function=generate_simd_element_wise_mul_codelet, operand_shapes=repeat(shape_3d, 2)),
            shape_3d,
            repeat(shape_3d, 2),
            lambda x: x[0] * x[1],
            binary_3d_3d_contrived_input_shapes
        ),
        (
            "mul3d1d",
            functools.partial(generate_simd_binary_element_wise_codelet, codelet_string_generation_function=generate_simd_element_wise_mul_codelet, operand_shapes=(shape_3d, shape_3d[len(shape_3d) - 1:])),
            shape_3d,
            (shape_3d, shape_3d[len(shape_3d) - 1:]),
            lambda x: x[0] * x[1],
            binary_3d_1d_contrived_input_shapes
        ),
        (
            "mul3d0d",
            functools.partial(generate_simd_binary_element_wise_codelet, codelet_string_generation_function=generate_simd_element_wise_mul_scalar_codelet, operand_shapes=(shape_3d,), scalar=10, is_scalar_first=False),
            shape_3d,
            (shape_3d, ()),
            lambda x: x[0] * x[1],
            binary_3d_0d_contrived_input_shapes
        ),
        (
            "div3d3d",
            functools.partial(generate_simd_binary_element_wise_codelet, codelet_string_generation_function=generate_simd_element_wise_div_codelet, operand_shapes=repeat(shape_3d, 2)),
            shape_3d,
            repeat(shape_3d, 2),
            lambda x: x[0] / x[1],
            binary_3d_3d_contrived_input_shapes
        ),
        (
            "div3d0d",
            functools.partial(generate_simd_binary_element_wise_codelet, codelet_string_generation_function=generate_simd_element_wise_div_scalar_codelet, operand_shapes=(shape_3d,), scalar=10, is_scalar_first=False),
            shape_3d,
            (shape_3d, ()),
            lambda x: x[0] / x[1],
            binary_3d_0d_contrived_input_shapes
        ),
        (
            "pow3d0d",
            functools.partial(generate_simd_binary_element_wise_codelet, codelet_string_generation_function=generate_simd_element_wise_pow_scalar_codelet, operand_shapes=(shape_3d,), scalar=2, is_scalar_first=False),
            shape_3d,
            (shape_3d, ()),
            lambda x: np.power(x[0], x[1]),
            binary_3d_0d_contrived_input_shapes
        ),
        (
            "sqrt3d",
            functools.partial(generate_simd_unary_element_wise_tensor_codelet, codelet_string_generation_function=generate_simd_element_wise_sqrt_codelet, operand_shape=shape_3d),
            shape_3d,
            repeat(shape_3d, 1),
            lambda x: np.sqrt(x[0]),
            unary_3d_contrived_input_shapes
        ),
        (
            "relu4d",
            functools.partial(generate_simd_unary_element_wise_tensor_codelet, codelet_string_generation_function=generate_simd_element_wise_relu_codelet, operand_shape=shape_4d),
            shape_4d,
            repeat(shape_4d, 1),
            lambda x: np.maximum(x[0], 0),
            unary_4d_contrived_input_shapes 
        ),
        (
            "sigmoid3d",
            functools.partial(generate_simd_unary_element_wise_tensor_codelet, codelet_string_generation_function=generate_simd_element_wise_sigmoid_codelet, operand_shape=shape_3d),
            shape_3d,
            repeat(shape_3d, 1),
            lambda x: 1 / (1 + np.exp(-x[0])),
            unary_3d_contrived_input_shapes
        ),
        (
            "tanh3d",
            functools.partial(generate_simd_unary_element_wise_tensor_codelet, codelet_string_generation_function=generate_simd_element_wise_tanh_codelet, operand_shape=shape_3d),
            shape_3d,
            repeat(shape_3d, 1),
            lambda x: np.tanh(x[0]),
            unary_3d_contrived_input_shapes
        ),
    ]
    OPERATION_ID_MAP = {i: operation for i, operation in enumerate(OPERATIONS)}
    operation_name, operation_func, iterable_dimensions, operand_shapes, ref_impl, unit_test_shapes = OPERATION_ID_MAP[operation_id]

    unit_tests = []
    for unit_test_input_shapes in unit_test_shapes:
        input_tensors = tuple(get_int32_tensor(unit_test_input_shape) if dtype == "i32" else get_int8_tensor(unit_test_input_shape) for unit_test_input_shape, dtype in unit_test_input_shapes)
        output_tensor = ref_impl(input_tensors) 
        unit_tests.append(UnitTest(input_tensors, (output_tensor,)))
    unit_tests = tuple(unit_tests)

    config = load_config(config_path)
    array_n = config["ARRAY_N"]
    array_m = config["ARRAY_M"]
    
    all_unit_test_dim_sizes = {dim_name: set() for dim_name in iterable_dimensions}
    for unit_test in unit_tests:
        for unit_test_input, operand_shape in zip(unit_test.inputs, operand_shapes):
            for dim_name, dim in zip(operand_shape, unit_test_input.shape):
                all_unit_test_dim_sizes[dim_name].add(dim)
    tile_spaces: tuple[TileSpace, ...] = tuple(TileSpace(tuple(dim_sizes)) for dim_sizes in all_unit_test_dim_sizes.values())
    loop_order_space: LoopOrderSpace = LoopOrderSpace(len(iterable_dimensions))

    if num_points is None:
        searcher = ExhaustiveSearcher(tile_spaces + (loop_order_space,))
        num_points = searcher.get_size_of_search_space()
    else:
        searcher = RandomSearcher(tile_spaces + (loop_order_space,))
    
    shared_data: dict[str, Any] = {
        "searcher": searcher
    }

    operation_output = run_codelet_operation_generation(operation_func, shared_data, num_points, num_jobs, unit_tests=unit_tests, config_path=config_path, array_n=array_n, array_m=array_m)
    if len(operation_output) > 0:
        with open(f"{operation_name}.json", "w") as f:
            json.dump({"data": operation_output}, f, indent=4) 
    else:
        raise RuntimeError("No valid configs found for operation. Try increasing the number of tried configs")
    
    return f"{operation_name}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI tool for processing tasks.")
    
    parser.add_argument("dataset", type=str, choices=["simd_relu4d", "profiled_simple_simd_ops"], 
                        help="Dataset to generate.")
    parser.add_argument("operation_id", type=int, help="Integer mapping to a specific operation.")
    parser.add_argument("config_path", type=str, help="The path to the configuration file for GeneSys.")

    parser.add_argument("--num_points", required=False, type=int, default=None, help="Number of data points to generate.")

    parser.add_argument("--num_jobs", required=False, type=int, default=1, help="Number of threads to launch (default is 1).")
    parser.add_argument("--map_path", required=False, type=str, default=None, help="A path to move the generated file after running.")

    args = parser.parse_args() 

    if args.dataset == "simd_relu4d":
        output_file_name = generate_simd_relu4d_dataset(args.operation_id, args.config_path, args.num_points, args.num_jobs)
    elif args.dataset == "profiled_simple_simd_ops":
        output_file_name = generate_profiled_simple_simd_ops_dataset(args.operation_id, args.config_path, args.num_points, args.num_jobs)
    else:
        raise RuntimeError("Dataset not supported.")
    
    if args.map_path is not None:
        assert os.path.exists(args.map_path)
        shutil.move(output_file_name, args.map_path + "/" + output_file_name)
    

if __name__ == "__main__":
    main()
