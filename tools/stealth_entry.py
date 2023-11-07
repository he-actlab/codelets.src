import argparse
from collections import OrderedDict
import json
import string
import os
from typing import Callable, Optional
import numpy as np
import tqdm
from threading import Thread, Lock
from queue import Queue
from stealth.codelet_string import *
from stealth.compile import compile
from stealth.stealth_codelet import *
from stealth.parse import parse_stealth_codelet
from stealth.unit_test import UnitTest, UnitTestResult
from stealth.run import run_layer, get_relevant_simulator_outputs
from stealth.search_space import *
from stealth.searcher import *
from codelets.examples.genesys import load_config


def get_shape_from_number_of_dimensions(number_of_dimensions: int) -> tuple[str, ...]:
    assert number_of_dimensions > 0, "Number of dimensions must be positive."

    NUMBER_OF_DIMENSIONS_TO_SHAPE = {
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
    return np.random.randint(-2**31, 2**31 - 1, shape, dtype=np.int32)


def get_random_vmem() -> str:
    return f"VMEM{np.random.randint(1, 2)}"


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

    assert "simd_width" in kwargs, "Missing simd_width argument."
    simd_width: int = kwargs["simd_width"]

    for _ in range(end_index - start_index):
        with lock:
            search_space_point = shared_data["searcher"].get_next_search_space_point()
        tiling_points = search_space_point[:len(operand_shape)]
        assert all(isinstance(tile_space_point, TileSpacePoint) for tile_space_point in tiling_points)
        dimension_tiling: tuple[int, ...] = tuple(tile_space_point.number_of_tiles for tile_space_point in tiling_points)

        loop_order_point = search_space_point[-1]
        assert isinstance(loop_order_point, LoopOrderSpacePoint)
        loop_order: tuple[int, ...] = loop_order_point.loop_order

        input_vmem = get_random_vmem()
        output_vmem = get_random_vmem()

        codelet_string: str = codelet_string_generation_function(("input_activation", operand_shape), ("output_activation", operand_shape), simd_width, dimension_tiling, loop_order, input_vmem, output_vmem)

        current_codelet_output = {}
        current_codelet_output["codelet"] = codelet_string
        for dim_name, number_of_tiles in zip(operand_shape, dimension_tiling):
            current_codelet_output[f"{dim_name}_tiles"] = number_of_tiles
        current_codelet_output["loop_order"] = [operand_shape[l] for l in loop_order]

        parsed_codelet = parse_stealth_codelet(codelet_string)
        stealth_codelet: StealthCodelet = build_codelet_from_parse_tree(parsed_codelet, codelet_string)

        unit_test_outputs = []
        for unit_test in unit_tests:
            input_operand_shape = unit_test.inputs[0].shape
            operand_dim_sizes = {dim_name: dim_size for dim_name, dim_size in zip(operand_shape, input_operand_shape)}
            checker_error_message = get_codelet_check_error_message(stealth_codelet, operand_dim_sizes, config)
            if checker_error_message is None:
                with lock:
                    compile(config_path, stealth_codelet, operand_dim_sizes, thread_id=thread_id)
                simulator_outputs = run_layer(f"stealth_outputs/compilation_output/test_benchmark{simd_width}x{simd_width}_{thread_id}_stealth")
                relevant_simulator_outputs = get_relevant_simulator_outputs(simulator_outputs)
                relevant_simulator_outputs["input_dimensions"] = input_operand_shape 
                unit_test_outputs.append(relevant_simulator_outputs)
            else:
                unit_test_outputs.append({"error": checker_error_message, "input_dimensions": input_operand_shape})
        
        current_codelet_output["unit_test_outputs"] = unit_test_outputs
        if len(unit_test_outputs) > 0:
            output_queue.put(current_codelet_output)


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


def generate_simd_relu4d_dataset(operation_id: int, config_path: str, num_points: Optional[int], num_jobs: int) -> None:
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
    simd_width = config["ARRAY_N"]

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



    operation_output = run_codelet_operation_generation(generate_simd_unary_element_wise_tensor_codelet, shared_data, num_points, num_jobs, codelet_string_generation_function=generate_simd_element_wise_relu_codelet, unit_tests=unit_tests, config_path=config_path, operand_shape=operand_shape, simd_width=simd_width)
    # operation_output = generate_simd_unary_element_wise_tensor_codelet(generate_simd_element_wise_relu_codelet, unit_tests, 4, config_path, num_points, num_jobs)
    if len(operation_output) > 0:
        with open(f"relu4d.json", "w") as f:
            json.dump({"data": operation_output}, f, indent=4) 
    else:
        raise RuntimeError("No valid configs found for operation. Try increasing the number of tried configs")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI tool for processing tasks.")
    
    parser.add_argument("dataset", type=str, choices=["simd_relu4d"], 
                        help="Dataset to generate.")
    parser.add_argument("operation_id", type=int, help="Integer mapping to a specific operation.")
    parser.add_argument("config_path", type=str, help="The path to the configuration file for GeneSys.")

    parser.add_argument("--num_points", required=False, type=int, default=None, help="Number of data points to generate.")

    parser.add_argument("--num_jobs", required=False, type=int, default=1, help="Number of threads to launch (default is 1).")

    args = parser.parse_args() 

    if args.dataset == "simd_relu4d":
        generate_simd_relu4d_dataset(args.operation_id, args.config_path, args.num_points, args.num_jobs)
    else:
        raise RuntimeError("Dataset not supported.")
    

if __name__ == "__main__":
    main()
