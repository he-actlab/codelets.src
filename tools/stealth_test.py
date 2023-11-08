import numpy as np
from stealth.stealth_codelet.builder import build_codelet_from_parse_tree
from stealth.stealth_codelet.interpreter import interpret
from stealth.parse import parse_stealth_codelet
from stealth.stealth_codelet.variable_substituter import substitute_variables


MAX_INT32 = 2 ** 31 - 1
MIN_INT32 = -2 ** 31
MAX_INT8 = 2 ** 7 - 1
MIN_INT8 = -2 ** 7


def reverse_weight_matrix(weight_matrix: np.ndarray) -> np.ndarray:
    return np.flip(weight_matrix, axis=(0, 1))


def floating_point_to_fixed_point(floating_point_max_error: float, fixed_point_bits: int) -> int:
    return int(floating_point_max_error * 2 ** fixed_point_bits)


def run_codelet(codelet_string: str, input_arrays: tuple[np.ndarray, ...], name_to_variable_map: dict[str, int], verbose: bool = False) -> tuple[np.ndarray, ...]:
    tree = parse_stealth_codelet(codelet_string, verbose=verbose)
    codelet = build_codelet_from_parse_tree(tree, codelet_string, verbose=verbose)
    codelet = substitute_variables(codelet, name_to_variable_map)
    output_arrays = interpret(codelet, input_arrays, 16, 16)
    return output_arrays


def check_output_arrays(output_arrays: tuple[np.ndarray, ...], expected_output_arrays: tuple[np.ndarray, ...], max_error: int = 0) -> None:
    assert len(output_arrays) == len(expected_output_arrays)
    for output_array, expected_output_array in zip(output_arrays, expected_output_arrays):
        if max_error == 0:
            assert np.equal(output_array, expected_output_array).all(), f"Expected output array:\n{expected_output_array}\nActual output array:\n{output_array}"
        else:
            # error_matrix = np.abs(np.subtract(output_array, expected_output_array))
            # output_error = np.sum(error_matrix)
            max_output_error = np.max(np.abs(np.subtract(output_array, expected_output_array)))
            assert max_output_error <= max_error, f"Max single-element error {max_output_error} exceeds max single-element error {max_error} where expected output array:\n{expected_output_array}\nActual output array:\n{output_array}"


def do_relu4d_unit_test(input_shape: tuple[int, int, int, int], N_tiles: int, H_tiles: int, W_tiles: int, C_tiles: int, simd_width: int, max_error: int = 0, verbose: bool = False) -> None:
    np.random.seed(0)
    input_array = np.random.randint(MIN_INT32, MAX_INT32, dtype=np.int32, size=input_shape)
    expected_output_array = np.maximum(input_array, 0)
    codelet_string = f"""def relu(x: i32[N, H, W, C] @ DRAM):
    o = alloc([N, H, W, C], DRAM, i32)
    for n in loop({N_tiles}, N // {N_tiles}):
        for c in loop({C_tiles}, C // {C_tiles}):
            for h in loop({H_tiles}, H // {H_tiles}):
                for w in loop({W_tiles}, W // {W_tiles}):
                    x1 = load(x[n, h, w, c], [N // {N_tiles}, H // {H_tiles}, W // {W_tiles}, C // {C_tiles}], VMEM1)
                    o1 = alloc([N // {N_tiles}, H // {H_tiles}, W // {W_tiles}, C // {C_tiles}], VMEM1, i32)
                    for n1 in loop(N // {N_tiles}):
                        for c1 in loop((C // {C_tiles}) // {simd_width}, {simd_width}):
                            for h1 in loop(H // {H_tiles}):
                                for w1 in loop(W // {W_tiles}):
                                    x2 = load(x1[n1, h1, w1, c1], [{simd_width}], SIMD)
                                    o2 = relu(x2, SIMD)
                                    store(o1[n1, h1, w1, c1], o2)
                    store(o[n, h, w, c], o1)
    return o
"""
    name_to_variable_map = {"N": input_shape[0], "H": input_shape[1], "W": input_shape[2], "C": input_shape[3]}
    output_arrays = run_codelet(codelet_string, (input_array,), name_to_variable_map, verbose=verbose)
    assert len(output_arrays) == 1
    check_output_arrays(output_arrays, (expected_output_array,), max_error=max_error)


def unit_test_codelet_interpreter_on_relu4d(only_fast: bool = True, max_error: int = 0, verbose: bool = False) -> None:
    inputs = [
        {"fast": True, "input_shape": (1, 4, 4, 16), "N_tiles": (1,), "H_tiles": (1, 2, 4), "W_tiles": (1, 2, 4), "C_tiles": (1,)},
        {"fast": False, "input_shape": (1, 224, 224, 64), "N_tiles": (1,), "H_tiles": (1, 2, 4, 7, 8, 14, 16, 28, 32, 56, 112, 224), "W_tiles": (1, 2, 4, 7, 8, 14, 16, 28, 32, 56, 112, 224), "C_tiles": (1, 2, 4)}
    ]
    for input_config in inputs:
        if only_fast and not input_config["fast"]:
            continue
        input_shape = input_config["input_shape"]
        for N_tiles in input_config["N_tiles"]:
            for H_tiles in input_config["H_tiles"]:
                for W_tiles in input_config["W_tiles"]:
                    for C_tiles in input_config["C_tiles"]:
                        if verbose:
                            print(f"Running relu4d test with input shape {input_shape}, N_tiles={N_tiles}, H_tiles={H_tiles}, W_tiles={W_tiles}, C_tiles={C_tiles}")
                        do_relu4d_unit_test(input_shape, N_tiles, H_tiles, W_tiles, C_tiles, 16, max_error=max_error, verbose=verbose)
                        if verbose:
                            print("Passed")


def do_reduce_mean3d_unit_test(input_shape: tuple[int, int, int], N_tiles: int, L_tiles: int, H_tiles: int, simd_width: int, max_error: int = 0, verbose: bool = False) -> None:
    np.random.seed(0)
    input_array = np.random.randint(MIN_INT32, MAX_INT32, dtype=np.int32, size=input_shape)
    expected_output_array = np.sum(input_array, axis=1, dtype=np.int32, keepdims=True) * floating_point_to_fixed_point(1 / input_shape[-2], 16)

    codelet_string = f"""def reduce_mean(x: i32[N, L, H] @ DRAM):
    o = alloc([N, 1, H], DRAM, i32)
    for n in loop({N_tiles}, N // {N_tiles}):
        for l in loop({L_tiles}, L // {L_tiles}):
            for h in loop({H_tiles}, H // {H_tiles}):
                x1 = load(x[n, h, l], [N // {N_tiles}, H // {H_tiles}, L // {L_tiles}], VMEM1)
                o1 = load(o[n, 0, l], [N // {N_tiles}, 1, L // {L_tiles}], VMEM2)
                for n1 in loop(N // {N_tiles}):
                    for l1 in loop((L // {L_tiles}) // {simd_width}, {simd_width}):
                        for h1 in loop(H // {H_tiles}):
                            x2 = load(x1[n1, h1, l1], [{simd_width}], SIMD)
                            o2 = load(o1[n1, 0, l1], [{simd_width}], SIMD)
                            o3 = add(x2, o2, SIMD) 
                            store(o1[n1, 0, l1], o3)
                store(o[n, 0, l], o1)
    for n2 in loop({N_tiles}, N // {N_tiles}):
        for l2 in loop({L_tiles}, L // {L_tiles}):
            o4 = load(o[n2, 0, l2], [N // {N_tiles}, 1, L // {L_tiles}], VMEM1)
            for n3 in loop(N // {N_tiles}):
                for l3 in loop((L // {L_tiles}) // {simd_width}, {simd_width}):
                    o5 = load(o4[n3, 0, l3], [{simd_width}], SIMD)
                    o6 = mul(o5, {floating_point_to_fixed_point(1 / input_shape[-2], 16)}, SIMD)
                    store(o4[n3, 0, l3], o6)
            store(o[n2, 0, l2], o4)
    return o
"""
    name_to_variable_map = {"N": input_shape[0], "L": input_shape[1], "H": input_shape[2]}
    output_arrays = run_codelet(codelet_string, (input_array,), name_to_variable_map, verbose=verbose)
    assert len(output_arrays) == 1
    check_output_arrays(output_arrays, (expected_output_array,), max_error=max_error)


def unit_test_codelet_interpreter_on_reduce_mean3d(only_fast: bool = True, max_error: int = 0, verbose: bool = False) -> None:
    inputs = [
        {"fast": True, "input_shape": (1, 16, 16), "N_tiles": (1,), "L_tiles": (1,), "H_tiles": (1,)},
        {"fast": False, "input_shape": (1, 1024, 1024), "N_tiles": (1,), "L_tiles": (1, 4, 8), "H_tiles": (1, 4)}
    ]
    for input_config in inputs:
        if only_fast and not input_config["fast"]:
            continue
        input_shape = input_config["input_shape"]
        for N_tiles in input_config["N_tiles"]:
            for L_tiles in input_config["L_tiles"]:
                for H_tiles in input_config["H_tiles"]:
                    if verbose:
                        print(f"Running reduce_mean3d test with input shape {input_shape}, N_tiles={N_tiles}, L_tiles={L_tiles}, H_tiles={H_tiles}")
                    do_reduce_mean3d_unit_test(input_shape, N_tiles, L_tiles, H_tiles, 16, max_error=max_error, verbose=verbose)
                    if verbose:
                        print("Passed")


def do_gemm_unit_test(data_shape: tuple[int, int], weight_shape: tuple[int, int], bias_shape: tuple[int], M_tiles: int, N_tiles: int, P_tiles: int, array_N: int, array_M: int, max_error: int = 0, verbose: bool = False) -> None:
    assert data_shape[1] == weight_shape[0]
    np.random.seed(0)
    data_array = np.random.randint(MIN_INT8, MAX_INT8, dtype=np.int8, size=data_shape)
    weight_array = np.random.randint(MIN_INT8, MAX_INT8, dtype=np.int8, size=weight_shape)
    bias_array = np.random.randint(MIN_INT32, MAX_INT32, dtype=np.int32, size=bias_shape)
    expected_output_array = np.matmul(np.int32(data_array), np.int32(weight_array)) + bias_array
    codelet_string = f"""def gemm_bias(x: i8[M, N] @ DRAM, w: i8[N, P] @ DRAM, b: i32[P] @ DRAM):
    o = alloc([M, P], DRAM, i32)
    for p in loop({P_tiles}, P // {P_tiles}):
        b1 = load(b[p], [P // {P_tiles}], BBUF)
        for n in loop({N_tiles}, N // {N_tiles}):
            w1 = load(w[n, p], [N // {N_tiles}, P // {P_tiles}], WBUF)
            for m in loop({M_tiles}, M // {M_tiles}):
                x1 = load(x[m, n], [M // {M_tiles}, N // {N_tiles}], IBUF)
                o1 = load(o[m, p], [M // {M_tiles}, P // {P_tiles}], OBUF)
                for p1 in loop((P // {P_tiles}) // {array_M}, {array_M}):
                    b2 = load(b1[p1], [1, {array_N}], PE_ARRAY)
                    for n1 in loop((N // {N_tiles}) // {array_N}, {array_N}):
                        w2 = load(w1[n1, p1], [{array_N}, {array_M}], PE_ARRAY)
                        for m1 in loop(M // {M_tiles}):
                            x2 = load(x1[m1, n1], [1, {array_N}], PE_ARRAY)
                            o2 = load(o1[m1, p1], [1, {array_M}], PE_ARRAY)
                            o3 = mvmul_bias(x2, w2, b2, o2, PE_ARRAY)
                            store(o1[m1, p1], o3)
                store(o[m, p], o1)
    return o
"""
    name_to_variable_map = {"M": data_shape[0], "N": data_shape[1], "P": weight_shape[1]}
    output_arrays = run_codelet(codelet_string, (data_array, weight_array, bias_array), name_to_variable_map, verbose=verbose)
    assert len(output_arrays) == 1
    check_output_arrays(output_arrays, (expected_output_array,), max_error=max_error)


def unit_test_codelet_interpreter_on_gemm(only_fast: bool = True, max_error: int = 0, verbose: bool = False) -> None:
    inputs = [
        {"fast": True, "data_shape": (1, 16), "weight_shape": (16, 16), "bias_shape": (16,), "M_tiles": (1,), "N_tiles": (1,), "P_tiles": (1,)},
        {"fast": True, "data_shape": (2, 16), "weight_shape": (16, 16), "bias_shape": (16,), "M_tiles": (1, 2), "N_tiles": (1,), "P_tiles": (1,)},
        {"fast": True, "data_shape": (1, 16), "weight_shape": (16, 32), "bias_shape": (32,), "M_tiles": (1,), "N_tiles": (1,), "P_tiles": (1, 2)},
        {"fast": False, "data_shape": (1, 1024), "weight_shape": (1024, 2048), "bias_shape": (2048,), "M_tiles": (1, 2, 4, 8, 16, 32, 64), "N_tiles": (1,), "P_tiles": (1,)},
        # {"fast": False, "data_shape": (1, 1024), "weight_shape": (1024, 2048), "bias_shape": (1, 2048), "M_tiles": (1, 2, 4, 8, 16, 32, 64), "N_tiles": (1, 2, 4, 8, 16, 32, 64), "P_tiles": (1, 2, 4, 8, 16, 32, 64)},
    ]
    for input_config in inputs:
        if only_fast and not input_config["fast"]:
            continue
        data_shape = input_config["data_shape"]
        weight_shape = input_config["weight_shape"]
        bias_shape = input_config["bias_shape"]
        for M_tiles in input_config["M_tiles"]:
            for N_tiles in input_config["N_tiles"]:
                for P_tiles in input_config["P_tiles"]:
                    if verbose:
                        print(f"Running gemm test with data shape {data_shape}, weight shape {weight_shape}, bias shape {bias_shape}, M_tiles={M_tiles}, N_tiles={N_tiles}, P_tiles={P_tiles}")
                    do_gemm_unit_test(data_shape, weight_shape, bias_shape, M_tiles, N_tiles, P_tiles, 16, 16, max_error=max_error, verbose=verbose)
                    if verbose:
                        print("Passed")


def unit_test_codelet_interpreter(only_fast: bool = True, verbose: bool = False) -> None:
    # unit_test_codelet_interpreter_on_relu4d(only_fast=only_fast, verbose=verbose)
    unit_test_codelet_interpreter_on_reduce_mean3d(only_fast=only_fast, verbose=verbose)
    # unit_test_codelet_interpreter_on_gemm(only_fast=only_fast, max_error=floating_point_to_fixed_point(0, 16), verbose=verbose)


if __name__ == "__main__":
    unit_test_codelet_interpreter(only_fast=False, verbose=True)
