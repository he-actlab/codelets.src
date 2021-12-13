import torch

from .numpy_layers import *
from pathlib import Path
from collections import Iterable
from .genesys_model_utils import get_resnet18, get_resnet50
from .datagen_functions import binary, unary
from . import FXP_CONFIGS
from fxpmath import Fxp
import fxpmath
from codelets import Datatype
import numpy as np
import json

WEIGHTS_CL_TO_CF = [3, 2, 0, 1] # (KH, KW, IC, OC) -> (OC, IC, KH, KW)
WEIGHTS_CF_TO_CL = [2, 3, 1, 0] # (OC, IC, KH, KW) -> (KH, KW, IC, OC)
ACT_CL_TO_CF = [0, 3, 1, 2] # (N, H, W, C) -> (N, C, H, W)
ACT_CF_TO_CL = [0, 2, 3, 1] # (N, C, H, W) -> (N, H, W, C)
BINARY_FNS = ["elem_add", "elem_sub", "elem_mul"]
UNARY_FNS = ["elem_tanh", "elem_tanh2d", "relu2d", "relu", "sigmoid", "elem_sigmoid"]
# FLIP_SHAPE_PERM = [2, 3, 1, 0]
# FLIP_SHAPE_PERM = [2, 3, 0, 1]



# Shuffles weights within a tile for correct mapping to systolic array
def shuffle_weights(w_orig, layer_type="conv"):

    # Layout of weights is in (KH, KW, IC, OC) format
    weights = w_orig.copy()

    w_dim = weights.shape
    result = np.zeros(w_dim, dtype=weights.dtype)
    tile_m = GENESYS_CFG['ARRAY_M']
    tile_n = GENESYS_CFG['ARRAY_N']
    coord_map = {}

    if layer_type == "conv":
        ic_coords = []
        oc_coords = []
        all_ics = []
        for kh in range(w_dim[0]):
            for kw in range(w_dim[1]):
                for ic in range(0, w_dim[2], tile_n):  # IC
                    for oc in range(0, w_dim[3], tile_m): # OC
                        for n in range(tile_n):  # Rows
                            for m in range(tile_m): # Columns
                                # Reverse order within a tile because systolic array is filled from last column first.

                                src_coord = kh, kw, ic + n, oc + m
                                dst_coord = kh, kw, ic + n, oc + tile_m - m - 1
                                # dst_coord = kh, kw, ic + tile_n - n - 1, oc + tile_m - m - 1

                                assert src_coord not in coord_map
                                coord_map[src_coord] = dst_coord
                                assert src_coord[-1] < result.shape[-1], f"Invalid coordinate for source: {src_coord[-1]}\n" \
                                                                         f"OC: {oc}, Column: {m}"
                                assert dst_coord[-1] < result.shape[-1], f"Invalid coordinate for source: {dst_coord[-1]}\n" \
                                                                         f"OC: {oc}, Column: {n}"

                                result[dst_coord[0]][dst_coord[1]][dst_coord[2]][dst_coord[3]] = weights[src_coord[0]][src_coord[1]][src_coord[2]][src_coord[3]]

    else:
        assert layer_type in ["linear", "gemm"]
       # result = np.zeros((w_dim[1], w_dim[0]), dtype=weights.dtype)
        for ic in range(0, w_dim[0], tile_n):
            for oc in range(0, w_dim[1], tile_m):
                for n in range(tile_n):
                    for m in range(tile_m):
                        # Reverse order within a tile because systolic array is filled from last column first.
                        # Adjacent values in memory are filled in systolic array column.
                        # So, if systolic array size is 32x32, weight at (0, 0) should be in (31,0) in memory
                        # weight at (1, 0) should be in (31,1) in memory and so on.
                        # dst_coord = (nn + n, mm + tile_m - m - 1)
                        # src_coord = (nn+n, mm+m)
                        # coord_map[src_coord] = dst_coord
                        #
                        # result[nn + n][mm + tile_m - m - 1] = weights[nn + n][mm + m]
                        # result[kh][kw][nn + n][mm + tile_m - m - 1] = weights[kh][kw][nn + n][mm + m]
                        src_coord = ic + n, oc + m
                        dst_coord = ic + n, oc + tile_m - m - 1
                        coord_map[src_coord] = dst_coord

                        result[dst_coord[0]][dst_coord[1]] = \
                        weights[src_coord[0]][src_coord[1]]

    return result, coord_map

# Sequentially write out tiles of weights which will be written in DRAM
# A tile is written out in column-major order.
# Column major order to enable a tile-size sequential read from DRAM to go to column of systolic array
def tiled_flatten(weights, dram_tiling, cdlt, layer_type = 'gemm'):

    if isinstance(weights, tuple):
        weights, coord_map = weights
        rev_coords = {v: k for k,v in coord_map.items()}
    else:
        rev_coords = {}
    final_coords = {}
    result = list()
    tile_m = GENESYS_CFG['ARRAY_M']
    tile_n = GENESYS_CFG['ARRAY_N']
    weight_symbols = list(cdlt.inputs[1].shape_symbols.keys())
    w_dim = weights.shape
    loop_order = [i for i in cdlt.get_loop_order() if i in weight_symbols]
    if layer_type == 'gemm':

        # big_tile_size_oc = dram_tiling['P']
        # big_tile_size_ic = dram_tiling['N']

        # w_dim_outer =

        big_tile_size_oc = dram_tiling[loop_order[0]]
        w_dim_outer = weight_symbols.index(loop_order[0])

        w_dim_inner = weight_symbols.index(loop_order[1])
        big_tile_size_ic = dram_tiling[loop_order[1]]

        for big_tile_oc in range(0, w_dim[w_dim_outer], big_tile_size_oc):  # Tile over OC
            for big_tile_ic in range(0, w_dim[w_dim_inner], big_tile_size_ic):  # Tile over IC
                for ic in range(0, big_tile_size_ic, tile_m):  # IC
                    for oc in range(0, big_tile_size_oc, tile_n):  # OC
                        for n in range(tile_n):  # Rows
                            for m in range(tile_m):  # Columns
                                # src_coord = (big_tile_ic + ic + m, big_tile_oc + oc + n)
                                src_coord = [None, None]
                                src_coord[w_dim_outer] = big_tile_oc + oc + n
                                src_coord[w_dim_inner] = big_tile_ic + ic + m
                                src_coord = tuple(src_coord)
                                dst_coord = np.unravel_index([len(result)], weights.shape)
                                final_coords[rev_coords[src_coord]] = dst_coord
                                # result.append(weights[big_tile_ic + ic + m][big_tile_oc + oc + n])
                                result.append(weights[src_coord[0]][src_coord[1]])
    else:
        assert layer_type == 'conv'
        big_tile_size_oc = dram_tiling['OC']
        big_tile_size_ic = dram_tiling['IC']

        for big_tile_oc in range(0, w_dim[3], big_tile_size_oc):  # Tile over OC
            for big_tile_ic in range(0, w_dim[2], big_tile_size_ic):  # Tile over IC
                for kh in range(w_dim[0]):
                    for kw in range(w_dim[1]):
                        for ic in range(0, big_tile_size_ic, tile_m):  # IC
                            for oc in range(0, big_tile_size_oc, tile_n):  # OC
                                for n in range(tile_n):  # Rows
                                    for m in range(tile_m):  # Columns
                                        src_coord = (kh, kw, big_tile_ic + ic + m, big_tile_oc + oc + n)
                                        dst_coord = np.unravel_index([len(result)], weights.shape)
                                        final_coords[rev_coords[src_coord]] = dst_coord
                                        result.append(weights[kh][kw][big_tile_ic + ic + m][big_tile_oc + oc + n])
    absolute_coords = {np.ravel_multi_index(k, weights.shape): np.ravel_multi_index(v, weights.shape) for k,v in final_coords.items()}
    return np.array(result, weights.dtype)

def dram_layout(weights, print_debug=False):
    dram_weights = []
    assert weights.dtype == np.int8 or weights.dtype == np.uint8
    flat_weights = weights.flatten()
    flat_weights = flat_weights.astype(np.uint8)
    n = flat_weights.shape[0]
    assert n >= 4
    i = 0
    nums = [i]
    while i < (n-4):
        concat_weights = (flat_weights[i]) + \
                         (flat_weights[i + 1] << 8) + \
                         (flat_weights[i + 2] << 16) + \
                         (flat_weights[i + 3] << 24)
        dram_weights.append(concat_weights)

        i += 4
        nums.append(i)

    concat_weights = flat_weights[i]
    if i + 1 < n:
        concat_weights += flat_weights[i + 1] << 8
    if i + 2 < n:
        concat_weights += flat_weights[i + 2] << 16
    if i + 3 < n:
        concat_weights += flat_weights[i + 3] << 24
    dram_weights.append(concat_weights)
    dram_weights = [str(x) for x in dram_weights]
    return dram_weights





def compute_existing_values(json_path):
    with open(f"{json_path}", "r") as f:
        params = json.load(f)

    inpt_shape = list(params['program'][0]['inputs'][0]['shape_symbols'].values())
    wgt_shape = list(params['program'][0]['inputs'][1]['shape_symbols'].values())
    out_shape = list(params['program'][0]['outputs'][0]['shape_symbols'].values())
    stride = params['program'][0]['operation_parameters']['stride']

    parent_dir = Path(f"{Path(json_path).parent}")
    inpt_data = np.loadtxt(f"{parent_dir}/input_raw.txt", dtype=np.int32).reshape(tuple(inpt_shape))
    wgt_data = np.loadtxt(f"{parent_dir}/weights_raw.txt", dtype=np.int32).reshape(tuple(wgt_shape))
    out_data = np.loadtxt(f"{parent_dir}/output.txt", dtype=np.int32).reshape(tuple(out_shape))
    res = manual_conv_from_existing(inpt_data, wgt_data, out_data, stride)
    np.testing.assert_allclose(res, out_data)

def compute_range(fxp_dtype):
    cfg = FXP_CONFIGS[fxp_dtype]
    if cfg['signed']:
        upper_val = (1 << (cfg['n_word'] - 1)) - 1
        lower_val = -upper_val - 1
    else:
        upper_val = (1 << cfg['n_word']) - 1
        lower_val = 0

    upper = upper_val / 2.0 ** cfg['n_frac']
    lower = lower_val / 2.0 ** cfg['n_frac']
    return lower, upper

def numpy_datagen(shape, bitwidth, scale=2, cast_to=None, fxp_dtype=None):
    if fxp_dtype:
        low, high = compute_range(fxp_dtype)
    else:
        low = -(1 << (bitwidth // scale - 1))
        high = (1 << (bitwidth // scale - 1)) - 1
    v = np.random.randint(low=low, high=high,
                          size=shape,
                          dtype=np.int64)
    if fxp_dtype:

        v = Fxp(v * np.random.random(1),
                **FXP_CONFIGS[fxp_dtype]
                )
    else:
        if cast_to:
            v = v.astype(cast_to)

    return v


def generate_random_values(cdlt, model_name, layer_name, **kwargs):
    if "conv" in layer_name:
        generate_random_values_conv(cdlt, model_name, layer_name, **kwargs)
    elif "maxpool" in layer_name or "max_pool" in layer_name:
        generate_random_values_maxpool(cdlt, model_name, layer_name, **kwargs)
    elif layer_name in BINARY_FNS:
        generate_random_values_binary(cdlt, model_name, layer_name, **kwargs)
    elif layer_name in UNARY_FNS:
        generate_random_values_unary(cdlt, model_name, layer_name, **kwargs)
    else:
        assert "gemm" in layer_name
        generate_random_values_gemm(cdlt, model_name, layer_name, **kwargs)


def generate_random_values_binary(cdlt, model_name, layer_name,
                                base_path=".",
                                format="nhwc",
                                use_random=True,
                                fixed_values=None,
                                actual_data=False,
                                generate_partial_values=False):
    input_dims = cdlt.inputs[0].shape

    tiling_parameters = cdlt.param_tiling
    # DRAM tiling is in level 1.
    dram_tiling = tiling_parameters[1]

    if use_random:
        input1 = numpy_datagen(input_dims, cdlt.inputs[0].dtype.bits(), fxp_dtype=f"{cdlt.inputs[0].dtype}")
        input2 = numpy_datagen(input_dims, cdlt.inputs[0].dtype.bits(), fxp_dtype=f"{cdlt.inputs[0].dtype}")

    else:
        input1 = np.zeros(input_dims, dtype=np.int64).reshape(-1)
        input2 = np.zeros(input_dims, dtype=np.int64).reshape(-1)
        bits = (1 << (cdlt.inputs[0].dtype.bits() - 1)) - 1
        for i in range(np.prod(input_dims)):
            input1[i] = i % (bits)
            input2[i] = i % (bits)
        input1 = input1.reshape(input_dims)
        input2 = input2.reshape(input_dims)

    with open(f'{base_path}/input1_raw.txt', 'w') as f:
        if isinstance(input1, Fxp):
            f.write('\n'.join([str(i) for i in input1.val.flatten().tolist()]))
        else:
            f.write('\n'.join([str(i) for i in input1.flatten().tolist()]))

    with open(f'{base_path}/input2_raw.txt', 'w') as f:
        if isinstance(input2, Fxp):
            f.write('\n'.join([str(i) for i in input2.val.flatten().tolist()]))
        else:
            f.write('\n'.join([str(i) for i in input2.flatten().tolist()]))

    if format.lower() == "nhwc" and len(input1.shape) == 4:
        input1 = input1.transpose((0, 3, 1, 2))
        input2 = input2.transpose((0, 3, 1, 2))


    output = binary(input1, input2, layer_name, f"{cdlt.inputs[0].dtype}")
    if len(output.shape) == 4:
        output = output.transpose((0, 2, 3, 1))

    if fixed_values is not None and "outputs" in fixed_values:
        np.testing.assert_allclose(output, fixed_values["outputs"])

    # Write outputs to file
    with open(f'{base_path}/output.txt', 'w') as f:
        if isinstance(output, Fxp):
            f.write('\n'.join([str(i) for i in output.val.flatten().tolist()]))
        else:
            f.write('\n'.join([str(i) for i in output.flatten().tolist()]))

def sigmoid_pw(xval):

    if not isinstance(xval, Iterable):
        xval = Fxp([xval], like=xval)

    def inner(x):
        if x < 0:
            slope = -1.
            start = 1.
        elif (x >= 0) and (x < 1.0):
            slope = 1/4.
            start = 0.5
        elif (x >= 1.0) and (x < 2.375):
            slope = 1/8.
            start = 0.625
        elif (x >= 2.375) and (x < 5.0):
            slope = 1 / 32.
            start = 0.84375
        else:
            return Fxp(1.0, like=xval)

        return slope*x + start

    res = list(map(inner, xval.flatten().tolist()))
    res = Fxp(res, like=xval)
    res.val = res.val.reshape(xval.shape)
    return res

def test_sigmoid_pw(xval):

    if not isinstance(xval, Iterable):
        xval = Fxp([xval], like=xval, overflow='saturate')

    def inner(x, slope, start):
        return slope*x + start
    conds = [xval < 0,
             (xval >= 0) and (xval < 1.0),
             (xval >= 1.0) and (xval < 2.375),
             (xval >= 2.375) and (xval < 5.0),
             (xval >= 5.0)
             ]
    fns = [lambda x: inner(x, -1.0, 1.0),
            lambda x: inner(x, 1/4, 0.5),
           lambda x: inner(x, 1/8, 0.5),
           lambda x: inner(x, 1/32, 0.84375),
           lambda x: Fxp(1.0, like=xval, overflow='saturate')
           ]
    res = np.piecewise(xval, conds, fns)
    return res

def generate_random_values_unary(cdlt, model_name, layer_name,
                                base_path=".",
                                format="nhwc",
                                use_random=True,
                                fixed_values=None,
                                actual_data=False,
                                generate_partial_values=False):
    input_dims = cdlt.inputs[0].shape

    tiling_parameters = cdlt.param_tiling
    # DRAM tiling is in level 1.
    dram_tiling = tiling_parameters[1]

    if use_random:
        input1 = numpy_datagen(input_dims, cdlt.inputs[0].dtype.bits(), fxp_dtype=f"{cdlt.inputs[0].dtype}")

    else:
        input1 = np.zeros(input_dims, dtype=np.int64).reshape(-1)
        bits = (1 << (cdlt.inputs[0].dtype.bits() - 1)) - 1
        for i in range(np.prod(input_dims)):
            input1[i] = i % (bits)
        input1 = input1.reshape(input_dims)

    with open(f'{base_path}/input1_raw.txt', 'w') as f:
        if isinstance(input1, Fxp):
            f.write('\n'.join([str(i) for i in input1.val.flatten().tolist()]))
        else:
            f.write('\n'.join([str(i) for i in input1.flatten().tolist()]))



    if format.lower() == "nhwc" and len(input1.shape) == 4:
        input1 = input1.transpose((0, 3, 1, 2))

    output = unary(input1, layer_name, f"{cdlt.inputs[0].dtype}")

    if len(output.shape) == 4:
        output = output.transpose((0, 2, 3, 1))

    if fixed_values is not None and "outputs" in fixed_values:
        np.testing.assert_allclose(output, fixed_values["outputs"])

    with open(f'{base_path}/output.txt', 'w') as f:
        if isinstance(output, Fxp):
            f.write('\n'.join([str(i) for i in output.val.flatten().tolist()]))
        else:
            f.write('\n'.join([str(i) for i in output.flatten().tolist()]))

def generate_random_values_maxpool(cdlt, model_name, layer_name,
                                base_path=".",
                                format="nhwc",
                                use_random=True,
                                fixed_values=None,
                                actual_data=False,
                                generate_partial_values=False):

    # input_dims = cdlt.inputs[0].shape
    input_dims = tuple(cdlt.inputs[0].tiling['DRAM'].values())
    out_dims = cdlt.outputs[0].shape
    KH = cdlt.required_params['KH'].value
    KW = cdlt.required_params['KW'].value

    pad = cdlt.required_params['pad'].value
    if isinstance(pad, (tuple, list)):
        pad = pad[0]
    stride_x = cdlt.required_params['sx'].value
    stride_y = cdlt.required_params['sy'].value
    tiling_parameters = cdlt.param_tiling
    # DRAM tiling is in level 1.
    dram_tiling = tiling_parameters[1]

    if use_random:
        # low = -(1 << (cdlt.inputs[0].dtype.bits() - 1))
        # high = (1 << (cdlt.inputs[0].dtype.bits() - 1)) - 1
        # input = np.random.randint(low=low, high=high,
        #                            size=input_dims,
        #                            dtype=np.int64)
        input = numpy_datagen(input_dims, cdlt.inputs[0].dtype.bits())

    else:
        input = np.zeros(input_dims, dtype=np.int64).reshape(-1)
        bits = (1 << (cdlt.inputs[0].dtype.bits() - 1)) - 1
        for i in range(np.prod(input_dims)):
            input[i] = i % bits
        input = input.reshape(input_dims)

    with open(f'{base_path}/input_raw.txt', 'w') as f:
        f.write('\n'.join([str(i) for i in input.flatten().tolist()]))

    if format.lower() == "nhwc":
        input = input.transpose((0, 3, 1, 2))

    # test_output = manual_conv(input, weights, cdlt, layout="nchw")
    # tout = F.max_pool2d(torch)
    # output = pool2d(input.astype(np.int32), KH, stride_x, out_dims[1], out_dims[2], padding=pad)
    output = pool2d(input.astype(np.int64), KH, stride_x, padding=0)
    # print(input.shape)
    # tout = F.max_pool2d(torch.from_numpy(input.astype(np.float64)), stride=stride_y,
    #                     kernel_size=KH, padding=pad)

    output = output.transpose((0, 2, 3, 1))
    if fixed_values is not None and "outputs" in fixed_values:
        np.testing.assert_allclose(output, fixed_values["outputs"])

    # np.testing.assert_allclose(output, test_output)
    ## CLIP OUTPUTS
    # output = output.clip(np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    output = output.flatten().tolist()
    output = [str(x) for x in output]

    # Write outputs to file
    with open(f'{base_path}/output.txt', 'w') as f:
        f.write('\n'.join(output))


def partial_values_gemm(cdlt, base_path, x, w, ref_out, o_coord):
    other_test, ic_vals = manual_gemm(x, w, o_coord)
    with open(f'{base_path}/out_coords.csv', 'w') as f:
        for k, v in ic_vals.items():
            # f'"{all_coords}", {ocoord_idx}, {icoord_idx}, {wcoord_idx}, {inputs[icoord]}, {weights[wcoord]}, {partial_sum}')

            f.write(f'IC={k}, (m/n/p), O_idx, I_idx, W_idx, I_val, W_val, partial\n')
            for l in v:
                f.write(f"IC={k}, " + "," + l + "\n")
    np.testing.assert_allclose(other_test, ref_out)


def partial_values_conv(cdlt, base_path, x, w, ref_out, o_coord):

    other_test, ic_vals = manual_conv(x, w, cdlt, o_coord, layout="nhwc")
    with open(f'{base_path}/out_coords.csv', 'w') as f:
        f.write(f'IC, (oc/n/ic/kh/kw/y/x), O_idx, I_idx, W_idx, I_val, W_val, partial\n')

        for k, v in ic_vals.items():
            for l in v:
                f.write(f"IC={k}, " + "," + l + "\n")
    np.testing.assert_allclose(other_test, ref_out)

def generate_random_values_conv(cdlt, model_name, layer_name,
                                base_path=".",
                                format="nhwc",
                                use_random=True,
                                fixed_values=None,
                                actual_data=False,
                                generate_partial_values=False):
    input_dims = cdlt.inputs[0].shape
    weight_dims = cdlt.inputs[1].shape
    out_dims = cdlt.outputs[0].shape
    stride = cdlt.required_params['stride'].value
    pad = cdlt.required_params['pad'].value

    tiling_parameters = cdlt.param_tiling
    # DRAM tiling is in level 1.
    dram_tiling = tiling_parameters[1]
    bias = np.zeros(shape=out_dims[-1], dtype=np.int32)

    if actual_data:
        layer_num = 0
        layer_name = "conv"
        input, weights, bias, output = get_model_values(model_name, layer_name, layer_num)
        assert input.shape == input_dims
        assert weights.shape == weight_dims
        assert output.shape == out_dims
    elif use_random:
        input = numpy_datagen(input_dims, cdlt.inputs[0].dtype.bits(), scale=1, cast_to=np.int8)
        weights = numpy_datagen(weight_dims, cdlt.inputs[0].dtype.bits(), scale=1, cast_to=np.int8)

        # input = np.random.randint(low=-128, high=127, size=input_dims, dtype=np.int8)
        # weights = np.random.randint(low=-128, high=127, size=weight_dims, dtype=np.int8)
        # bias = np.random.randint(low=0, high=127, size=out_dims[-1], dtype=np.int32)
    elif fixed_values is not None:
        if "input" in fixed_values:
            assert "input" in fixed_values and "weights" in fixed_values
            input = np.zeros(input_dims, dtype=np.int8).reshape(-1)
            for i in range(np.prod(input_dims)):
                input[i] = fixed_values['input']
            input = input.reshape(input_dims)

            weights = np.zeros(weight_dims, dtype=np.int8).reshape(-1)
            for j in range(np.prod(weight_dims)):
                weights[j] = fixed_values['weights']
            weights = weights.reshape(weight_dims)
        else:
            assert "folder_path" in fixed_values
            with open(f'{fixed_values["folder_path"]}/input_raw.txt', 'r') as f:
                input = np.asarray([np.int8(l) for l in f.read().splitlines()], dtype=np.int8).reshape(input_dims)

            with open(f'{fixed_values["folder_path"]}/weights_raw.txt', 'r') as f:
                weights = np.asarray([np.int8(l) for l in f.read().splitlines()], dtype=np.int8).reshape(weight_dims)
            with open(f'{fixed_values["folder_path"]}/output.txt', 'r') as f:
                test_outputs = np.asarray([np.int32(l) for l in f.read().splitlines()], dtype=np.int32).reshape(out_dims)
    else:

        input = np.zeros(input_dims, dtype=np.int8).reshape(-1)
        for i in range(np.prod(input_dims)):
            input[i] = i % 128
        input = input.reshape(input_dims)
        cf_weight_dims = [weight_dims[i] for i in WEIGHTS_CL_TO_CF]
        weights = np.zeros(cf_weight_dims, dtype=np.int8).reshape(-1)
        for j in range(np.prod(weight_dims)):
            weights[j] = j % 128
        weights = weights.reshape(cf_weight_dims).transpose(*(WEIGHTS_CF_TO_CL))

    with open(f'{base_path}/input_shuffled.txt', 'w') as f:
        f.write('\n'.join(dram_layout(input)))

    with open(f'{base_path}/input_raw.txt', 'w') as f:
        f.write('\n'.join([str(i) for i in input.flatten().tolist()]))

    with open(f'{base_path}/weights_shuffled.txt', 'w') as f:
        f.write('\n'.join(dram_layout(
            tiled_flatten(
                shuffle_weights(weights, layer_type="conv"),
                dram_tiling, cdlt, layer_type='conv'))))

    with open(f'{base_path}/weights_shuffled_raw.txt', 'w') as f:
        f.write('\n'.join([str(i) for i in tiled_flatten(shuffle_weights(weights, layer_type="conv"), dram_tiling, cdlt,
                                                         layer_type='conv').tolist()]))

    with open(f'{base_path}/weights_raw.txt', 'w') as f:
        f.write('\n'.join([str(i) for i in weights.flatten().tolist()]))



    if format.lower() == "nhwc":
        input = input.transpose(0, 3, 1, 2)
        # Need to flip from (KH, KW, IC, OC) to (OC, IC, KH, KW)
        weights = weights.transpose(*tuple(WEIGHTS_CL_TO_CF))

    # test_output = manual_conv(input, weights, cdlt, layout="nchw")

    conv_param = {'stride': stride, 'pad': 0}
    b = np.zeros(weights.shape[0], dtype=np.int32)

    # output, _ = conv_forward_im2col(input.astype(np.int32), weights.astype(np.int32), b, conv_param)


    output, _ = conv_forward_naive(input.astype(np.int32), weights.astype(np.int32), b, conv_param)

    # tout = F.conv2d(torch.from_numpy(input.astype(np.float64)), torch.from_numpy(weights.astype(np.float64)),
    #                 torch.from_numpy(b.astype(np.float64)), stride=stride, padding=0)

    output = output.transpose(0, 2, 3, 1)
    if generate_partial_values:
        tinput = input.transpose(*tuple(ACT_CF_TO_CL))
        tweights = weights.transpose(*tuple(WEIGHTS_CF_TO_CL))
        coords = np.unravel_index(0, output.shape)
        partial_values_conv(cdlt, base_path, tinput, tweights, output, coords)
    if fixed_values is not None and "outputs" in fixed_values:
        np.testing.assert_allclose(output, fixed_values["outputs"])

    # np.testing.assert_allclose(output, test_output)

    output = output.flatten().tolist()
    output = [str(x) for x in output]

    # Write outputs to file
    with open(f'{base_path}/output.txt', 'w') as f:
        f.write('\n'.join(output))

    bias = bias.flatten().tolist()
    bias = [str(x) for x in bias]

    # Write outputs to file
    with open(f'{base_path}/bias.txt', 'w') as f:
        f.write('\n'.join(bias))


def generate_random_values_gemm(cdlt, model_name, layer_name,
                                base_path=".",
                                format="nhwc",
                                use_random=True,
                                fixed_values=None,
                                actual_data=False,
                                generate_partial_values=False):
    input_dims = cdlt.inputs[0].shape
    weight_dims = cdlt.inputs[1].shape
    out_dims = cdlt.outputs[0].shape
    tiling_parameters = cdlt.param_tiling
    # DRAM tiling is in level 1.
    dram_tiling = tiling_parameters[1]
    output = None
    bias = np.zeros(shape=out_dims[-1], dtype=np.int32)
    if actual_data:
        layer_num = 0
        layer_name = "linear"
        input, weights, bias, output = get_model_values(model_name, layer_name, layer_num)
        assert input.shape == input_dims
        assert weights.shape == weight_dims
        assert output.shape == out_dims
    elif use_random:
        input = numpy_datagen(input_dims, cdlt.inputs[0].dtype.bits(), scale=1)
        weights = numpy_datagen(weight_dims, cdlt.inputs[1].dtype.bits(), scale=1)

        # input = np.random.randint(low=-128, high=127, size=input_dims, dtype=np.int8)
        # weights = np.random.randint(low=-128, high=127, size=weight_dims, dtype=np.int8)
    elif fixed_values is not None:
        if "input" in fixed_values:
            assert "input" in fixed_values and "weights" in fixed_values
            input = np.zeros(input_dims, dtype=np.int8).reshape(-1)
            for i in range(np.prod(input_dims)):
                input[i] = fixed_values['input']
            input = input.reshape(input_dims)

            weights = np.zeros(weight_dims, dtype=np.int8).reshape(-1)
            for j in range(np.prod(weight_dims)):
                weights[j] = fixed_values['weights']
            weights = weights.reshape(weight_dims)
        else:
            assert "folder_path" in fixed_values
            with open(f'{fixed_values["folder_path"]}/input_raw.txt', 'r') as f:
                input = np.asarray([np.int8(l) for l in f.read().splitlines()], dtype=np.int8).reshape(input_dims)

            with open(f'{fixed_values["folder_path"]}/weights_raw.txt', 'r') as f:
                weights = np.asarray([np.int8(l) for l in f.read().splitlines()], dtype=np.int8).reshape(weight_dims)
            with open(f'{fixed_values["folder_path"]}/output.txt', 'r') as f:
                test_outputs = np.asarray([np.int32(l) for l in f.read().splitlines()], dtype=np.int32).reshape(out_dims)

    else:

        input = np.zeros(input_dims, dtype=np.int8).reshape(-1)
        for i in range(np.prod(input_dims)):
            input[i] = i % 128
        input = input.reshape(input_dims)
        weights = np.zeros(weight_dims, dtype=np.int8).reshape(-1)
        for j in range(np.prod(weight_dims)):
            weights[j] = j % 128
        weights = weights.reshape(weight_dims)


    with open(f'{base_path}/input_shuffled.txt', 'w') as f:
        f.write('\n'.join(dram_layout(input)))

    with open(f'{base_path}/input_raw.txt', 'w') as f:
        f.write('\n'.join([str(i) for i in input.flatten().tolist()]))

    with open(f'{base_path}/weights_shuffled.txt', 'w') as f:
        f.write('\n'.join(dram_layout(tiled_flatten(shuffle_weights(weights, layer_type="gemm"), dram_tiling, cdlt, layer_type="gemm"))))

    with open(f'{base_path}/weights_shuffled_raw.txt', 'w') as f:
        f.write('\n'.join([str(i) for i in tiled_flatten(shuffle_weights(weights, layer_type="gemm"), dram_tiling, cdlt).tolist()]))

    with open(f'{base_path}/weights_raw.txt', 'w') as f:
        f.write('\n'.join([str(i) for i in weights.flatten().tolist()]))

    bias = np.zeros(weights.shape[0], dtype=np.int32)

    if bias is not None:
        bias = bias.flatten().tolist()
        bias = [str(x) for x in bias]
        with open(f'{base_path}/bias.txt', 'w') as f:
            f.write('\n'.join(bias))

    if output is None:
        output = np.dot(np.int32(input), np.int32(weights))

    if generate_partial_values:

        partial_values_gemm(cdlt, base_path, input, weights, output, (0, 0))
    output = output.flatten().tolist()
    output = [str(x) for x in output]

    # # Write outputs to file
    with open(f'{base_path}/output.txt', 'w') as f:
        f.write('\n'.join(output))


def get_model_values(model_name, layer_name, layer_num, write_data=False):
    if model_name == "resnet18":
        layer_data, model = get_resnet18(True, layer_name, layer_num)
    elif model_name == "resnet50":
        layer_data, model = get_resnet50(True, layer_name, layer_num)
    else:
        raise RuntimeError

    if "conv" in layer_name.lower():
        x, wgt, b, out = pad_conv(layer_data)
    else:
        assert "linear" in layer_name.lower() or "gemm" in layer_name.lower()
        x, wgt, b, out = pad_gemm(layer_data)
    if write_data:
        base_filename = f'{model_name}_{layer_name.lower()}'

        with open(f'{base_filename}_input_i8.txt', 'w') as f:
            # f.write('\n'.join(dram_layout(x)))
            f.write('\n'.join([str(i) for i in x.flatten()]))

        with open(f'{base_filename}_weights_i8.txt', 'w') as f:
            # f.write('\n'.join(dram_layout(shuffle_weights(wgt))))
            f.write('\n'.join([str(i) for i in wgt.flatten()]))

        out = out.flatten().tolist()
        out = [str(x) for x in out]
        with open(f'{base_filename}_output_i32.txt', 'w') as f:
            f.write('\n'.join(out))

        b = b.flatten().tolist()
        b = [str(x) for x in b]
        with open(f'{base_filename}_bias_i32.txt', 'w') as f:
            f.write('\n'.join(b))
    else:
        return x, wgt, b, out
