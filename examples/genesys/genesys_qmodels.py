
from pathlib import Path
from collections import Iterable
from .genesys_model_utils import get_resnet18, get_resnet50
from .datagen_functions import binary, unary, numpy_datagen, manual_conv_from_existing, \
    maxpool2d, avgpool2d,  manual_conv, manual_gemm, conv_forward_naive, pad_conv, \
    pad_gemm, save_array, global_avg_pool, depthwise_conv2d
from .data_transformations import transform_data
from decimal import Decimal
from . import FXP_CONFIGS

from codelets import Datatype
import numpy as np
import json

WEIGHTS_CL_TO_CF = [3, 2, 0, 1] # (KH, KW, IC, OC) -> (OC, IC, KH, KW)
WEIGHTS_CF_TO_CL = [2, 3, 1, 0] # (OC, IC, KH, KW) -> (KH, KW, IC, OC)
ACT_CL_TO_CF = [0, 3, 1, 2] # (N, H, W, C) -> (N, C, H, W)
ACT_CF_TO_CL = [0, 2, 3, 1] # (N, C, H, W) -> (N, H, W, C)
BINARY_FNS = ["elem_add", "elem_sub", "elem_mul"]
UNARY_FNS = ["elem_tanh", "elem_tanh2d", "relu2d", "relu", "sigmoid", "elem_sigmoid", "leaky_relu", "clip", "elem_clip", "elem_ceil2d",
             "elem_pow2d", "reduce_mean2d", "reduce_min2d"]
# FLIP_SHAPE_PERM = [2, 3, 1, 0]
# FLIP_SHAPE_PERM = [2, 3, 0, 1]


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



def generate_random_values(cdlt, model_name, layer_name, **kwargs):
    if "depthwise_conv" in layer_name:
        generate_random_values_dw_conv(cdlt, model_name, layer_name, **kwargs)
    elif "conv" in layer_name:
        generate_random_values_conv(cdlt, model_name, layer_name, **kwargs)
    elif "maxpool" in layer_name or "max_pool" in layer_name:
        generate_random_values_maxpool(cdlt, model_name, layer_name, **kwargs)
    elif "global_avgpool" in layer_name or "global_avg_pool" in layer_name:
        generate_random_values_global_avgpool(cdlt, model_name, layer_name, **kwargs)
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

    save_array(f'{base_path}/input1_raw.txt', input1)

    save_array(f'{base_path}/input2_raw.txt', input2)



    if format.lower() == "nhwc" and len(input1.shape) == 4:
        input1 = input1.transpose((0, 3, 1, 2))
        input2 = input2.transpose((0, 3, 1, 2))


    output = binary(input1, input2, layer_name, f"{cdlt.inputs[0].dtype}")
    if len(output.shape) == 4:
        output = output.transpose((0, 2, 3, 1))

    if fixed_values is not None and "outputs" in fixed_values:
        np.testing.assert_allclose(output, fixed_values["outputs"])

    # Write outputs to file
    save_array(f'{base_path}/output.txt', output)

    # with open(f'{base_path}/output.txt', 'w') as f:
    #     if isinstance(output, Fxp):
    #         f.write('\n'.join([str(i) for i in output.val.flatten().tolist()]))
    #     else:
    #         f.write('\n'.join([str(i) for i in output.flatten().tolist()]))



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
        if "sigmoid" in cdlt.op_name:
            scale = 1.5
        else:
            scale = 1
        input1 = numpy_datagen(input_dims, cdlt.inputs[0].dtype.bits(), fxp_dtype=f"{cdlt.inputs[0].dtype}", scale=scale)

    else:
        input1 = np.zeros(input_dims, dtype=np.int64).reshape(-1)
        bits = (1 << (cdlt.inputs[0].dtype.bits() - 1)) - 1
        for i in range(np.prod(input_dims)):
            input1[i] = i % (bits)
        input1 = input1.reshape(input_dims)

    save_array(f'{base_path}/input1_raw.txt', input1)


    if format.lower() == "nhwc" and len(input1.shape) == 4:
        input1 = input1.transpose((0, 3, 1, 2))
    if "clip" in cdlt.op_name:
        minval = cdlt.required_params['min'].value
        maxval = cdlt.required_params['max'].value
        params = (minval, maxval)
    elif "pow" in cdlt.op_name:
        exp = cdlt.required_params['exp'].value
        params = (exp,)
    elif "reduce_mean" in cdlt.op_name:
        axis = cdlt.required_params['axis'].value
        params = (axis,)
    else:
        params = tuple([])


    output = unary(input1, layer_name, f"{cdlt.inputs[0].dtype}", *params)

    if len(output.shape) == 4:
        output = output.transpose((0, 2, 3, 1))

    if fixed_values is not None and "outputs" in fixed_values:
        np.testing.assert_allclose(output, fixed_values["outputs"])
    save_array(f'{base_path}/output.txt', output)


def generate_random_values_avg(cdlt, model_name, layer_name,
                                base_path=".",
                                format="nhwc",
                                use_random=True,
                                fixed_values=None,
                                actual_data=False,
                                generate_partial_values=False):

    input_dims = tuple(cdlt.inputs[0].tiling['DRAM'].values())
    KH = cdlt.required_params['KH'].value

    stride_x = cdlt.required_params['sx'].value
    # DRAM tiling is in level 1.

    if use_random:
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


    output = maxpool2d(input.astype(np.int64), KH, stride_x, padding=0)


    output = output.transpose((0, 2, 3, 1))
    if fixed_values is not None and "outputs" in fixed_values:
        np.testing.assert_allclose(output, fixed_values["outputs"])

    output = output.flatten().tolist()
    output = [str(x) for x in output]

    # Write outputs to file
    with open(f'{base_path}/output.txt', 'w') as f:
        f.write('\n'.join(output))



def generate_random_values_maxpool(cdlt, model_name, layer_name,
                                base_path=".",
                                format="nhwc",
                                use_random=True,
                                fixed_values=None,
                                actual_data=False,
                                generate_partial_values=False):

    input_dims = tuple(cdlt.inputs[0].tiling['DRAM'].values())
    KH = cdlt.required_params['KH'].value

    stride_x = cdlt.required_params['sx'].value
    # DRAM tiling is in level 1.

    if use_random:
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


    output = maxpool2d(input.astype(np.int64), KH, stride_x, padding=0)


    output = output.transpose((0, 2, 3, 1))
    if fixed_values is not None and "outputs" in fixed_values:
        np.testing.assert_allclose(output, fixed_values["outputs"])

    output = output.flatten().tolist()
    output = [str(x) for x in output]

    # Write outputs to file
    with open(f'{base_path}/output.txt', 'w') as f:
        f.write('\n'.join(output))

def generate_random_values_global_avgpool(cdlt, model_name, layer_name,
                                          base_path=".",
                                          format="nhwc",
                                          use_random=True,
                                          fixed_values=None,
                                          actual_data=False,
                                          generate_partial_values=False):

    input_dims = tuple(cdlt.inputs[0].tiling['DRAM'].values())

    # DRAM tiling is in level 1.

    if use_random:
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


    output = global_avg_pool(input.astype(np.int64), f"{cdlt.inputs[0].dtype}")


    output = output.transpose((0, 2, 3, 1))
    if fixed_values is not None and "outputs" in fixed_values:
        np.testing.assert_allclose(output, fixed_values["outputs"])

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

def generate_random_values_dw_conv(cdlt, model_name, layer_name,
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
        input = numpy_datagen(input_dims, cdlt.inputs[0].dtype.bits(), scale=2, fxp_dtype=f"{cdlt.inputs[0].dtype}")
        weights = numpy_datagen(weight_dims, cdlt.inputs[0].dtype.bits(), scale=2, fxp_dtype=f"{cdlt.inputs[0].dtype}")

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

    save_array(f"{base_path}/input.txt", input)
    save_array(f"{base_path}/weights.txt", weights)

    if format.lower() == "nhwc":
        input = input.transpose(0, 3, 1, 2)
        # Need to flip from (KH, KW, IC, OC) to (OC, IC, KH, KW)
        weights = weights.transpose(*tuple(WEIGHTS_CL_TO_CF))

    # test_output = manual_conv(input, weights, cdlt, layout="nchw")

    conv_param = {'stride': stride, 'pad': 0}
    # b = np.zeros(weights.shape[0], dtype=np.int32)
    output = depthwise_conv2d(input, weights, stride, 0, f"{cdlt.inputs[0].dtype}")
    # output, _ = conv_forward_im2col(input.astype(np.int32), weights.astype(np.int32), b, conv_param)


    # output, _ = conv_forward_naive(input.astype(np.int32), weights.astype(np.int32), b, conv_param)


    output = output.transpose(0, 2, 3, 1)
    if generate_partial_values:
        tinput = input.transpose(*tuple(ACT_CF_TO_CL))
        tweights = weights.transpose(*tuple(WEIGHTS_CF_TO_CL))
        coords = np.unravel_index(0, output.shape)

        partial_values_conv(cdlt, base_path, tinput, tweights, output, coords)

    if fixed_values is not None and "outputs" in fixed_values:
        np.testing.assert_allclose(output, fixed_values["outputs"])


    # output = output.flatten().tolist()
    # output = [str(x) for x in output]

    # Write outputs to file
    save_array(f"{base_path}/output.txt", output)
    # with open(f'{base_path}/output.txt', 'w') as f:
    #     f.write('\n'.join(output))

    # bias = bias.flatten().tolist()
    # bias = [str(x) for x in bias]
    #
    # # Write outputs to file
    # with open(f'{base_path}/bias.txt', 'w') as f:
    #     f.write('\n'.join(bias))

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
        input = numpy_datagen(input_dims, cdlt.inputs[0].dtype.bits(), scale=1, fxp_dtype=f"{cdlt.inputs[0].dtype}")
        weights = numpy_datagen(weight_dims, cdlt.inputs[0].dtype.bits(), scale=1, fxp_dtype=f"{cdlt.inputs[0].dtype}")

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
        f.write("\n".join(transform_data(input, "input", "shuffled", cdlt)))

        # f.write('\n'.join(dram_layout(input)))

    with open(f'{base_path}/input_raw.txt', 'w') as f:
        f.write("\n".join(transform_data(input, "input", "raw", cdlt)))

        # f.write('\n'.join([str(i) for i in input.flatten().tolist()]))

    with open(f'{base_path}/weights_shuffled.txt', 'w') as f:
        f.write("\n".join(transform_data(weights, "weights", "shuffled", cdlt)))


    with open(f'{base_path}/weights_shuffled_raw.txt', 'w') as f:
        f.write("\n".join(transform_data(weights, "weights", "shuffled_raw", cdlt)))


    with open(f'{base_path}/weights_raw.txt', 'w') as f:
        f.write("\n".join(transform_data(weights, "weights", "raw", cdlt)))




    if format.lower() == "nhwc":
        input = input.transpose(0, 3, 1, 2)
        # Need to flip from (KH, KW, IC, OC) to (OC, IC, KH, KW)
        weights = weights.transpose(*tuple(WEIGHTS_CL_TO_CF))

    # test_output = manual_conv(input, weights, cdlt, layout="nchw")

    conv_param = {'stride': stride, 'pad': 0}
    b = np.zeros(weights.shape[0], dtype=np.int32)

    # output, _ = conv_forward_im2col(input.astype(np.int32), weights.astype(np.int32), b, conv_param)


    output, _ = conv_forward_naive(input.astype(np.int32), weights.astype(np.int32), b, conv_param)


    output = output.transpose(0, 2, 3, 1)
    if generate_partial_values:
        tinput = input.transpose(*tuple(ACT_CF_TO_CL))
        tweights = weights.transpose(*tuple(WEIGHTS_CF_TO_CL))
        coords = np.unravel_index(0, output.shape)
        partial_values_conv(cdlt, base_path, tinput, tweights, output, coords)
    if fixed_values is not None and "outputs" in fixed_values:
        np.testing.assert_allclose(output, fixed_values["outputs"])


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
        input = numpy_datagen(input_dims, cdlt.inputs[0].dtype.bits(), scale=1, fxp_dtype=f"{cdlt.inputs[0].dtype}")
        weights = numpy_datagen(weight_dims, cdlt.inputs[1].dtype.bits(), scale=1, fxp_dtype=f"{cdlt.inputs[1].dtype}")

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
        f.write("\n".join(transform_data(input, "input", "shuffled", cdlt)))


    with open(f'{base_path}/input_raw.txt', 'w') as f:
        f.write("\n".join(transform_data(input, "input", "raw", cdlt)))


    with open(f'{base_path}/weights_shuffled.txt', 'w') as f:
        f.write("\n".join(transform_data(weights, "weights", "shuffled", cdlt)))


    with open(f'{base_path}/weights_shuffled_raw.txt', 'w') as f:
        f.write("\n".join(transform_data(weights, "weights", "shuffled_raw", cdlt)))


    with open(f'{base_path}/weights_raw.txt', 'w') as f:
        f.write("\n".join(transform_data(weights, "weights", "raw", cdlt)))


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
