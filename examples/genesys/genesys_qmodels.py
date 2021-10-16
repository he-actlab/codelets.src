import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.nn.functional as F
from typing import Union, List
from . import GENESYS_CFG
from numpy.lib.stride_tricks import as_strided

from .genesys_model_utils import get_resnet18, get_resnet50
import numpy as np

WEIGHTS_CL_TO_CF = [3, 2, 0, 1] # (KH, KW, IC, OC) -> (OC, IC, KH, KW)
WEIGHTS_CF_TO_CL = [2, 3, 1, 0] # (OC, IC, KH, KW) -> (KH, KW, IC, OC)
ACT_CL_TO_CF = [0, 3, 1, 2] # (N, H, W, C) -> (N, C, H, W)
ACT_CF_TO_CL = [0, 2, 3, 1] # (N, C, H, W) -> (N, H, W, C)
# FLIP_SHAPE_PERM = [2, 3, 1, 0]
# FLIP_SHAPE_PERM = [2, 3, 0, 1]

class QLayer(nn.Module):
    def __init__(self, layer: str, input_width: Union[int, List] = None, output_width: Union[int, List] = None,
                 method: str = 'truncate', **kwargs):
        super(QLayer, self).__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.method = method
        assert method == 'truncate' or method == 'scale'
        self.layer = getattr(nn, layer)(**kwargs)

    def forward_scaled(self, input: Tensor) -> Tensor:
        if self.input_width is not None:
            # quantize_per_tensor works only on float tensors
            qinput = torch.quantize_per_tensor(input.float(), self.input_width[0], self.input_width[1],
                                               self.input_width[2])
            output = self.layer.forward(qinput.dequantize())
        else:
            # Compute forward using floats since lot of torch operators don't have int support on CPUs
            output = self.layer.forward(input.float())
        if self.output_width is not None:
            qoutput = torch.quantize_per_tensor(output, self.output_width[0], self.output_width[1],
                                                self.output_width[2])
            return qoutput.dequantize()
        else:
            return output.int()

    def forward_truncate(self, input: Tensor) -> Tensor:
        if self.input_width is not None:
            input_mask = torch.ones(input.size()) * ((1 << self.input_width) - 1)
            qinput = torch.bitwise_and(input, input_mask.int())
            output = self.layer.forward(qinput.float()).int()
        else:
            # Compute forward using floats since lot of torch operators don't have int support on CPUs
            output = self.layer.forward(input.float()).int()
        if self.output_width is not None:
            output_mask = torch.ones(output.size()) * ((1 << self.output_width) - 1)
            qoutput = torch.bitwise_and(output, output_mask.int())
            return qoutput
        else:
            return output

    def forward(self, input: Tensor) -> Tensor:
        if self.method == 'truncate':
            return self.forward_truncate(input)
        else:
            return self.forward_scaled(input)

    @property
    def weight(self):
        return self.layer.weight

    @property
    def bias(self):
        return self.layer.bias

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
        for mm in range(0, w_dim[1], tile_m):
            for nn in range(0, w_dim[0], tile_n):
                for m in range(tile_m):
                    for n in range(tile_n):
                        # Reverse order within a tile because systolic array is filled from last column first.
                        # Adjacent values in memory are filled in systolic array column.
                        # So, if systolic array size is 32x32, weight at (0, 0) should be in (31,0) in memory
                        # weight at (1, 0) should be in (31,1) in memory and so on.
                        dst_coord = (nn + n, mm + tile_m - m - 1)
                        src_coord = (nn+n, mm+m)
                        coord_map[src_coord] = dst_coord

                        result[nn + n][mm + tile_m - m - 1] = weights[nn + n][mm + m]
                        # result[kh][kw][nn + n][mm + tile_m - m - 1] = weights[kh][kw][nn + n][mm + m]
    return result, coord_map

# Sequentially write out tiles of weights which will be written in DRAM
# A tile is written out in column-major order.
# Column major order to enable a tile-size sequential read from DRAM to go to column of systolic array
def tiled_flatten(weights, dram_tiling, layer_type = 'gemm'):

    if isinstance(weights, tuple):
        weights, coord_map = weights
        rev_coords = {v: k for k,v in coord_map.items()}
    else:
        rev_coords = {}
    final_coords = {}
    result = list()
    tile_m = GENESYS_CFG['ARRAY_M']
    tile_n = GENESYS_CFG['ARRAY_N']
    w_dim = weights.shape
    if layer_type == 'gemm':
        big_tile_size = dram_tiling['P']
        for big_tile in range(0, w_dim[1], big_tile_size):
            for nn in range(0, w_dim[0], tile_n):
                for mm in range(0, big_tile_size, tile_m):
                    for m in range(tile_m):
                        for n in range(tile_n):
                            result.append(weights[nn + n][big_tile + mm + m])
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

    # tsize = 64*64*3*3
    # for i in range(9):
    #     print(f"{i*8192 + tsize} --> {absolute_coords[i*8192+ tsize]}")
    #     print(f"{np.unravel_index(i*8192+ tsize, weights.shape)} --> {np.unravel_index(absolute_coords[i*8192+ tsize][0], weights.shape)}\n")



    # print()
    # Interleave weights to maximize bandwidth use depending on array size and bandwidth
    # bw = GENESYS_CFG['PARAM_BUF_CHANNEL_BW'] // 8
    # systolic_array_row_size = weights.dtype.itemsize * tile_m
    # systolic_array_column_size = weights.dtype.itemsize * tile_n
    # interleave_factor = bw // systolic_array_column_size
    # assert interleave_factor >= 1
    # assert tile_n == tile_m
    # # Set of tile_n size elements to interleave
    # window = tile_n * interleave_factor
    # for i in range(0, len(result), window):
    #     print(i)
    #     interleaved_values = []
    #     for j in range(tile_n):
    #         for k in range(interleave_factor):
    #             interleaved_values.append(result[i + j + (k*tile_n)])
    #     result[i:i + window] = interleaved_values
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



def gen_conv_testcase(input_dim, weight_dim, stride=1, padding=0, base_path=".", bias=False):
    # Input is (N, H, W, C)
    input = np.random.randint(low=0, high=127, size=input_dim, dtype=np.int8)
    # Weights is (KH, KW, OC, IC) layout
    weights = np.random.randint(low=0, high=127, size=weight_dim, dtype=np.int8)
    with open(f'{base_path}/input.txt', 'w') as f:
        f.write('\n'.join(dram_layout(input)))
    with open(f'{base_path}/weights.txt', 'w') as f:
        f.write('\n'.join(dram_layout(shuffle_weights(weights))))


    model = QLayer('Conv2d', in_channels=weight_dim[3], out_channels=weight_dim[2], kernel_size=weight_dim[0],
                   stride=stride,
                   padding=padding, bias=bias)
    input_tensor = torch.from_numpy(input)
    input_tensor = input_tensor.float()
    model.weight.data = torch.from_numpy(weights)
    model.weight.data = model.weight.data.float()
    # Reshape as Conv2D layer in pytorch needs input as (N,C,H,W)
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    # Reshape as Conv2D layer in pytorch needs weight as (OC,IC,KH,KW)
    model.weight.data = model.weight.data.permute(2, 3, 0, 1)
    output = model(input_tensor)
    model.eval()
    # Output from pytorch is (N, OC, H, W)
    # Reshape output as Genesys will generate output as (N, H, W, OC)
    output = output.permute(0, 2, 3, 1).numpy()
    output = output.flatten().tolist()
    output = [str(x) for x in output]
    # Write outputs to file
    with open(f'{base_path}/output.txt', 'w') as f:
        f.write('\n'.join(output))


def gen_fc_layer_testcase(input_dim, output_dim, big_tile_size=1, bias=False):
    # Input is (N, *, H), output is (N, *, W)
    input = np.random.randint(low=0, high=127, size=input_dim, dtype=np.int8)
    # Weights is of dimension (H, W)
    weights = np.random.randint(low=0, high=127, size=(input_dim[-1], output_dim[-1]), dtype=np.int8)
    bias_values = np.zeros(output_dim[-1])
    if bias:
        bias_values = np.random.randint(low=0, high=127, size=output_dim[-1], dtype=np.int8)
        with open('bias.txt', 'w') as f:
            f.write('\n'.join(dram_layout(bias_values)))
    with open('input.txt', 'w') as f:
        f.write('\n'.join(dram_layout(input)))
    with open('weights_nondram.txt', 'w') as f:
        weights_nondram = tiled_flatten(shuffle_weights(weights, layer_type='linear'), big_tile_size).tolist()
        weights_nondram = [str(x) for x in weights_nondram]
        f.write('\n'.join(weights_nondram))
    with open('weights_raw.txt', 'w') as f:
        weights_raw = weights.flatten().tolist()
        weights_raw = [str(x) for x in weights_raw]
        f.write('\n'.join(weights_raw))
    with open('weights.txt', 'w') as f:
        final_weights = tiled_flatten(shuffle_weights(weights, layer_type='linear'), big_tile_size)
        f.write('\n'.join(dram_layout(final_weights)))
    model = QLayer('Linear', in_features=input_dim[-1], out_features=output_dim[-1], bias=bias)
    input_tensor = torch.from_numpy(input)
    input_tensor = input_tensor.float()
    # Pytorch layer weights are stored in transposed format.
    model.weight.data = torch.from_numpy(np.transpose(weights))
    model.weight.data = model.weight.data.float()
    output = model(input_tensor)
    model.eval()
    output = output.numpy()
    output = output.flatten().tolist()
    output = [str(x) for x in output]
    # Write outputs to file
    with open('output.txt', 'w') as f:
        f.write('\n'.join(output))


def gen_fc_layer_testcase_numpy(input_dim, output_dim, big_tile_size=1, bias=False):
    # Input is (N, *, H), output is (N, *, W)
    input = np.random.randint(low=0, high=127, size=input_dim, dtype=np.int8)
    # Weights is of dimension (H, W)
    weights = np.random.randint(low=0, high=127, size=(input_dim[-1], output_dim[-1]), dtype=np.int8)
    bias_values = np.zeros(output_dim[-1])
    if bias:
        bias_values = np.random.randint(low=0, high=127, size=output_dim[-1], dtype=np.int8)
        with open('bias.txt', 'w') as f:
            f.write('\n'.join(dram_layout(bias_values)))
    with open('input.txt', 'w') as f:
        f.write('\n'.join(dram_layout(input)))
    with open('weights_nondram.txt', 'w') as f:
        weights_nondram = tiled_flatten(shuffle_weights(weights, layer_type='linear'), big_tile_size).tolist()
        weights_nondram = [str(x) for x in weights_nondram]
        f.write('\n'.join(weights_nondram))
    with open('weights_raw.txt', 'w') as f:
        weights_raw = weights.flatten().tolist()
        weights_raw = [str(x) for x in weights_raw]
        f.write('\n'.join(weights_raw))
    with open('weights.txt', 'w') as f:
        final_weights = tiled_flatten(shuffle_weights(weights, layer_type='linear'), big_tile_size)
        f.write('\n'.join(dram_layout(final_weights)))

    result = np.matmul(input.astype(np.int32), weights.astype(np.int32))
    output = result.flatten().tolist()
    output = [str(x) for x in output]
    with open('output.txt', 'w') as f:
        f.write('\n'.join(output))


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    # assert (H + 2 * padding - field_height) % stride == 0
    # assert (W + 2 * padding - field_height) % stride == 0
    out_height = np.int32((H + 2 * padding - field_height) / stride + 1)
    out_width = np.int32((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def conv_forward_im2col(x, w, b, conv_param):
    """
    A fast implementation of the forward pass for a convolutional layer
    based on im2col and col2im.
    """
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions

    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    # Create output
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    # x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache

def pad_conv(layer_data):
    x = layer_data['input']
    out = layer_data['output']
    wgt = layer_data['params']['weight']
    b = layer_data['params']['bias']
    if x.shape[-1] % GENESYS_CFG['ARRAY_M'] != 0:
        ic_init = x.shape[-1]
        ic_pad = (GENESYS_CFG['ARRAY_M'] - ic_init) % GENESYS_CFG['ARRAY_M']
        assert (ic_pad + ic_init) % GENESYS_CFG['ARRAY_M'] == 0
        padding = (0, ic_pad)
        x_pad = ((0, 0), (0, 0), (0, 0), padding)

        x = np.pad(x, x_pad, "constant")
        assert wgt.shape[-1] == ic_init
        wgt = np.pad(wgt, x_pad, "constant")

    if out.shape[-1] % GENESYS_CFG['ARRAY_N'] != 0:
        oc_init = out.shape[-1]
        oc_pad = (GENESYS_CFG['ARRAY_N'] - oc_init) % GENESYS_CFG['ARRAY_N']
        assert (oc_pad + oc_init) % GENESYS_CFG['ARRAY_N'] == 0
        padding = (0, oc_pad)
        out_pad = ((0, 0), (0, 0), (0, 0), padding)
        out = np.pad(out, out_pad, "constant")
        assert wgt.shape[-2] == oc_init
        wgt_pad = ((0, 0), (0, 0), padding, (0, 0))
        wgt = np.pad(wgt, wgt_pad, "constant")
        assert b.shape[0] == oc_init
        b = np.pad(b, padding, "constant")
    return x, wgt, b, out


def pad_gemm(layer_data):
    x = layer_data['input']
    out = layer_data['output']
    wgt = layer_data['params']['weight']
    b = layer_data['params']['bias']
    if x.shape[-1] % GENESYS_CFG['ARRAY_M'] != 0:
        ic_init = x.shape[-1]
        ic_pad = (GENESYS_CFG['ARRAY_M'] - ic_init) % GENESYS_CFG['ARRAY_M']
        assert (ic_pad + ic_init) % GENESYS_CFG['ARRAY_M'] == 0
        padding = (0, ic_pad)
        x_pad = ((0, 0), padding)

        x = np.pad(x, x_pad, "constant")
        assert wgt.shape[0] == ic_init
        wgt_pad = (padding, (0, 0))
        wgt = np.pad(wgt, wgt_pad, "constant")

    if out.shape[-1] % GENESYS_CFG['ARRAY_N'] != 0:
        oc_init = out.shape[-1]
        oc_pad = (GENESYS_CFG['ARRAY_N'] - oc_init) % GENESYS_CFG['ARRAY_N']
        assert (oc_pad + oc_init) % GENESYS_CFG['ARRAY_N'] == 0
        padding = (0, oc_pad)
        out_pad = ((0, 0), padding)
        out = np.pad(out, out_pad, "constant")
        assert wgt.shape[-1] == oc_init
        wgt_pad = ((0, 0), padding)
        wgt = np.pad(wgt, wgt_pad, "constant")
        assert b.shape[0] == oc_init
        b = np.pad(b, padding, "constant")
    return x, wgt, b, out

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.
    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None

    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = x.shape
    F, C_filter, HH, WW = w.shape
    assert C == C_filter, 'Number of channels are not equal between input and filter'
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    H_new = int(1 + (H + 2*pad - HH) / stride)
    W_new = int(1 + (W + 2*pad - WW) / stride)

    out = np.zeros((N, F, H_new, W_new), dtype=x.dtype)

    last_row = H + 2*pad - HH + 1
    last_col = W + 2*pad - WW + 1

    for f in range(F):
        i_out = 0
        for i in range(0, last_row, stride):
            j_out = 0
            for j in range(0, last_col, stride):
                x_current = x_pad[:, :, i:(i+HH), j:(j+WW)]
                out[:, f, i_out, j_out] = np.dot(x_current.reshape((N, -1)), w[f].flatten()) + b[f]
                j_out += 1
            i_out += 1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache

def manual_gemm(inputs, weights, o_coord):
    M = inputs.shape[0]
    N = inputs.shape[1]
    P = weights.shape[1]
    outputs = np.zeros((M, P), dtype=np.int32)
    inputs = inputs.astype(np.int32)
    weights = weights.astype(np.int32)
    compilation_info = {i: [] for i in range(N)}
    for p in range(P):
        for n in range(N):
            for m in range(M):
                partial_sum = inputs[m, n] * weights[n, p]
                outputs[m, p] += partial_sum

                if (m, p) == o_coord:
                    all_coords = (m, n, p)
                    icoord = (m, n)
                    icoord_idx = np.ravel_multi_index([m, n], inputs.shape)
                    wcoord = (n, p)
                    wcoord_idx = np.ravel_multi_index([n, p], weights.shape)
                    ocoord = (m, p)
                    ocoord_idx = np.ravel_multi_index([m, p], outputs.shape)
                    compilation_info[n].append(
                        f'"{all_coords}", {ocoord_idx}, {icoord_idx}, {wcoord_idx}, {inputs[icoord]}, {weights[wcoord]}, {partial_sum}')
    return outputs, compilation_info

def manual_conv(inputs, weights, cdlt, o_coord, layout="nhwc"):
    if layout == "nhwc":
        N, IH, IW, IC = inputs.shape
        KH, KW, IC_, OC = weights.shape
        # KH, KW, OC, IC_ = weights.shape
        N_, OH, OW, OC_ = cdlt.outputs[0].shape
        out_shape = cdlt.outputs[0].shape
    else:
        N, IC, IH, IW = inputs.shape
        OC, IC_, KH, KW,  = weights.shape
        N_, OH, OW, OC_ = cdlt.outputs[0].shape
        out_shape = (N_, OC_, OH, OW)
    assert isinstance(o_coord, tuple) and len(o_coord) == 4
    assert N_ == N
    assert IC == IC_
    assert OC == OC_
    outputs = np.zeros(out_shape, dtype=np.int32)
    inputs = inputs.astype(np.int32)
    weights = weights.astype(np.int32)
    stride = cdlt.required_params['stride'].value
    compilation_info = {i: [] for i in range(IC)}
    if layout == "nhwc":
        for oc in range(OC):
            for n in range(N):
                for ic in range(IC):
                    for kh in range(KH):
                        for kw in range(KW):
                            for y in range(OH):
                                for x in range(OW):


                                    partial_sum = inputs[n, kh + y*stride, kw + x*stride, ic] * weights[kh, kw, ic, oc]
                                    outputs[n, y, x, oc] += partial_sum
                                    if (n, y, x, oc) == o_coord:
                                        all_coords = (oc, n, ic, kh, kw, y, x)
                                        icoord = (n, kh + y*stride, kw + x*stride, ic)
                                        icoord_idx = np.ravel_multi_index([n, kh + y*stride, kw + x*stride, ic], inputs.shape)
                                        wcoord = (kh, kw, ic, oc)
                                        wcoord_idx = np.ravel_multi_index([kh, kw, ic, oc], weights.shape)
                                        ocoord = (n, y, x, oc)
                                        ocoord_idx = np.ravel_multi_index([n, y, x, oc], outputs.shape)
                                        compilation_info[ic].append(f'"{all_coords}", {ocoord_idx}, {icoord_idx}, {wcoord_idx}, {inputs[icoord]}, {weights[wcoord]}, {partial_sum}')

                                    # outputs[n, y, x, oc] += inputs[n, kh + y*stride, kw + x*stride, ic] * weights[kh, kw, oc, ic]

    else:
        compilation_info = {}
        for oc in range(OC):
            for n in range(N):
                for ic in range(IC):
                    for kh in range(KH):
                        for kw in range(KW):
                            for y in range(OH):
                                for x in range(OW):
                                    outputs[n, oc, y, x] += inputs[n, ic, kh + y * stride, kw + x * stride] * weights[
                                        oc, ic, kh, kw]
        outputs = outputs.transpose(0, 2, 3, 1)

    return outputs, compilation_info

def generate_random_values(cdlt, model_name, layer_name, **kwargs):
    print(layer_name)
    if "conv" in layer_name:
        generate_random_values_conv(cdlt, model_name, layer_name, **kwargs)
    elif "maxpool" in layer_name or "max_pool" in layer_name:
        generate_random_values_maxpool(cdlt, model_name, layer_name, **kwargs)
    else:
        assert "gemm" in layer_name
        generate_random_values_gemm(cdlt, model_name, layer_name, **kwargs)

def pool2d(x, k, stride, padding=0):
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    N, C, H, W = x_padded.shape

    pool_height, pool_width = k, k

    # assert (H - pool_height) % stride == 0, 'Invalid height'
    # assert (W - pool_width) % stride == 0, 'Invalid width'

    out_height = (H - pool_height) // stride + 1
    out_width = (W - pool_width) // stride + 1

    x_split = x_padded.reshape(N * C, 1, H, W)
    x_cols = im2col_indices(x_split, pool_height, pool_width, padding=0, stride=stride)
    x_cols_argmax = np.argmax(x_cols, axis=0)
    x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
    out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)
    # cache = (x, x_cols, x_cols_argmax, pool_param)
    return out


def maxpool(image, f=2, s=2):
    '''
    Downsample `image` using kernel size `f` and stride `s`
    '''
    n_c, c, h_prev, w_prev = image.shape
    h = int((h_prev - f) / s) + 1
    w = int((w_prev - f) / s) + 1

    downsampled = np.zeros((n_c,c, h, w))
    for i in range(n_c):
        for j in range(c):
            # slide maxpool window over each part of the image and assign the max value at each step to the output
            curr_y = out_y = 0
            while curr_y + f <= h_prev:
                curr_x = out_x = 0
                while curr_x + f <= w_prev:
                    downsampled[i, j, out_y, out_x] = np.max(image[i,j, curr_y:curr_y + f, curr_x:curr_x + f])
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
    return downsampled

def generate_random_values_maxpool(cdlt, model_name, layer_name,
                                base_path=".",
                                format="nhwc",
                                use_random=True,
                                fixed_values=None,
                                actual_data=False,
                                generate_partial_values=False):
    input_dims = cdlt.inputs[0].shape
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
        input = np.random.randint(low=0, high=127, size=input_dims, dtype=np.int8)
    else:
        input = np.zeros(input_dims, dtype=np.int8).reshape(-1)
        for i in range(np.prod(input_dims)):
            input[i] = i % 128
        input = input.reshape(input_dims)

    with open(f'{base_path}/input_raw.txt', 'w') as f:
        f.write('\n'.join([str(i) for i in input.flatten().tolist()]))

    if format.lower() == "nhwc":
        input = input.transpose(0, 3, 1, 2)

    # test_output = manual_conv(input, weights, cdlt, layout="nchw")
    # tout = F.max_pool2d(torch)
    # output = pool2d(input.astype(np.int32), KH, stride_x, out_dims[1], out_dims[2], padding=pad)
    output = pool2d(input.astype(np.int32), KH, stride_x, padding=pad)
    print(input.shape)
    tout = F.max_pool2d(torch.from_numpy(input.astype(np.float64)),
                        kernel_size=KH, padding=pad)
    print(input.shape)
    print(tout.shape)
    # np.testing.assert_allclose(output, tout.numpy())
    # output = maxpool(input.astype(np.int32), f=KH, s=stride_x)
    # print(output.shape)

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
        input = np.random.randint(low=0, high=127, size=input_dims, dtype=np.int8)
        weights = np.random.randint(low=0, high=127, size=weight_dims, dtype=np.int8)
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
        f.write('\n'.join(dram_layout(tiled_flatten(shuffle_weights(weights, layer_type="conv"), dram_tiling, layer_type='conv'))))

    with open(f'{base_path}/weights_shuffled_raw.txt', 'w') as f:
        f.write('\n'.join([str(i) for i in tiled_flatten(shuffle_weights(weights, layer_type="conv"), dram_tiling, layer_type='conv').tolist()]))

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
        print(f"Here")
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
        input = np.random.randint(low=0, high=127, size=input_dims, dtype=np.int8)
        weights = np.random.randint(low=0, high=127, size=weight_dims, dtype=np.int8)
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
        f.write('\n'.join(dram_layout(tiled_flatten(shuffle_weights(weights, layer_type="gemm"), dram_tiling))))

    with open(f'{base_path}/weights_shuffled_raw.txt', 'w') as f:
        f.write('\n'.join([str(i) for i in tiled_flatten(shuffle_weights(weights, layer_type="gemm"), dram_tiling).tolist()]))

    with open(f'{base_path}/weights_raw.txt', 'w') as f:
        f.write('\n'.join([str(i) for i in weights.flatten().tolist()]))

    if bias is not None:
        bias = bias.flatten().tolist()
        bias = [str(x) for x in bias]
        with open(f'{base_path}/bias.txt', 'w') as f:
            f.write('\n'.join(bias))

    if output is None:
        output = np.dot(np.int32(input), np.int32(weights))

    # if generate_partial_values:
    #     partial_values_gemm(cdlt, base_path, input, weights, output, (0, 0))
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
