import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Union, List
from . import GENESYS_CFG
from .genesys_model_utils import get_resnet18, get_resnet50
import numpy as np


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


def shuffle_weights(weights, layer_type="conv"):
    # Layout of weights is in (KH, KW, OC, IC) format
    w_dim = weights.shape
    result = np.zeros(w_dim, dtype=weights.dtype)
    tile_m = GENESYS_CFG['ARRAY_M']
    tile_n = GENESYS_CFG['ARRAY_N']
    if layer_type == "conv":
        for kh in range(w_dim[0]):
            for kw in range(w_dim[1]):
                for mm in range(0, w_dim[2], tile_m):
                    for nn in range(0, w_dim[3], tile_n):
                        for m in range(tile_m):
                            for n in range(tile_n):
                                # Reverse order within a tile because systolic array is filled from last column first.
                                result[kh][kw][mm + tile_m - m - 1][nn + n] = weights[kh][kw][mm + m][nn + n]
                                # result[kh][kw][nn + n][mm + tile_m - m - 1] = weights[kh][kw][nn + n][mm + m]
    else:
        assert layer_type == "linear"

        for mm in range(0, w_dim[1], tile_m):
            for nn in range(0, w_dim[0], tile_n):
                for m in range(tile_m):
                    for n in range(tile_n):
                        # Reverse order within a tile because systolic array is filled from last column first.
                        result[nn + n][mm + tile_m - m - 1] = weights[nn + n][mm + m]
                                # result[kh][kw][nn + n][mm + tile_m - m - 1] = weights[kh][kw][nn + n][mm + m]
    return result

def dram_layout(weights):
    dram_weights = []
    assert weights.dtype == np.int8 or weights.dtype == np.uint8
    flat_weights = weights.flatten()
    flat_weights = flat_weights.astype(np.uint8)
    n = flat_weights.shape[0]
    assert n >= 4
    for i in range(0, n-4, 4):
        concat_weights = (flat_weights[i]) + \
                         (flat_weights[i + 1] << 8) + \
                         (flat_weights[i + 2] << 16) + \
                         (flat_weights[i + 3] << 24)
        dram_weights.append(concat_weights)
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

def gen_conv_testcase(input_dim, weight_dim, base_path=".", stride = 1, padding = 0, bias = False):
    # Input is (N, H, W, C)
    input = np.random.randint(low=0, high=127, size=input_dim, dtype=np.int8)
    # Weights is (KH, KW, OC, IC) layout
    weights = np.random.randint(low=0, high=127, size=weight_dim, dtype=np.int8)
    with open(f'{base_path}/input.txt', 'w') as f:
        f.write('\n'.join(dram_layout(input)))
    with open(f'{base_path}/weights.txt', 'w') as f:
        f.write('\n'.join(dram_layout(shuffle_weights(weights))))
    model = QLayer('Conv2d', in_channels=weight_dim[3], out_channels=weight_dim[2], kernel_size=weight_dim[0], stride=stride,
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

def generate_random_values(cdlt, layer_name, base_path=".", format="nhwc"):
    input_dims = cdlt.inputs[0].shape
    weight_dims = cdlt.inputs[1].shape
    stride = cdlt.required_params['stride'].value
    pad = cdlt.required_params['pad'].value

    input = np.random.randint(low=0, high=127, size=input_dims, dtype=np.int8)
    weights = np.random.randint(low=0, high=127, size=weight_dims, dtype=np.int8)
    assert "conv" in layer_name

    with open(f'{base_path}/input.txt', 'w') as f:
        f.write('\n'.join(dram_layout(input)))
    with open(f'{base_path}/weights.txt', 'w') as f:
        f.write('\n'.join(dram_layout(shuffle_weights(weights))))
    if format.lower() == "nhwc" and "conv" in layer_name:
        input = input.transpose(0, 3, 1, 2)
        weights = weights.transpose(2, 3, 0, 1)

    model = QLayer('Conv2d', in_channels=weight_dims[3], out_channels=weight_dims[2], kernel_size=weight_dims[0],
                   stride=stride,
                   padding=0,
                   bias=False)

    input_tensor = torch.from_numpy(input)
    input_tensor = input_tensor.float()
    model.weight.data = torch.from_numpy(weights)
    model.weight.data = model.weight.data.float()
    model.eval()
    output = model(input_tensor)
    # Output from pytorch is (N, OC, H, W)
    # Reshape output as Genesys will generate output as (N, H, W, OC)
    output = output.permute(0, 2, 3, 1).numpy()
    output = output.flatten().tolist()
    output = [str(x) for x in output]

    # Write outputs to file
    with open(f'{base_path}/output.txt', 'w') as f:
        f.write('\n'.join(output))

    # if len(cdlt.inputs) == 3:
    #     bias = model.bias.detach().numpy()
    #     print(bias.dtype)
    #     bias = bias.flatten().tolist()
    #     bias = [str(x) for x in bias]
    #     with open(f'{base_path}/bias.txt', 'w') as f:
    #         f.write('\n'.join(bias))

def get_model_values(model_name, layer_name, layer_num):

    if model_name == "resnet18":
        layer_data, model = get_resnet18(True, layer_name, layer_num)
    elif model_name == "resnet50":
        layer_data, model = get_resnet50(True, layer_name, layer_num)
    else:
        raise RuntimeError

    if "conv" in layer_name.lower():
        x, wgt, b, out = pad_conv(layer_data)
    else:
        assert "linear" in layer_name.lower()
        x, wgt, b, out = pad_gemm(layer_data)
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




