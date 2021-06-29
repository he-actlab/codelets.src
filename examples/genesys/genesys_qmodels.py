import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, List
from . import GENESYS_CFG
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


def shuffle_weights(weights):
    # Layout of weights is in (KH, KW, OC, IC) format
    w_dim = weights.shape
    result = np.zeros(w_dim, dtype=weights.dtype)
    tile_m = GENESYS_CFG['ARRAY_M']
    tile_n = GENESYS_CFG['ARRAY_N']
    for kh in range(w_dim[0]):
        for kw in range(w_dim[1]):
            for mm in range(0, w_dim[2], tile_m):
                for nn in range(0, w_dim[3], tile_n):
                    for m in range(tile_m):
                        for n in range(tile_n):
                            # Reverse order within a tile because systolic array is filled from last column first.
                            result[kh][kw][mm + tile_m - m - 1][nn + n] = weights[kh][kw][mm + m][nn + n]
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

def gen_conv_testcase(input_dim, weight_dim, stride = 1, padding = 0, bias = False):
    # Input is (N, H, W, C)
    input = np.random.randint(low=0, high=127, size=input_dim, dtype=np.int8)
    # Weights is (KH, KW, OC, IC) layout
    weights = np.random.randint(low=0, high=127, size=weight_dim, dtype=np.int8)
    with open('input.txt', 'w') as f:
        f.write('\n'.join(dram_layout(input)))
    with open('weights.txt', 'w') as f:
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
    print(output)
    # Output from pytorch is (N, OC, H, W)
    # Reshape output as Genesys will generate output as (N, H, W, OC)
    output = output.permute(0, 2, 3, 1).numpy()
    output = output.flatten().tolist()
    output = [str(x) for x in output]
    # Write outputs to file
    with open('output.txt', 'w') as f:
        f.write('\n'.join(output))








