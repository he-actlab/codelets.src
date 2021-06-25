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
    # Assumption weights are in (KH, KW, OC, IC) format
    w_dim = weights.shape
    # The result is organized as (KH, KW, IC, OC)
    result = np.zeros(w_dim, dtype=weights.dtype)
    tile_m = GENESYS_CFG['ARRAY_M']
    tile_n = GENESYS_CFG['ARRAY_N']
    for kh in range(w_dim[0]):
        for kw in range(w_dim[1]):
            for mm in range(0, w_dim[2], tile_m):
                for nn in range(0, w_dim[3], tile_n):
                    for m in range(tile_m):
                        for n in range(tile_n):
                            # Reverse the innermost tile because last column is filled first in systolic array
                            result[kh][kw][mm + m][nn + tile_n - n - 1] = weights[kh][kw][mm + m][nn + n]
    return result


