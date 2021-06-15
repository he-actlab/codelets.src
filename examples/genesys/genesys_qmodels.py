import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, List


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
