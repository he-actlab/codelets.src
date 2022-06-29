from typing import List
from functools import partial
import numpy as np
from . import ReferenceOp, quantize_np, create_operand_data, transform_data
WEIGHTS_CL_TO_CF = [3, 2, 0, 1] # (KH, KW, IC, OC) -> (OC, IC, KH, KW)
WEIGHTS_CF_TO_CL = [2, 3, 1, 0] # (OC, IC, KH, KW) -> (KH, KW, IC, OC)
ACT_CL_TO_CF = [0, 3, 1, 2] # (N, H, W, C) -> (N, C, H, W)
ACT_CF_TO_CL = [0, 2, 3, 1] # (N, C, H, W) -> (N, H, W, C)

class Conv(ReferenceOp):

    def __init__(self, cdlt, program, use_bias=True, use_quantization=True):
        self.use_bias = use_bias
        self.use_quantization = use_quantization
        operands = [cdlt.inputs[0], cdlt.inputs[1], cdlt.inputs[2]]
        outputs = [cdlt.outputs[0]]
        self.stride = cdlt.required_params['stride'].value
        super().__init__(cdlt, operands, outputs, program, scale=1)

    @property
    def data(self):
        return self.operands[0]

    @property
    def weight(self):
        return self.operands[1]

    @property
    def bias(self):
        return self.operands[2]

    def fn_impl(self, inouts):

        data = inouts['inputs'][0].data
        wgt = inouts['inputs'][1].data
        bias = inouts['inputs'][2].data

        inouts["inputs"].append(
            create_operand_data(transform_data(data, "input", "shuffled", self.cdlt, self.hag), self.data, fmt='shuffled'))
        inouts["inputs"].append(
            create_operand_data(transform_data(data, "input", "raw", self.cdlt, self.hag), self.data, fmt='raw'))
        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "shuffled", self.cdlt, self.hag), self.weight, fmt='shuffled'))
        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "shuffled_raw", self.cdlt, self.hag), self.weight,
                                fmt='shuffled_raw'))
        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "raw", self.cdlt, self.hag), self.weight, fmt='raw'))

        data = data.transpose(0, 3, 1, 2)
        wgt = wgt.transpose(*tuple(WEIGHTS_CL_TO_CF))
        output = self.conv_forward(data, wgt, bias, self.stride, 0)
        output = output.transpose(0, 2, 3, 1)
        inouts['outputs'] = [output]
        assert output.shape == self.cdlt.outputs[0].shape, "Output shape is incorrect:\n" \
                                                           f"Operand shape: {self.cdlt.outputs[0].shape}\n" \
                                                           f"Data shape: {output.shape}"
        return inouts


    def conv_forward(self, x, w, b, stride, pad):

        N, C, H, W = x.shape
        F, C_filter, HH, WW = w.shape
        assert C == C_filter, 'Number of channels are not equal between input and filter'
        ###########################################################################
        # TODO: Implement the convolutional forward pass.                         #
        # Hint: you can use the function np.pad for padding.                      #
        ###########################################################################
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

        H_new = int(1 + (H + 2 * pad - HH) / stride)
        W_new = int(1 + (W + 2 * pad - WW) / stride)

        out = np.zeros((N, F, H_new, W_new), dtype=x.dtype)

        last_row = H + 2 * pad - HH + 1
        last_col = W + 2 * pad - WW + 1

        for f in range(F):
            i_out = 0
            for i in range(0, last_row, stride):
                j_out = 0
                for j in range(0, last_col, stride):
                    x_current = x_pad[:, :, i:(i + HH), j:(j + WW)]
                    out[:, f, i_out, j_out] = np.dot(x_current.reshape((N, -1)), w[f].flatten()) + b[f]
                    j_out += 1
                i_out += 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return out


class Gemm(ReferenceOp):

    def __init__(self, cdlt, program, use_bias=True, use_quantization=True):
        self.use_bias = use_bias
        self.use_quantization = use_quantization
        if self.use_bias:
            operands = [cdlt.inputs[0], cdlt.inputs[1], cdlt.inputs[2]]
        else:
            operands = [cdlt.inputs[0], cdlt.inputs[1]]
        outputs = [cdlt.outputs[0]]
        super().__init__(cdlt, operands, outputs, program, scale=1)

    @property
    def data(self):
        return self.operands[0]

    @property
    def weight(self):
        return self.operands[1]

    @property
    def bias(self):
        return self.operands[2]

    def fn_impl(self, inouts):
        data = inouts['inputs'][0].data
        wgt = inouts['inputs'][1].data


        inouts["inputs"].append(
            create_operand_data(transform_data(data, "input", "shuffled", self.cdlt, self.hag), self.data, fmt='shuffled'))
        inouts["inputs"].append(
            create_operand_data(transform_data(data, "input", "raw", self.cdlt, self.hag), self.data, fmt='raw'))
        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "shuffled", self.cdlt, self.hag), self.weight, fmt='shuffled'))
        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "shuffled_raw", self.cdlt, self.hag), self.weight,
                                fmt='shuffled_raw'))
        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "raw", self.cdlt, self.hag), self.weight, fmt='raw'))

        output = np.dot(np.int32(data), np.int32(wgt))
        if self.use_bias:
            output = output + inouts['inputs'][2].data
        inouts['outputs'] = [output]
        return inouts


def load_sa_impls(cfg):

    if cfg['USE_QUANTIZATION']:
        SA_IMPLS = {
            "conv_bias": partial(Conv, use_bias=True, use_quantization=True),
            "conv": partial(Conv, use_bias=False, use_quantization=True),
            "gemm": partial(Gemm, use_bias=True, use_quantization=True),
            'matmul': partial(Gemm, use_bias=False, use_quantization=True),
            'matmul2d': partial(Gemm, use_bias=False, use_quantization=True),
            'matmul3d': partial(Gemm, use_bias=False, use_quantization=True),
            'matmul4d': partial(Gemm, use_bias=False, use_quantization=True),
            'matmul4d2d': partial(Gemm, use_bias=False, use_quantization=True)
        }
    else:
        SA_IMPLS = {
            "conv_bias": partial(Conv, use_bias=True, use_quantization=False),
            "conv": partial(Conv, use_bias=False, use_quantization=False),
            "gemm": partial(Gemm, use_bias=True, use_quantization=False),
            'matmul2d': partial(Gemm, use_bias=False, use_quantization=True),
            'matmul': partial(Gemm, use_bias=False, use_quantization=False),
            'matmul3d': partial(Gemm, use_bias=False, use_quantization=False),
            'matmul4d': partial(Gemm, use_bias=False, use_quantization=False),
            'matmul4d2d': partial(Gemm, use_bias=False, use_quantization=False)

        }
    return SA_IMPLS