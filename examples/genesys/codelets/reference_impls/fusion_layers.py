from examples.genesys import OP_DTYPES, ASIC_CONFIG, \
    FXP_CONFIGS, QUANT_SCALE, SIGN_SHIFT, SW_PIPELINE_TEST
from collections import Iterable, namedtuple
from functools import partial
import numpy as np
from fxpmath import Fxp
from examples.genesys import FXP_CONFIGS
from . import ReferenceOp, quantize_np, create_operand_data, transform_data
DFG = namedtuple('DFG', ['op', 'args'])
WEIGHTS_CL_TO_CF = [3, 2, 0, 1] # (KH, KW, IC, OC) -> (OC, IC, KH, KW)
WEIGHTS_CF_TO_CL = [2, 3, 1, 0] # (OC, IC, KH, KW) -> (KH, KW, IC, OC)
ACT_CL_TO_CF = [0, 3, 1, 2] # (N, H, W, C) -> (N, C, H, W)
ACT_CF_TO_CL = [0, 2, 3, 1] # (N, C, H, W) -> (N, H, W, C)

class FusionOp(ReferenceOp):
    SYS_ARRAY_OPS = ["conv", "matmul", "gemm"]
    def __init__(self, fusion_name, cdlt, use_quantization=True):
        self.use_quantization = use_quantization
        self.fusion_name = fusion_name
        self.dfg = FUSION_OP_INFO[fusion_name]['dfg']
        operands = [i for i in cdlt.inputs]
        outputs = [o for o in cdlt.outputs]
        super().__init__(cdlt, operands, outputs)

    def fn_impl(self, inouts):
        res = self.eval_dfg(self.dfg, inouts)
        inouts['outputs'] = [res]
        return inouts


    def eval_dfg(self, dfg, inouts):
        args = []
        for i in dfg.args:
            if isinstance(i, DFG):
                a = self.eval_dfg(i, inouts)
            elif isinstance(i, str):
                assert i in self.cdlt.required_params, f"Unable to evaluate dfg op {dfg.op} with required param {i} in " \
                                                       f"fusion {self.fusion_name}. Possible params:\n" \
                                                       f"{list(self.cdlt.required_params.keys())}"
                a = self.cdlt.required_params[i].value
            else:
                assert isinstance(i, int)
                if dfg.op in FusionOp.SYS_ARRAY_OPS:
                    a = (inouts['inputs'][i], self.operands[i])
                else:
                    a = inouts['inputs'][i]
            args.append(a)
        args = tuple(args)
        if dfg.op == "div":
            out = self.div(*args)
        elif dfg.op == "mul":
            out = self.mul(*args)
        elif dfg.op == "add":
            out = self.add(*args)
        elif dfg.op == "sub":
            out = self.sub(*args)
        elif dfg.op == "pow":
            out = self.pow(*args)
        elif dfg.op == "sqrt":
            out = self.sqrt(*args)
        elif dfg.op == "square":
            out = self.square(*args)
        elif dfg.op == "matmul":
            out = self.matmul(inouts, *args)
        elif dfg.op == "conv":
            out = self.conv2d(inouts, *args)
        elif dfg.op == "relu":
            out = self.relu(*args)
        elif dfg.op == "leaky_relu":
            out = self.leaky_relu(*args)
        elif dfg.op == "clip":
            out = self.clipfn(*args)
        elif dfg.op == "depthwise_conv":
            out = self.dw_conv2d(*args)
        else:
            raise RuntimeError(f"Unsupported dfg op: {dfg.op}")
        return out

    def div(self, op1, op2):
        output = quantize_np(op1//op2, "FXP32")
        return output

    def mul(self, op1, op2):
        output = quantize_np(op1*op2, "FXP32")
        return output

    def add(self, op1, op2):
        return op1 + op2

    def sub(self, op1, op2):
        return op1 + op2

    def pow(self, data, exp):
        out = np.copy(data)
        for _ in range(exp - 1):
            out = quantize_np(out * data, "FXP32")
        return out

    def sqrt(self, data, exp):
        raise RuntimeError("Sqrt not yet implemented")

    def square(self, data):
        raise RuntimeError("Square not yet implemented")

    def matmul(self, inouts, data, wgt):
        # TODO: Need to fix this
        assert isinstance(data, tuple)
        assert isinstance(wgt, tuple)
        data, data_op = data
        wgt, wgt_op = wgt
        inouts["inputs"].append(
            create_operand_data(transform_data(data, "input", "shuffled", self.cdlt), data_op, fmt='shuffled'))
        inouts["inputs"].append(
            create_operand_data(transform_data(data, "input", "raw", self.cdlt), data_op, fmt='raw'))

        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "shuffled", self.cdlt), wgt_op, fmt='shuffled'))
        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "shuffled_raw", self.cdlt), wgt_op,
                                fmt='shuffled_raw'))
        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "raw", self.cdlt), wgt_op, fmt='raw'))

        output = np.dot(np.int32(data), np.int32(wgt))
        raise RuntimeError("Need to finish Matmul impl")

    def conv2d(self, inouts, data, wgt, bias, stride, pad):
        # Already padded data, so we can replace
        pad = 0
        assert isinstance(data, tuple)
        assert isinstance(wgt, tuple)
        data, data_op = data
        data = data.data
        wgt, wgt_op = wgt
        wgt = wgt.data

        bias, bias_op = bias
        bias = bias.data

        inouts["inputs"].append(
            create_operand_data(transform_data(data, "input", "shuffled", self.cdlt), data_op, fmt='shuffled'))
        inouts["inputs"].append(
            create_operand_data(transform_data(data, "input", "raw", self.cdlt), data_op, fmt='raw'))
        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "shuffled", self.cdlt), wgt_op, fmt='shuffled'))
        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "shuffled_raw", self.cdlt), wgt_op,
                                fmt='shuffled_raw'))
        inouts["inputs"].append(
            create_operand_data(transform_data(wgt, "weights", "raw", self.cdlt), wgt_op, fmt='raw'))

        x = data.transpose(0, 3, 1, 2)
        w = wgt.transpose(*tuple(WEIGHTS_CL_TO_CF))
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
                    out[:, f, i_out, j_out] = np.dot(x_current.reshape((N, -1)), w[f].flatten()) + bias[f]
                    j_out += 1
                i_out += 1

        out = out.transpose(0, 2, 3, 1)
        return out

    def relu(self, inpt):
        output = np.maximum(inpt, 0, inpt)
        return output

    def leaky_relu(self, xval, alpha):
        dtype = "FXP32"
        if not isinstance(xval, Iterable):
            xval = np.asarray([xval])
        pw1 = Fxp(1.0, **FXP_CONFIGS[dtype])

        alpha_val = Fxp(alpha, **FXP_CONFIGS[dtype]).val
        one_val = Fxp(1.0, **FXP_CONFIGS[dtype]).val
        conds = [
            (xval <= 0),
            (xval > 0)
        ]

        fns = [
            lambda x: x * alpha_val,
            lambda x: x * one_val
        ]

        res = np.piecewise(xval, conds, fns)
        return res

    def clipfn(self, data, minval, maxval):
        return np.clip(data, maxval, minval)

    def dw_conv2d(self, data, w, b, stride, pad):
        dtype = "FXP32"
        padded_input = np.pad(data,
                              pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)),
                              mode='constant',
                              constant_values=0)
        padded_input = padded_input.astype(np.int64)

        kh, kw = w.shape[2], w.shape[3]
        batch, in_depth, height, width = data.shape
        assert in_depth == w.shape[0]
        oh = int(1 + (height + 2 * pad - kh) / stride)
        ow = int(1 + (width + 2 * pad - kw) / stride)
        output = np.zeros((batch, in_depth, oh, ow)).astype(np.int64)

        for n in range(batch):
            for c in range(in_depth):
                # For each input channel separately, apply its corresponsing filter
                # to the input.
                for i in range(oh):
                    for j in range(ow):
                        for fi in range(kh):
                            for fj in range(kw):
                                w_element = w[c, 0, fi, fj]
                                mul_res = padded_input[n, c, i * stride + fi, j * stride + fj] * w_element
                                mul_res_quant = quantize_np(mul_res, dtype)
                                # mul_res_quant = mul_res
                                output[n, c, i, j] += mul_res_quant
                output[n, c] += b[c]
        return output


if SW_PIPELINE_TEST:
    FUSION_OP_INFO = {
        'div_add': {
            'cdlt': partial(FusionOp, 'div_add'),
            'dfg' : DFG('add', [DFG('div', [0, 'mul_rhs']),
                                1]),
            'seq': ["Div",
                    "Add"]
        },
        'add_relu': {
            'cdlt': partial(FusionOp, 'add_relu'),
            'dfg': DFG('relu', [DFG('add', [0, 1])]),
            'seq': ['Add', 'Relu'],
        },
        'add_leaky_relu': {
            'cdlt': partial(FusionOp, 'add_leaky_relu'),
            'dfg': DFG('leaky_relu', [DFG('add', [0, 1]), 'alpha']),
            'seq': ['Add', 'LeakyRelu'],
        },
        'leaky_relu_add': {
            'cdlt': partial(FusionOp, 'div_add'),
            'dfg': DFG('add', [DFG('leaky_relu', [0, 'alpha']), 1]),

            'seq': ['LeakyRelu', 'Add'],
        },
        'clip_depthwise_conv_bias': {
            'cdlt': partial(FusionOp, 'clip_depthwise_conv_bias'),
            'dfg': DFG('depthwise_conv',
                       [DFG('clip', [0, 'min', 'max']), 1, 2, 'stride', 'pad']),
            'seq': ['Clip', 'DepthwiseConv'],

        },
        'clip_depthwise_conv_bias_clip': {
            'cdlt': partial(FusionOp, 'clip_depthwise_conv_bias_clip'),
            'dfg': DFG('clip',
                       [DFG('depthwise_conv',
                            [DFG('clip', [0, 'min', 'max']), 1, 2, 'stride', 'pad']), 'min', 'max']),
            'seq': ['Clip', 'DepthwiseConv', 'Clip'],

        },
        'add_add': {
          'cdlt': partial(FusionOp, 'add_add'),
            'dfg': DFG('add', [DFG('add', [0, 1]), 2]),
            'seq': ["Add", "Add"]
        },
        'add_add4d': {
            'cdlt': partial(FusionOp, 'add_add4d'),
            'dfg': DFG('add', [DFG('add', [0, 1]), 2]),
            'seq': ["Add", "Add"]
        },
        'mul_add': {
            'cdlt': partial(FusionOp, 'mul_add'),
            'dfg': DFG('add', [DFG('mul', [0, 1]), 2]),
            'seq': ["Mul", "Add"]
        },
        'sub_mul': {
            'cdlt': partial(FusionOp, 'sub_mul'),
            'dfg': DFG('mul', [DFG('sub', [0, "sub_rhs"]), "mul_rhs"]),
            'seq': ["Sub", "Mul"]
        },
        'sub_pow': {
            'cdlt': partial(FusionOp, 'sub_pow'),
            'dfg': DFG('square', [DFG('sub', [0, 1])]),
            'seq': ["Sub", "Pow"],
        },
        'add_sqrt_div': {
            'cdlt': partial(FusionOp, 'add_sqrt_div'),
            'dfg': DFG('div', [DFG('sqrt', [
                                            DFG('add', [0, 'add_lhs'])
                                        ]),
                               1]),

            'seq': ["Add", "Sqrt", "Div"],
        },
        'matmul_add': {
            'cdlt': partial(FusionOp, 'matmul_add'),
            'dfg': DFG('gemm', [0, 1, 2]),
            'seq': ["MatMul", "Add"]
        },

        'single_layer_info':
            {
                'Conv' : {'inputs': 3, 'outputs': 1},
                'Relu' : {'inputs': 1, 'outputs': 1},
                'LeakyRelu' : {'inputs': 1, 'outputs': 1},
                'Add' : {'inputs': 2, 'outputs': 1},
                'MaxPool': {'inputs': 1, 'outputs': 1}
            }
    }
else:
    FUSION_OP_INFO = {
    'add_add': {
      'cdlt': partial(FusionOp, 'add_add'),
        'dfg': DFG('add', [DFG('add', [0, 1]), 2]),
        'seq': ["Add", "Add"]
    },
    'mul_add': {
        'cdlt': partial(FusionOp, 'mul_add'),
        'dfg': DFG('add', [DFG('mul', [0, 1]), 2]),
        'seq': ["Mul", "Add"]
    },
    'sub_mul': {
        'cdlt': partial(FusionOp, 'sub_mul'),
        'dfg': DFG('mul', [DFG('sub', [0, "sub_rhs"]), "mul_rhs"]),
        'seq': ["Sub", "Mul"]
    },

    'sub_pow': {
        'cdlt': partial(FusionOp, 'sub_pow'),
        'dfg': DFG('square', [DFG('sub', [0, 1])]),
        'seq': ["Sub", "Pow"],
    },
    'add_sqrt_div': {
        'cdlt': partial(FusionOp, 'add_sqrt_div'),
        'dfg': DFG('div', [DFG('sqrt', [
            DFG('add', [0, 'add_lhs'])
        ]),
                           1]),
        'seq': ["Add", "Sqrt", "Div"],
    },
    'matmul_add': {
        'cdlt': partial(FusionOp, 'matmul_add'),
        'dfg': DFG('gemm', [0, 1, 2]),
        'seq': ["MatMul", "Add"]
    },
    'matmul_add_add': {
        'cdlt': partial(FusionOp, 'matmul_add_add'),
        'dfg': DFG('add', [DFG('gemm', [0, 1, 2]), 3]),
        'seq': ["MatMul", "Add", "Add"]
    },
    'matmul_add_gelu': {
        'cdlt': partial(FusionOp, 'matmul_add_gelu'),
        'dfg': DFG('gelu', [DFG('gemm', [0, 1, 2])]),
        'seq': ["MatMul", "Add", "Gelu"]
    },
    'matmul_div_add': {
        'cdlt': partial(FusionOp, 'matmul_div_add'),
        'dfg': DFG('add', [DFG('mul', [DFG('matmul', [0, 1]), 'mul_rhs']), 2]),
        'seq': ["MatMul", "Div", "Add"]
    },
    'conv_relu': {
        'cdlt': partial(FusionOp, 'conv_relu'),
        'dfg': DFG('relu', [DFG('conv', [0,1,2,'stride', 'pad'])]),
        'seq': ['Conv', 'Relu']
    },
    'conv_bias_relu': {
        'cdlt': partial(FusionOp, 'conv_relu'),
        'dfg': DFG('relu', [DFG('conv', [0,1,2,'stride', 'pad'])]),
        'seq': ['Conv', 'Relu']
    },
    'conv_bias_add_relu': {
        'cdlt': partial(FusionOp, 'conv_add_relu'),
        'dfg' : DFG('relu', [DFG('add', [DFG('conv', [0,1,2, 'stride', 'pad']), 3])]),
        'seq': ['Conv', 'Add', 'Relu'],
    },
    'conv_bias_add': {
        'cdlt': partial(FusionOp, 'conv_add'),
        'dfg': DFG('add', [DFG('conv', [0,1,2,'stride', 'pad']), 3]),
        'seq': ['Conv', 'Add'],
    },
    'conv_bias_clip': {
        'cdlt': partial(FusionOp, 'conv_clip'),
        'dfg': DFG('clip', [DFG('conv', [0, 1, 2, 'stride', 'pad']), 'min', 'max']),
        'seq': ['Conv', 'Clip'],
    },
    'conv_clip': {
        'cdlt': partial(FusionOp, 'conv_clip'),
        'dfg': DFG('clip', [DFG('conv', [0, 1, 2, 'stride', 'pad']), 'min', 'max']),
        'seq': ['Conv', 'Clip'],
    },
    'depthwise_conv_bias_clip': {
        'cdlt': partial(FusionOp, 'depthwise_conv_clip'),
        'dfg': DFG('clip', [DFG('depthwise_conv', [0, 1, 2, 'stride', 'pad']), 'min', 'max']),
        'seq': ['DepthwiseConv', 'Clip'],
    },
    'conv_bias_leaky_relu': {
        'cdlt': partial(FusionOp, 'conv_leaky_relu'),
        'dfg': DFG('leaky_relu', [DFG('conv', [0, 1, 2, 'stride', 'pad']), 'alpha']),
        'seq': ['Conv', 'LeakyRelu']
    },
    'conv_bias_add_leaky_relu': {
        'cdlt': partial(FusionOp, 'conv_add_leaky_relu'),
        'dfg': DFG('leaky_relu', [DFG('add', [DFG('conv', [0, 1, 2, 'stride', 'pad']), 3]), 'alpha']),
        'seq': ['Conv', 'Add', 'LeakyRelu'],
    },
    'conv_bias_leaky_relu_add': {
        'cdlt': partial(FusionOp, 'conv_leaky_relu_add'),
        'dfg': DFG('add', [DFG('leaky_relu', [DFG('conv', [0, 1, 2, 'stride', 'pad']), 'alpha']), 3]),

        'seq': ['Conv', 'LeakyRelu', 'Add'],
    },
    'conv_bias_clip_depthwise_conv_bias': {
        'cdlt': partial(FusionOp, 'conv_bias_clip_depthwise_conv_bias'),
        'dfg': DFG('depthwise_conv', [
            DFG('clip', [
                DFG('conv', [0, 1, 2, 's1', 'p1']), 'min', 'max']),
        3, 4, 's2', 'p2']),
        'seq': ['Conv', 'Clip', 'DepthwiseConv'],

    },
    'conv_bias_clip_depthwise_conv_bias_clip': {
        'cdlt': partial(FusionOp, 'conv_bias_clip_depthwise_conv_bias_clip'),
        'dfg': DFG('clip', [DFG('depthwise_conv', [
            DFG('clip', [
                DFG('conv', [0, 1, 2, 's1', 'p1']), 'min', 'max']),
            3, 4, 's2', 'p2']), 'min', 'max']),
        'seq': ['Conv', 'Clip', 'DepthwiseConv', 'Clip'],

    },
    'single_layer_info':
        {
            'Conv' : {'inputs': 3, 'outputs': 1},
            'Relu' : {'inputs': 1, 'outputs': 1},
            'LeakyRelu' : {'inputs': 1, 'outputs': 1},
            'Add' : {'inputs': 2, 'outputs': 1},
            'MaxPool': {'inputs': 1, 'outputs': 1}
        }
}
FUSION_IMPLS = {k : v['cdlt'] for k, v in FUSION_OP_INFO.items() if k != 'single_layer_info'}
