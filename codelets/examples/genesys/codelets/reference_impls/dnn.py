from codelets.examples.genesys import FXP_CONFIGS
from codelets.examples.genesys.codelets.util import range_from_cfg, as_fxp, fxp_floor

from functools import partial
from fxpmath import Fxp
import numpy as np
from . import ReferenceOp, quantize_np, \
    im2col_indices, pad_tensor, get_slice

import itertools
WEIGHTS_CL_TO_CF = [3, 2, 0, 1] # (KH, KW, IC, OC) -> (OC, IC, KH, KW)
WEIGHTS_CF_TO_CL = [2, 3, 1, 0] # (OC, IC, KH, KW) -> (KH, KW, IC, OC)
ACT_CL_TO_CF = [0, 3, 1, 2] # (N, H, W, C) -> (N, C, H, W)
ACT_CF_TO_CL = [0, 2, 3, 1] # (N, C, H, W) -> (N, H, W, C)


class Pool(ReferenceOp):

    def __init__(self, pool_type, cdlt, program):
        self.pool_type = pool_type
        operands = [cdlt.inputs[0],]
        outputs = [cdlt.outputs[0]]
        self.dtype = "FXP32"
        super().__init__(cdlt, operands, outputs, program, scale=1)

    def fn_impl(self, inouts):
        data = inouts['inputs'][0].data
        k = self.cdlt.required_params['KH'].value
        stride = self.cdlt.required_params['sx'].value

        data = data.transpose(0, 3, 1, 2)
        output = self.pool2(data, k, stride, 0, self.pool_type)

        output = output.transpose(*tuple(ACT_CF_TO_CL))
        inouts['outputs'] = [output]

        return inouts


    def max_pool_(self, x, k, stride, pad):
        x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
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

        return out


    def avg_pool_(self, x, k, stride, pad):
        x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        N, C, H, W = x_padded.shape

        pool_height, pool_width = k, k

        # assert (H - pool_height) % stride == 0, 'Invalid height'
        # assert (W - pool_width) % stride == 0, 'Invalid width'

        out_height = (H - pool_height) // stride + 1
        out_width = (W - pool_width) // stride + 1

        x_split = x_padded.reshape(N * C, 1, H, W)
        x_cols = im2col_indices(x_split, pool_height, pool_width, padding=0, stride=stride)
        # x_cols_argmax = np.argmax(x_cols, axis=0)
        x_cols_sum = np.sum(x_cols, axis=0)

        denom = Fxp(1.0 / (x_cols.shape[0]), **FXP_CONFIGS[self.dtype]).val.item()

        x_cols_mean = quantize_np(x_cols_sum*denom, self.dtype)
        out = x_cols_mean.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)


        return out

    def pool2(self, np_data, k, stride, pad, pool_type, dilation=(1, 1), ceil_mode=False, count_include_pad=True):
        dtype = np_data.dtype
        kernel = (k, k)
        padding_before = [0, 0, pad, pad]
        padding_after = [0, 0, pad, pad]
        strides = (stride, stride)
        out_shape = [np_data.shape[0], np_data.shape[1]]
        for dim in range(2, len(np_data.shape)):
            i = dim - 2
            val = (
                    float(
                        np_data.shape[dim]
                        - (kernel[i] - 1) * dilation[i]
                        - 1
                        + padding_before[i]
                        + padding_after[i]
                    )
                    / strides[i]
            )

            if ceil_mode:
                out_shape.append(int(np.ceil(val) + 1))
            else:
                out_shape.append(int(np.floor(val) + 1))

        out_shape = tuple(out_shape)

        pad_value = 0
        if pool_type == "max" and not count_include_pad:
            pad_value = -float('inf')
        pad_data = pad_tensor(np_data, pad_value, padding_before, padding_after, dtype)
        pad_map = pad_tensor(np.ones_like(np_data), 0, padding_before, padding_after, "bool")

        dim_iterators = []
        for spatial_dimension in range(2, len(np_data.shape)):
            dim_iterators.append(range(out_shape[spatial_dimension]))
        coord_iterator = itertools.product(*dim_iterators)

        ret_np = np.zeros(shape=out_shape).astype(dtype)



        for coordinate in coord_iterator:
            # Get index into the values that any pool operation will use for given coordinate
            np_index = get_slice(
                spatial_dimensions=len(out_shape) - 2,
                pad_np=pad_data,
                dim_coord=coordinate,
                kernel=kernel,
                strides=strides,
                dilation=dilation,
            )

            output_slice = tuple([slice(None), slice(None)] + list(coordinate))
            reduction_axis = tuple(range(2, len(np_data.shape)))
            if pool_type == "avg":
                count_non_padded = (
                    pad_data[np_index].size if count_include_pad else np.sum(pad_map[np_index])
                )
                # We summed over the non-spatial dimensions too so divide by them
                count_non_padded /= out_shape[0] * out_shape[1]
                denom = Fxp(1.0 / (count_non_padded), **FXP_CONFIGS[self.dtype]).val.item()
                if count_non_padded == 0:
                    ret_np[output_slice] = 0
                else:
                    # ret_np[output_slice] = (
                    #         np.sum(pad_data[np_index], axis=reduction_axis)
                    # ) / count_non_padded

                    ret_np[output_slice] = (
                        np.sum(pad_data[np_index], axis=reduction_axis)
                    )
                    ret_np[output_slice] = quantize_np(ret_np[output_slice] * denom, self.dtype)


            elif pool_type == "max":
                count_non_padded = np.sum(pad_map[np_index])
                # All padded values, default to 0
                ret_np[output_slice] = np.max(pad_data[np_index], axis=reduction_axis)
            else:
                raise ValueError("Pool type {} is not supported".format(pool_type))


        return ret_np


    def pool1(self, x, k, stride, pad):
        method = 'avg'


        m, n = x.shape[:2]
        ky, kx = k, k
        if stride is None:
            stride = (ky, kx)
        sy, sx = stride, stride

        _ceil = lambda x, y: int(np.ceil(x / float(y)))

        if pad:
            ny = _ceil(m, sy)
            nx = _ceil(n, sx)
            size = ((ny - 1) * sy + ky, (nx - 1) * sx + kx) + x.shape[2:]
            mat_pad = np.full(size, np.nan)
            mat_pad[:m, :n, ...] = x
        else:
            mat_pad = x[:(m - ky) // sy * sy + ky, :(n - kx) // sx * sx + kx, ...]

        view = self.as_stride(mat_pad, (ky, kx), (sy, sx))

        if method == 'max':
            result = np.nanmax(view, axis=(2, 3))
        else:
            result = np.nanmean(view, axis=(2, 3))
        return result

    def as_stride(self, arr, sub_shape, stride):
        '''Get a strided sub-matrices view of an ndarray.
        See also skimage.util.shape.view_as_windows()
        '''
        s0, s1 = arr.strides[:2]
        m1, n1 = arr.shape[:2]
        m2, n2 = sub_shape
        view_shape = (1 + (m1 - m2) // stride[0], 1 + (n1 - n2) // stride[1], m2, n2) + arr.shape[2:]
        strides = (stride[0] * s0, stride[1] * s1, s0, s1) + arr.strides[2:]
        subs = np.lib.stride_tricks.as_strided(arr, view_shape, strides=strides)
        return subs

class Softmax(ReferenceOp):

    def __init__(self, cdlt, program):
        operands = [cdlt.inputs[0],]
        outputs = [cdlt.outputs[0]]
        self.dtype = "FXP32"
        a, b, c = 0.3585, 1.353, 0.344
        s = 0.6
        self.qln2 = as_fxp(np.log(2.)/s, self.dtype)
        self.qb = as_fxp(b/s, self.dtype)
        self.qc = as_fxp(c/a*s**2, self.dtype)
        super().__init__(cdlt, operands, outputs, program, scale=1)


    def iexp(self, x):

        z = quantize_np(x // self.qln2, self.dtype)
        z = z * -1
        y = quantize_np(z*self.qln2, self.dtype)
        x = x + y
        x = x + self.qb
        b = x.copy()
        x = quantize_np(x * b, self.dtype)
        x = x + self.qb

        # z = np.floor(z)
        z = fxp_floor(z)
        x = x >> z
        return x


    def fn_impl(self, inouts):
        x = inouts['inputs'][0].data

        axis = self.cdlt.required_params['axis'].value
        minval, _ = range_from_cfg(FXP_CONFIGS[self.dtype])
        y = np.max(x, axis=axis, keepdims=True)

        x = x - y
        x = self.iexp(x)
        dx = np.sum(x, axis=axis, keepdims=1)
        out = x//dx
        out = quantize_np(out, self.dtype)

        inouts['outputs'] = [out]
        return inouts

class DWConv(ReferenceOp):

    def __init__(self, cdlt, program, use_bias=True, use_quantization=True):
        self.dtype = "FXP32"
        self.use_bias = use_bias
        self.use_quantization = use_quantization
        if self.use_bias:
            operands = [cdlt.inputs[0], cdlt.inputs[1], cdlt.inputs[2]]
        else:
            operands = [cdlt.inputs[0], cdlt.inputs[1]]
        outputs = [cdlt.outputs[0]]
        self.stride = cdlt.required_params['stride'].value
        super().__init__(cdlt, operands, outputs, program, scale=2)

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
        if self.use_bias:
            bias = inouts['inputs'][2].data
        else:
            bias = None


        data = data.transpose(0, 3, 1, 2)
        wgt = wgt.transpose(*tuple(WEIGHTS_CL_TO_CF))
        output = self.dw_conv2d(data, wgt, self.stride, 0, bias=bias)
        output = output.transpose(0, 2, 3, 1)
        inouts['outputs'] = [output]
        return inouts

    def dw_conv2d(self, data, w, stride, pad, bias=None):

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
                                mul_res_quant = quantize_np(mul_res, self.dtype)
                                # mul_res_quant = mul_res
                                output[n, c, i, j] += mul_res_quant
                if bias:
                    output[n,c] += bias[c]
        return output


class GlobalAvgPool(ReferenceOp):

    def __init__(self, cdlt, program):
        operands = [cdlt.inputs[0],]
        outputs = [cdlt.outputs[0]]
        self.dtype = "FXP32"
        super().__init__(cdlt, operands, outputs, program, scale=2)


    def fn_impl(self, inouts):
        data = inouts['inputs'][0].data.copy()
        data = data.transpose(0, 3, 1, 2)
        out = np.sum(data, axis=(2, 3), keepdims=True)
        denom = Fxp(1.0 / (data.shape[2] * data.shape[3]), **FXP_CONFIGS[self.dtype]).val.item()
        out = out * denom
        out = quantize_np(out, self.dtype)
        out = out.transpose((0, 2, 3, 1))
        inouts['outputs'] = [out]
        return inouts


class Gelu(ReferenceOp):

    def __init__(self, cdlt, program):
        operands = [cdlt.inputs[0],]
        outputs = [cdlt.outputs[0]]
        self.dtype = "FXP32"
        super().__init__(cdlt, operands, outputs, program, scale=1)
        a = -0.2888
        b = 1.769
        c = 1.
        s = 0.6/np.sqrt(2)
        s_erf = a*s**2

        self.b_s = as_fxp(-b / s, self.dtype)
        self.qb = as_fxp(b / s, self.dtype)
        self.qc = as_fxp(c/a*s**2, self.dtype)
        self.q1 = as_fxp(1./s_erf, self.dtype)


    def fn_impl(self, inouts):
        x = inouts['inputs'][0].data

        s_x = np.sign(x)
        y = np.abs(x)
        y = np.minimum(y, self.b_s)
        y = y + self.qb
        y = quantize_np(y*y, self.dtype)
        y = y + self.qc
        y = quantize_np(y*s_x, self.dtype)
        y = y + self.q1
        output = quantize_np(x * y, self.dtype)


        inouts['outputs'] = [output]
        return inouts

class BiasAdd(ReferenceOp):

    def __init__(self, cdlt, program):
        operands = [cdlt.inputs[0], cdlt.inputs[1]]
        outputs = [cdlt.outputs[0]]
        super().__init__(cdlt, operands, outputs, program)

    def fn_impl(self, inouts):
        data = inouts['inputs'][0].data
        bias = inouts['inputs'][1].data

        output = data + bias

        inouts['outputs'] = [output]
        return inouts

class UnImplementedOp(ReferenceOp):

    def __init__(self, cdlt, program):
        operands = []
        outputs = []
        super().__init__(cdlt, operands, outputs, program)
        raise RuntimeError(f"Op {cdlt.op_name} is not yet implemented")

def load_dnn_impls(cfg):

    DNN_IMPLS = {
        "avg_pool": partial(Pool, "avg"),
        "softmax4d": Softmax,
        "softmax": Softmax,
        "bias_add": BiasAdd,
        "batch_norm": UnImplementedOp,
        "cross_entropy_loss": UnImplementedOp,
        "depthwise_conv": partial(DWConv, use_bias=False),
        "depthwise_conv_bias": partial(DWConv, use_bias=True),
        "global_avg_pool": GlobalAvgPool,
        "max_pool": partial(Pool, "max"),
        "mean_var": UnImplementedOp,
        "gelu": Gelu,
    }
    return DNN_IMPLS


