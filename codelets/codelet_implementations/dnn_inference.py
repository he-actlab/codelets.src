from codelets.common.datatype import COMMON_DTYPES
from codelets.codelet_template import CodeletTemplate

def elem_add_template():
    with CodeletTemplate("elem_add") as elem_add:
        N = elem_add.dummy_op("N", elem_add.node.inputs[0].shape[0])
        C = elem_add.dummy_op("C", elem_add.node.inputs[0].shape[1])
        H = elem_add.dummy_op("H", elem_add.node.inputs[0].shape[2])
        W = elem_add.dummy_op("W", elem_add.node.inputs[0].shape[3])
        op1 = elem_add.add_input("op1", [N, C, H, W], COMMON_DTYPES[2])
        op2 = elem_add.add_input("op2", [N, C, H, W], COMMON_DTYPES[2])
        out = elem_add.add_output("out", [N, C, H, W], COMMON_DTYPES[2])

        with elem_add.loop(N) as n:
            with elem_add.loop(C) as c:
                with elem_add.loop(H) as h:
                    with elem_add.loop(W) as w:
                        compute_out = elem_add.compute("ADD", [op1[n, c, h, w], op2[n, c, h, w]])
        _ = elem_add.transfer(compute_out, out)

    return elem_add

def relu_template():
    with CodeletTemplate("relu") as relu:
        N = relu.dummy_op("N", relu.node.inputs[0].shape[0])
        C = relu.dummy_op("C", relu.node.inputs[0].shape[1])
        H = relu.dummy_op("H", relu.node.inputs[0].shape[2])
        W = relu.dummy_op("W", relu.node.inputs[0].shape[3])
        op1 = relu.add_input("op1", [N, C, H, W], COMMON_DTYPES[2])
        op2 = relu.add_input("op2", [N, C, H, W], COMMON_DTYPES[2])
        out = relu.add_output("out", [N, C, H, W], COMMON_DTYPES[2])

        with relu.loop(N) as n:
            with relu.loop(C) as c:
                with relu.loop(H) as h:
                    with relu.loop(W) as w:
                        compute_out = relu.compute("RELU", [op1[n, c, h, w], op2[n, c, h, w]])
        _ = relu.transfer(compute_out, out)

    return relu

def relu2d_template():
    with CodeletTemplate("relu2d") as relu:
        N = relu.dummy_op("N", relu.node.inputs[0].shape[0])
        C = relu.dummy_op("C", relu.node.inputs[0].shape[1])

        op1 = relu.add_input("op1", [N, C], COMMON_DTYPES[2])
        op2 = relu.add_input("op2", [N, C], COMMON_DTYPES[2])
        out = relu.add_output("out", [N, C], COMMON_DTYPES[2])

        with relu.loop(N) as n:
            with relu.loop(C) as c:
                compute_out = relu.compute("RELU", [op1[n, c], op2[n, c]])
        _ = relu.transfer(compute_out, out)

    return relu


def tanh_template():
    with CodeletTemplate("elem_tanh") as tanh:
        N = tanh.dummy_op("N", tanh.node.inputs[0].shape[0])
        C = tanh.dummy_op("C", tanh.node.inputs[0].shape[1])
        H = tanh.dummy_op("H", tanh.node.inputs[0].shape[2])
        W = tanh.dummy_op("W", tanh.node.inputs[0].shape[3])
        op1 = tanh.add_input("op1", [N, C, H, W], COMMON_DTYPES[2])
        op2 = tanh.add_input("op2", [N, C, H, W], COMMON_DTYPES[2])
        out = tanh.add_output("out", [N, C, H, W], COMMON_DTYPES[2])

        with tanh.loop(N) as n:
            with tanh.loop(C) as c:
                with tanh.loop(H) as h:
                    with tanh.loop(W) as w:
                        compute_out = tanh.compute("TANH", [op1[n, c, h, w], op2[n, c, h, w]])
        _ = tanh.transfer(compute_out, out)

    return tanh

def tanh2d_template():
    with CodeletTemplate("elem_tanh2d") as tanh2d:
        N = tanh2d.dummy_op("N", tanh2d.node.inputs[0].shape[0])
        C = tanh2d.dummy_op("C", tanh2d.node.inputs[0].shape[1])

        op1 = tanh2d.add_input("op1", [N, C], COMMON_DTYPES[2])
        op2 = tanh2d.add_input("op2", [N, C], COMMON_DTYPES[2])
        out = tanh2d.add_output("out", [N, C], COMMON_DTYPES[2])

        with tanh2d.loop(N) as n:
            with tanh2d.loop(C) as c:
                compute_out = tanh2d.compute("TANH", [op1[n, c], op2[n, c]])
        _ = tanh2d.transfer(compute_out, out)

    return tanh2d

def gemm_template():
    with CodeletTemplate("gemm") as gemm:

        P = gemm.dummy_op("P", gemm.node.inputs[2].shape[0])
        N = gemm.dummy_op("N", gemm.node.inputs[0].shape[1])
        M = gemm.dummy_op("M", gemm.node.inputs[0].shape[0])
        data = gemm.add_input("data", [M, N], COMMON_DTYPES[0])
        weight = gemm.add_input("weight", [N, P], COMMON_DTYPES[0])
        bias = gemm.add_input("bias", [P], COMMON_DTYPES[0])
        out = gemm.add_output("out", [M, P], COMMON_DTYPES[2])
        with gemm.loop(P) as p:
            with gemm.loop(M) as m:
                with gemm.loop(N) as n:
                    mvmul_out = gemm.compute("MACC", [data[m, n], weight[n, p], out[m, p]])
            compute_out = gemm.compute("ADD", [mvmul_out, bias[p]])
        _ = gemm.transfer(compute_out, out)

    return gemm

def conv_template():
    with CodeletTemplate("conv") as conv:
        OC = conv.dummy_op("OC", conv.node.outputs[0].shape[1])
        N = conv.dummy_op("N", conv.node.inputs[0].shape[0])
        IC = conv.dummy_op("IC", conv.node.inputs[0].shape[1])
        KH = conv.dummy_op("KH", conv.node.inputs[1].shape[2])
        KW = conv.dummy_op("KW", conv.node.inputs[1].shape[3])
        OH = conv.dummy_op("OH", conv.node.outputs[0].shape[2])
        OW = conv.dummy_op("OW", conv.node.outputs[0].shape[3])
        IH = conv.dummy_op("IH", conv.node.inputs[0].shape[2])
        IW = conv.dummy_op("IW", conv.node.inputs[0].shape[3])
        data = conv.add_input("data", [N, IC, IH, IW], COMMON_DTYPES[0])
        weight = conv.add_input("weight", [OC, IC, KH, KW], COMMON_DTYPES[0])
        bias = conv.add_input("weight", [OC], COMMON_DTYPES[0])
        out = conv.add_output("out", [N, OC, OH, OW], COMMON_DTYPES[2])
        stride = conv.dummy_op("stride", conv.node.stride)
        # OS ->
        with conv.loop(OC) as oc:
            with conv.loop(N) as n:
                with conv.loop(IC) as ic:
                    with conv.loop(KH) as kh:
                        with conv.loop(KW) as kw:
                            with conv.loop(OH) as y:
                                with conv.loop(OW) as x:
                                    macc_res = conv.compute("MACC", [data[n, ic, y*stride + kh, x*stride + kw],
                                                                        weight[oc, ic, kh, kw],
                                                                        out[n, oc, y, x]
                                                                        ])
                                    compute_out = conv.compute("ADD", [macc_res, bias[oc]])
                                    conv.transfer(compute_out, out[n, oc, y, x])
    return conv


def maxpool_template():
    with CodeletTemplate("max_pool") as mpool:
        C = mpool.dummy_op("C", mpool.node.inputs[0].shape[1])
        N = mpool.dummy_op("N", mpool.node.inputs[0].shape[0])
        KH = mpool.dummy_op("KH", mpool.node.kernel_size[0])
        KW = mpool.dummy_op("KW", mpool.node.kernel_size[0])
        OH = mpool.dummy_op("OH", mpool.node.outputs[0].shape[2])
        OW = mpool.dummy_op("OW", mpool.node.outputs[0].shape[3])
        IH = mpool.dummy_op("IH", mpool.node.inputs[0].shape[2])
        IW = mpool.dummy_op("IW", mpool.node.inputs[0].shape[3])
        data = mpool.add_input("data", [N, C, IH, IW], COMMON_DTYPES[0])
        out = mpool.add_output("out", [N, C, OH, OW], COMMON_DTYPES[2])
        sy = mpool.dummy_op("sy", mpool.node.stride[0])
        sx = mpool.dummy_op("sx", mpool.node.stride[1])
        ninf_const = mpool.constant(-10000)
        # OS ->
        with mpool.loop(N) as n:
            with mpool.loop(C) as c:
                max_val = mpool.transfer(ninf_const, None)
                with mpool.loop(KH) as kh:
                    with mpool.loop(KW) as kw:
                        with mpool.loop(OH) as y:
                            with mpool.loop(OW) as x:
                                max_partial = mpool.compute("MAX", [data[n, c, y*sy + kh, x*sx + kw],
                                                                    max_val])
                                _ = mpool.transfer(max_partial, max_val)
                                mpool.transfer(max_val, out[n, c, y, x])
    return mpool

def avg_pool_template():
    with CodeletTemplate("avg_pool") as mpool:
        C = mpool.dummy_op("C", mpool.node.inputs[0].shape[1])
        N = mpool.dummy_op("N", mpool.node.inputs[0].shape[0])
        KH = mpool.dummy_op("KH", mpool.node.kernel_size[0])
        KW = mpool.dummy_op("KW", mpool.node.kernel_size[0])
        OH = mpool.dummy_op("OH", mpool.node.outputs[0].shape[2])
        OW = mpool.dummy_op("OW", mpool.node.outputs[0].shape[3])
        IH = mpool.dummy_op("IH", mpool.node.inputs[0].shape[2])
        IW = mpool.dummy_op("IW", mpool.node.inputs[0].shape[3])
        data = mpool.add_input("data", [N, C, IH, IW], COMMON_DTYPES[0])
        out = mpool.add_output("out", [N, C, OH, OW], COMMON_DTYPES[2])
        sy = mpool.dummy_op("sy", mpool.node.stride[0])
        sx = mpool.dummy_op("sx", mpool.node.stride[1])
        ninf_const = mpool.constant(-10000)
        # OS ->
        with mpool.loop(N) as n:
            with mpool.loop(C) as c:
                max_val = mpool.transfer(ninf_const, None)
                with mpool.loop(KH) as kh:
                    with mpool.loop(KW) as kw:
                        with mpool.loop(OH) as y:
                            with mpool.loop(OW) as x:
                                max_partial = mpool.compute("MAX", [data[n, c, y*sy + kh, x*sx + kw],
                                                                    max_val])
                                _ = mpool.transfer(max_partial, max_val)
                                mpool.transfer(max_val, out[n, c, y, x])
    return mpool

def batch_norm_template():
    with CodeletTemplate("batch_norm") as bn:
        N = bn.dummy_op("N", bn.node.inputs[0].shape[0])
        C = bn.dummy_op("C", bn.node.inputs[0].shape[1])
        H = bn.dummy_op("H", bn.node.inputs[0].shape[2])
        W = bn.dummy_op("W", bn.node.inputs[0].shape[3])
        data = bn.add_input("data", [N, C, H, W], COMMON_DTYPES[2])
        scale = bn.add_input("scale", [C], COMMON_DTYPES[2])
        offset = bn.add_input("offset", [C], COMMON_DTYPES[2])
        mean = bn.add_input("mean", [C], COMMON_DTYPES[2])
        istd = bn.add_input("istd", [C], COMMON_DTYPES[2])
        out = bn.add_output("out", [N, C, H, W], COMMON_DTYPES[2])
        with bn.loop(C) as c:
            with bn.loop(N) as n:
                with bn.loop(H) as h:
                    with bn.loop(W) as w:
                        xhat = bn.compute("SUB", [data[n, c, h, w], mean[c]])
                        num_mul = bn.compute("MUL", [xhat, istd[c]])
                        scaled = bn.compute("MUL", [num_mul, scale[c]])
                        res = bn.compute("ADD", [scaled, offset[c]])
                        _ = bn.transfer(res, out[n, c, h, w])

    return bn

def tensor_transpose_template():
    with CodeletTemplate("tensor_transpose") as tt:
        N = tt.dummy_op("N", tt.node.inputs[0].shape[0])
        C = tt.dummy_op("C", tt.node.inputs[0].shape[1])
        H = tt.dummy_op("H", tt.node.inputs[0].shape[2])
        W = tt.dummy_op("W", tt.node.inputs[0].shape[3])
        data = tt.add_input("data", [N, C, H, W], COMMON_DTYPES[2])
        output = tt.add_output("out", [N, C, H, W], COMMON_DTYPES[2])

def tensor_reshape_template():
    with CodeletTemplate("tensor_reshape") as trs:
        N = trs.dummy_op("N", trs.node.inputs[0].shape[0])
        C = trs.dummy_op("C", trs.node.inputs[0].shape[1])
        H = trs.dummy_op("H", trs.node.inputs[0].shape[2])
        W = trs.dummy_op("W", trs.node.inputs[0].shape[3])
        data = trs.add_input("data", [N, C, H, W], COMMON_DTYPES[2])
        output = trs.add_output("out", [N, C, H, W], COMMON_DTYPES[2])

def tensor_pad_template():
    with CodeletTemplate("tensor_pad") as tp:
        N = tp.dummy_op("N", tp.node.inputs[0].shape[0])
        C = tp.dummy_op("C", tp.node.inputs[0].shape[1])
        H = tp.dummy_op("H", tp.node.inputs[0].shape[2])
        W = tp.dummy_op("W", tp.node.inputs[0].shape[3])
        data = tp.add_input("data", [N, C, H, W], COMMON_DTYPES[2])
        output = tp.add_output("out", [N, C, H, W], COMMON_DTYPES[2])

def tensor_flip_template():
    with CodeletTemplate("tensor_flip") as tf:
        N = tf.dummy_op("N", tf.node.inputs[0].shape[0])
        C = tf.dummy_op("C", tf.node.inputs[0].shape[1])
        H = tf.dummy_op("H", tf.node.inputs[0].shape[2])
        W = tf.dummy_op("W", tf.node.inputs[0].shape[3])
        data = tf.add_input("data", [N, C, H, W], COMMON_DTYPES[2])
        output = tf.add_output("out", [N, C, H, W], COMMON_DTYPES[2])

def mean_var_template():
    with CodeletTemplate("mean_var") as mv:
        N = mv.dummy_op("N", mv.node.inputs[0].shape[0])
        C = mv.dummy_op("C", mv.node.inputs[0].shape[1])
        H = mv.dummy_op("H", mv.node.inputs[0].shape[2])
        W = mv.dummy_op("W", mv.node.inputs[0].shape[3])
        istd = mv.add_output("istd", [C], COMMON_DTYPES[2])
        mean = mv.add_output("mean", [C], COMMON_DTYPES[2])

        data = mv.add_input("data", [N, C, H, W], COMMON_DTYPES[2])
        zero_constant = mv.constant(0)
        denom = mv.constant(mv.node.inputs[0].shape[0]*mv.node.inputs[0].shape[2]*mv.node.inputs[0].shape[3])
        with mv.loop(C) as c:
            accum = mv.transfer(zero_constant, None)
            istd_accum = mv.transfer(zero_constant, None)
            with mv.loop(N) as n:
                with mv.loop(H) as h:
                    with mv.loop(W) as w:
                        add_res = mv.compute("ADD", [data[n, c, h, w], accum])
                        _ = mv.transfer(add_res, accum)
                        mul_res = mv.compute("MUL", [data[n, c, h, w], data[n, c, h, w]])
                        istd_res = mv.compute("ADD", [mul_res, istd_accum])
                        _ = mv.transfer(istd_res, istd_accum)
            mean_sqr = mv.compute("MUL", [accum, accum])
            mean_sqr_div = mv.compute("DIV", [mean_sqr, denom])
            xhat = mv.compute("SUB", [istd_accum, mean_sqr_div])
            istd_temp0 = mv.compute("DIV", [xhat, denom])
            istd_res = mv.compute("INV_SQRT", [istd_temp0])
            mean_res = mv.compute("DIV", [accum, denom])
            _ = mv.transfer(istd_res, istd[c])
            _ = mv.transfer(mean_res, mean[c])

    return mv


def reduce_sum_template():
    with CodeletTemplate("reduce_sum") as rsum:
        N = rsum.dummy_op("N", rsum.node.inputs[0].shape[0])
        C = rsum.dummy_op("C", rsum.node.inputs[0].shape[1])

        data = rsum.add_input("data", [N, C], COMMON_DTYPES[2])
        out = rsum.add_output("out", [C], COMMON_DTYPES[2])
        zero_constant = rsum.constant(0)
        with rsum.loop(C) as c:
            accum = rsum.transfer(zero_constant, None)
            with rsum.loop(N) as n:
                compute_out = rsum.compute("ADD", [data[n, c], accum])
                _ = rsum.transfer(compute_out, accum)
            _ = rsum.transfer(accum, out[c])

    return rsum

DNN_INFERENCE_MAPPINGS = {
    'elem_add': elem_add_template(),
    'gemm': gemm_template(),
    'conv': conv_template(),
    'batch_norm': batch_norm_template(),
    'mean_var': mean_var_template(),
    'tensor_transpose': tensor_transpose_template(),
    'reduce_sum': reduce_sum_template(),
    'relu': relu_template(),
    'relu2d': relu2d_template(),
    'elem_tanh2d': tanh2d_template(),
    'elem_tanh': tanh_template(),
    'tensor_reshape': tensor_reshape_template(),
    'tensor_flip': tensor_flip_template(),
    'tensor_pad': tensor_pad_template(),
}