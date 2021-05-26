from codelets.common.datatype import COMMON_DTYPES
from codelets.codelet_template import CodeletTemplate

def sgd_template():
    # TODO: Generalize ewise operations to enable reusability of codelets
    with CodeletTemplate("sgd") as sgd:
        N = sgd.dummy_op("N", sgd.node.inputs[0].shape[0])
        param = sgd.add_input("param", [N], COMMON_DTYPES[0])
        grad = sgd.add_input("grad", [N], COMMON_DTYPES[0])
        updated_param = sgd.add_input("grad", [N], COMMON_DTYPES[0])
        lr = sgd.constant(sgd.node.kwargs['lr'])
        momentum = sgd.constant(sgd.node.kwargs['momentum'])

        with sgd.loop(N) as n:
            momentum_param = sgd.compute("MUL", [param[n], momentum])
            lr_grad = sgd.compute("MUL", [grad[n], lr])
            res = sgd.compute("MUL", [lr_grad, momentum_param])
            sgd.transfer(res, updated_param[n])

    return sgd


def batchnorm_grad_template():

    with CodeletTemplate("batchnorm_grad") as bn_grad:
        N = bn_grad.dummy_op("N", bn_grad.node.inputs[0].shape[0])
        C = bn_grad.dummy_op("C", bn_grad.node.inputs[0].shape[1])
        H = bn_grad.dummy_op("H", bn_grad.node.inputs[0].shape[2])
        W = bn_grad.dummy_op("W", bn_grad.node.inputs[0].shape[3])

        data = bn_grad.add_input("data", [N, C, H, W], COMMON_DTYPES[2])
        scale = bn_grad.add_input("scale", [C], COMMON_DTYPES[2])
        offset = bn_grad.add_input("offset", [C], COMMON_DTYPES[2])
        mean = bn_grad.add_input("mean", [C], COMMON_DTYPES[2])
        istd = bn_grad.add_input("istd", [C], COMMON_DTYPES[2])
        grad = bn_grad.add_input("grad", [N, C, H, W], COMMON_DTYPES[2])

        data_grad = bn_grad.add_output("data_grad", [N, C, H, W], COMMON_DTYPES[2])
        scale_grad = bn_grad.add_output("scale_grad", [C], COMMON_DTYPES[2])
        offset_grad = bn_grad.add_output("offset_grad", [C], COMMON_DTYPES[2])
        denom = bn_grad.constant(bn_grad.node.inputs[0].shape[0]*bn_grad.node.inputs[0].shape[2]*bn_grad.node.inputs[0].shape[3])
        zero_constant = bn_grad.constant(0)

        with bn_grad.loop(C) as c:
            scale_grad_accum = bn_grad.transfer(zero_constant, None)
            offset_grad_accum = bn_grad.transfer(zero_constant, None)
            with bn_grad.loop(N) as n:
                with bn_grad.loop(H) as h:
                    with bn_grad.loop(W) as w:
                        numer = bn_grad.compute("SUB", [data[n, c, h, w], mean[c]])
                        xhat = bn_grad.compute("MUL", [numer, istd[c]])
                        numer1 = bn_grad.compute("MUL", [xhat, grad[n, c, h, w]])
                        scale_grad_accum_update = bn_grad.compute("ADD", [scale_grad_accum, numer1])
                        _ = bn_grad.transfer(scale_grad_accum_update, scale_grad_accum)
                        offset_grad_accum_update = bn_grad.compute("ADD", [grad[n, c, h, w], offset_grad_accum])
                        _ = bn_grad.transfer(offset_grad_accum_update, offset_grad_accum)
            coeff0 = bn_grad.compute("MUL", [scale[c], istd[c]])
            coeff = bn_grad.compute("DIV", [coeff0, denom])

            with bn_grad.loop(N) as n1:
                with bn_grad.loop(H) as h1:
                    with bn_grad.loop(W) as w1:

                        temp2 = bn_grad.compute("MUL", [denom, grad[n1, c, h1, w1]])
                        temp3 = bn_grad.compute("MUL", [xhat[n1, c, h1, w1], scale_grad_accum])
                        temp4 = bn_grad.compute("SUB", [temp2, temp3])
                        temp5 = bn_grad.compute("SUB", [temp4, offset_grad_accum])
                        data_grad_res = bn_grad.compute("SUB", [coeff, temp5])
                        _ = bn_grad.transfer(data_grad_res, data_grad[n1, c, h1, w1])
            bn_grad.transfer(scale_grad_accum, scale_grad[c])
            bn_grad.transfer(offset_grad_accum, offset_grad[c])
    return bn_grad

def flatten_grad_template():

    with CodeletTemplate("flatten_grad") as fg:

        N = fg.dummy_op("N", fg.node.inputs[0].shape[0])
        C = fg.dummy_op("C", fg.node.inputs[0].shape[1])
        H = fg.dummy_op("H", fg.node.inputs[0].shape[2])
        W = fg.dummy_op("W", fg.node.inputs[0].shape[3])

        data = fg.add_input("data", [N, C], COMMON_DTYPES[2])
        grad = fg.add_input("grad", [N, C], COMMON_DTYPES[2])
        out = fg.add_output("out", [N, C, H, W], COMMON_DTYPES[2])


    return fg

DNN_TRAINING_MAPPINGS = {
    'sgd': sgd_template(),
    'batchnorm_grad': batchnorm_grad_template(),
    'flatten_grad': flatten_grad_template()
}