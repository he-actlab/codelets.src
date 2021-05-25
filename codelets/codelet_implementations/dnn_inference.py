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
                        # conv.transfer


    return conv


DNN_MAPPINGS = {
    'elem_add': elem_add_template(),
    'gemm': gemm_template(),
    'conv': conv_template()
}