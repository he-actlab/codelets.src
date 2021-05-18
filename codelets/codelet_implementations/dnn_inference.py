from codelets.common.datatype import COMMON_DTYPES
from codelets.codelet_template import CodeletTemplate

def elem_add():
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

def gemm():
    with CodeletTemplate("gemm") as gemm:

        P = gemm.dummy_op("P", gemm.node.inputs[2].shape[0])
        N = gemm.dummy_op("N", gemm.node.inputs[0].shape[1])
        M = gemm.dummy_op("M", gemm.node.inputs[0].shape[0])
        data = gemm.add_input("data", [M, N], COMMON_DTYPES[0])
        weight = gemm.add_input("weight", [N, P], COMMON_DTYPES[0])
        bias = gemm.add_input("bias", [P], COMMON_DTYPES[0])
        out = gemm.add_output("out", [M, P], COMMON_DTYPES[2])

        compute_out = gemm.compute("MVMUL", [data, weight, bias])
        _ = gemm.transfer(compute_out, out)

    with CodeletTemplate("gemm") as gemm:

        P = gemm.dummy_op("P", gemm.node.inputs[2].shape[0])
        N = gemm.dummy_op("N", gemm.node.inputs[0].shape[1])
        M = gemm.dummy_op("M", gemm.node.inputs[0].shape[0])
        data = gemm.add_input("data", [M, N], COMMON_DTYPES[0])
        weight = gemm.add_input("weight", [N, P], COMMON_DTYPES[0])
        bias = gemm.add_input("bias", [P], COMMON_DTYPES[0])
        out = gemm.add_output("out", [M, P], COMMON_DTYPES[2])
        with gemm.loop(N) as n:
            with gemm.loop(P) as p:
                with gemm.loop(M) as m:
                    compute_out = gemm.compute("MUL", [data, weight])
            compute_out1 = gemm.compute("ADD", [compute_out, out])
            _ = gemm.transfer(compute_out1, out)

    return gemm


DNN_MAPPINGS = {
    'elem_add': elem_add(),
    'gemm': gemm()
}