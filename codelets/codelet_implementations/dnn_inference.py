from codelets.common.datatype import COMMON_DTYPES
from codelets.codelet_template import CodeletTemplate

def elem_add():
    with CodeletTemplate("elem_add") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        op1 = cdlt.add_input("op1", [N, C, H, W], COMMON_DTYPES[2])
        op2 = cdlt.add_input("op2", [N, C, H, W], COMMON_DTYPES[2])
        out = cdlt.add_output("out", [N, C, H, W], COMMON_DTYPES[2])

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        compute_out = cdlt.compute("ADD", [op1[n, c, h, w], op2[n, c, h, w]])
        _ = cdlt.transfer(compute_out, out)

    return cdlt


DNN_MAPPINGS = {
    'elem_add': elem_add()
}