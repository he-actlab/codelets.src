from codelets.adl.graph import ArchitectureNode
from codelets.templates.codelet_template import CodeletTemplate
from examples.genesys import OP_DTYPES
from functools import partial

from . import add_simd_constraint

def elem_binary_op(cdlt_name: str, instr_name: str, hag: ArchitectureNode):

    with CodeletTemplate(cdlt_name) as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        op2 = cdlt.create_operand_template("op2", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([op1, op2])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(op1, ["DRAM", "VMEM1"])
                        cdlt.transfer(op2, ["DRAM", "VMEM2"])
                        out.set_write_destination("VMEM1")
                        cdlt.compute(instr_name, [op1[n, c, h, w], op2[n, c, h, w]], [out[n, c, h, w]], target="SIMD")
                        cdlt.transfer(out, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")

        cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt

BINARY_CODELETS = {
    "elem_add": partial(elem_binary_op, "elem_add", "ADD"),
    "elem_sub": partial(elem_binary_op, "elem_sub", "SUB"),
    "elem_div": partial(elem_binary_op, "elem_div", "DIV"),
    "elem_mul": partial(elem_binary_op, "elem_mul", "MUL"),
    "elem_less": partial(elem_binary_op, "elem_less", "LT"),
    "elem_equal": partial(elem_binary_op, "elem_equal", "EQUAL"),
}