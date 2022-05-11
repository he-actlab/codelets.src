from codelets.adl.graph import ArchitectureNode
from codelets.templates.codelet_template import CodeletTemplate
from codelets.templates.operation_template import OperationTemplate
from examples.genesys import OP_DTYPES
from functools import partial

from . import add_simd_constraint

def elem_binary_op_(cdlt_name: str, instr_name: str, hag: ArchitectureNode):

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

def elem_binary_op(cdlt_name: str, instr_name: str, nop1_dims, nop2_dims, hag: ArchitectureNode):
    DIM_NAMES = ["N", "C", "H", "W"]
    with CodeletTemplate(cdlt_name) as cdlt:
        if nop1_dims > nop2_dims:
            num_dims = nop1_dims
            inpt_idx = 0
        else:
            num_dims = nop2_dims
            inpt_idx = 1
        dummy_dims = []
        op1_dims = []
        op2_dims = []
        assert num_dims <= len(DIM_NAMES)
        for i in range(num_dims):
            dim = cdlt.dummy_op(DIM_NAMES[i], cdlt.node.inputs[inpt_idx].shape[i])
            if i < nop1_dims:
                op1_dims.append(dim)
            if i < nop2_dims:
                op2_dims.append(dim)
            dummy_dims.append(dim)

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, op1_dims, default_dtype=OP_DTYPES[2])
        op2 = cdlt.create_operand_template("op2", OP_DTYPES, op2_dims, default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, dummy_dims, default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([op1, op2])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        loops = []
        op1_indices = []
        op2_indices = []
        for i, d in enumerate(dummy_dims):
            l = cdlt.loop(d)
            loops.append(l)
            if i < nop1_dims:
                op1_indices.append(l)
            if i < nop2_dims:
                op2_indices.append(l)
            OperationTemplate.loop_ctxt_level += 1
            OperationTemplate.loop_stack.append(l.loop_id)
            OperationTemplate.loop_ctx_dependencies.append(l.op_str)

        cdlt.transfer(op1, ["DRAM", "VMEM1"])
        cdlt.transfer(op2, ["DRAM", "VMEM2"])
        out.set_write_destination("VMEM1")
        if nop1_dims == 0:
            cdlt.compute(instr_name, [op1, op2[tuple(op2_indices)]], [out[tuple(loops)]], target="SIMD")
        elif nop2_dims == 0:
            cdlt.compute(instr_name, [op1[tuple(op1_indices)], op2], [out[tuple(loops)]], target="SIMD")
        else:
            cdlt.compute(instr_name, [op1[tuple(op1_indices)], op2[tuple(op2_indices)]], [out[tuple(loops)]], target="SIMD")
        cdlt.transfer(out, ["VMEM1", "DRAM"])

        for d in reversed(dummy_dims):
            OperationTemplate.loop_ctxt_level -= 1
            OperationTemplate.loop_stack.pop()
            OperationTemplate.loop_ctx_dependencies.pop()

        cdlt.configure("end", "SIMD")

        cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt

BINARY_CODELETS = {
    "elem_add": partial(elem_binary_op, "elem_add", "ADD", 4, 4),
    "elem_add1d": partial(elem_binary_op, "elem_add1d", "ADD", 3, 1),
    "elem_add3d": partial(elem_binary_op, "elem_add3d", "ADD", 3, 3),
    "elem_sub": partial(elem_binary_op_, "elem_sub", "SUB"),
    "elem_div": partial(elem_binary_op_, "elem_div", "DIV"),
    "elem_mul": partial(elem_binary_op_, "elem_mul", "MUL"),
    "elem_less": partial(elem_binary_op_, "elem_less", "LT"),
    "elem_equal": partial(elem_binary_op_, "elem_equal", "EQUAL"),
}