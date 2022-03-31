from codelets.adl.graph import ArchitectureNode
from codelets.templates.codelet_template import CodeletTemplate
from examples.genesys import OP_DTYPES, FXP_CONFIGS
from . import range_from_cfg, add_simd_constraint
from functools import partial


def elem_unary_op(cdlt_name: str,
                  instr_name: str,
                  imm_val,
                  hag: ArchitectureNode):

    with CodeletTemplate(cdlt_name) as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        if imm_val is not None:
            assert isinstance(imm_val, int)
            cdlt.configure("start", "IMM", immediate_value=imm_val, index=0)
            param = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(op1, ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM2")
                        if imm_val is not None:
                            cdlt.compute(instr_name, [op1[n, c, h, w], param], [out[n, c, h, w]], target="SIMD")
                        else:
                            cdlt.compute(instr_name, [op1[n, c, h, w]], [out[n, c, h, w]], target="SIMD")
                        cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    return cdlt


def coarse_flatten(hag: ArchitectureNode):

    with CodeletTemplate("coarse_flatten") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        OC = cdlt.dummy_op("OC", cdlt.node.outputs[0].shape[1])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, OC], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
    return cdlt


def coarse_flatten2d(hag: ArchitectureNode):

    with CodeletTemplate("coarse_flatten2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])


        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        cdlt.configure("end", "SIMD")
    return cdlt


def tensor_transpose(hag: ArchitectureNode):

    with CodeletTemplate("tensor_transpose") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        cdlt.configure("end", "SIMD")


        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        pass
    return cdlt


def reduce_sum(hag: ArchitectureNode):

    with CodeletTemplate("reduce_sum") as cdlt:


        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(data, ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM1")
                cdlt.compute("ADD", [data[n,c], data[n,c]], [out[c]], target="SIMD")
                cdlt.transfer(out, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt

def elem_cast(hag):

    with CodeletTemplate("elem_cast") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])

        out = cdlt.create_operand_template("out_cast", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        # fix C dim to array size
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:

                        cdlt.transfer(op1, ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM2")
                        cdlt.compute("RELU", [op1[n, c, h, w]], [out[n, c, h, w]], target="SIMD")
                        cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")


    return cdlt


def elem_cast2d(hag):

    with CodeletTemplate("elem_cast2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])

        out = cdlt.create_operand_template("out_relu", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(op1, ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM1")
                cdlt.compute("RELU", [op1[n, c]], [out[n, c]], target="SIMD")
                cdlt.transfer(out, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def elem_exp(hag):

    with CodeletTemplate("elem_exp") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])

        out = cdlt.create_operand_template("out_exp", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        # fix C dim to array size
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:

                        cdlt.transfer(op1, ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM2")
                        cdlt.compute("EXP", [op1[n, c, h, w]], [out[n, c, h, w]], target="SIMD")
                        cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")


    return cdlt


def inv_sqrt(hag):

    with CodeletTemplate("inv_sqrt") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])

        out = cdlt.create_operand_template("out_exp", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        # fix C dim to array size
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:

                        cdlt.transfer(op1, ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM2")
                        cdlt.compute("INV_SQRT", [op1[n, c, h, w]], [out[n, c, h, w]], target="SIMD")
                        cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")


    return cdlt


def leaky_relu(hag):

    with CodeletTemplate("leaky_relu") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])

        out = cdlt.create_operand_template("out_relu", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])


        cdlt.configure("start", "SIMD")
        alpha = cdlt.dummy_op("alpha", cdlt.node.alpha, dtype="FXP32")
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        cdlt.configure("start", "IMM", immediate_value=alpha, index=0)
        alpha = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        ## IMPORTANT: The configure index needs to correspond to the order in which the corresponding temporary is created
        # This is a temporary hotfix to enable IMM value indexing during instruction generation
        # fix C dim to array size
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:

                        cdlt.transfer(op1, ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM2")
                        cdlt.compute("LEAKY_RELU", [op1[n, c, h, w], alpha], [out[n, c, h, w]], target="SIMD")
                        cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")


    return cdlt


def relu2d(hag):

    with CodeletTemplate("relu2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])

        out = cdlt.create_operand_template("out_relu", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(op1, ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM1")
                cdlt.compute("RELU", [op1[n, c]], [out[n, c]], target="SIMD")
                cdlt.transfer(out, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def elem_tanh(hag: ArchitectureNode):

    with CodeletTemplate("elem_tanh") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out_tanh", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        cdlt.configure("start", "IMM", immediate_value=16, index=0)
        param = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(op1, ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM1")
                        cdlt.compute("TANH", [op1[n, c, h, w], param], [out[n, c, h, w]], target="SIMD")
                        cdlt.transfer(out, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt.add_compilation_param("LOOP_TILE_ORDER", ["N", "C", "H", "W"])

    return cdlt


def elem_tanh2d(hag: ArchitectureNode):

    with CodeletTemplate("elem_tanh2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])

        out = cdlt.create_operand_template("out_tanh", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=16, index=0)
        param = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        # fix C dim to array size
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(op1, ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM2")
                cdlt.compute("TANH", [op1[n, c], param], [out[n, c]], target="SIMD")
                cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")


    return cdlt


def elem_ceil2d(hag: ArchitectureNode):

    with CodeletTemplate("elem_ceil2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(op1, ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM2")
                cdlt.compute("CEIL", [op1[n, c]], [out[n, c]], target="SIMD")
                cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
        cdlt.add_compilation_param("LOOP_TILE_ORDER", ["N", "C"])
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def elem_pow2d(hag: ArchitectureNode):
    with CodeletTemplate("elem_pow2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        exp = cdlt.dummy_op("exp", cdlt.node.kwargs['exp'], dtype="FXP32")
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        cdlt.set_inputs([op1])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=0, index=0)

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(op1, ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM2")
                cdlt.compute("MOVE", [op1[n, c]], [out[n, c]], target="SIMD")
                cdlt.compute("POW", [op1[n, c], out[n, c]], [out[n, c]], target="SIMD")
                cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")


    return cdlt


def reduce_mean2d(hag: ArchitectureNode):
    #

    # # TODO: Add option to create operand
    # THIS ASSUMES THE AXIS IS THE OUTERMOST AXIS. IN THE FUTURE, NEED TO ADAPT TO DIFFERENT AXES
    with CodeletTemplate("reduce_mean2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        ONE = cdlt.dummy_op("ONE", cdlt.node.outputs[0].shape[0])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [ONE, C], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        # Change this to be the reciprocal as a FXP value

        denom = cdlt.dummy_op("denom", 1/(cdlt.node.inputs[0].shape[0]), dtype="FXP32")
        axis = cdlt.dummy_op("axis", cdlt.node.kwargs['axes'][0])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        cdlt.configure("start", "SIMD")
        ## IMPORTANT: The configure index needs to correspond to the order in which the corresponding temporary is created
        # This is a temporary hotfix to enable IMM value indexing during instruction generation
        cdlt.configure("start", "IMM", immediate_value=0, index=0)
        zero_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        cdlt.configure("start", "IMM", immediate_value=denom, index=1)
        denom_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        with cdlt.loop(ONE) as o:
            with cdlt.loop(C) as c:
                with cdlt.loop(N) as n:
                    cdlt.transfer(data, ["DRAM", "VMEM1"])
                    # TODO: Zero out output data at compile time
                    cdlt.transfer(out, ["DRAM", "VMEM2"])
                    out.set_write_destination("VMEM2")
                    cdlt.compute("ADD", [data[n,c], out[o, c]], [out[o, c]], target="SIMD")
                    cdlt.compute("MUL", [out[o, c], denom_op], [out[o, c]], target="SIMD")
                cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    cdlt.add_compilation_param("LEVEL1_hint", f"splits['N'] == 1")

    return cdlt


def reduce_min2d(hag: ArchitectureNode):
    #
    # # TODO: Add option to create operand
    # THIS ASSUMES THE AXIS IS THE OUTERMOST AXIS. IN THE FUTURE, NEED TO ADAPT TO DIFFERENT AXES
    with CodeletTemplate("reduce_min2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        ONE = cdlt.dummy_op("ONE", cdlt.node.outputs[0].shape[0])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [ONE, C], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        # Change this to be the reciprocal as a FXP value

        axis = cdlt.dummy_op("axis", cdlt.node.kwargs['axes'][0])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        cdlt.configure("start", "SIMD")
        ## IMPORTANT: The configure index needs to correspond to the order in which the corresponding temporary is created
        # This is a temporary hotfix to enable IMM value indexing during instruction generation
        _, max_val = range_from_cfg(FXP_CONFIGS[str(OP_DTYPES[2])])

        cdlt.configure("start", "IMM", immediate_value=max_val, index=0)

        with cdlt.loop(ONE) as o:
            with cdlt.loop(C) as c:
                with cdlt.loop(N) as n:
                    cdlt.transfer(data, ["DRAM", "VMEM1"])
                    # TODO: Zero out output data at compile time
                    cdlt.transfer(out, ["DRAM", "VMEM2"])
                    out.set_write_destination("VMEM2")
                    cdlt.compute("MIN", [data[n, c], out[o, c]], [out[o, c]], target="SIMD")
                cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt.add_compilation_param("LEVEL1_hint", f"splits['N'] == 1")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def tensor_transpose2d(hag: ArchitectureNode):
    #

    # # TODO: Add option to create operand
    # THIS ASSUMES THE AXIS IS THE OUTERMOST AXIS. IN THE FUTURE, NEED TO ADAPT TO DIFFERENT AXES
    with CodeletTemplate("tensor_transpose2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [C, N], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        # Change this to be the reciprocal as a FXP value

        # axis = cdlt.dummy_op("axis", cdlt.node.kwargs['axes'][0])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        cdlt.configure("start", "SIMD")

        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                cdlt.transfer(data, ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM2")
                cdlt.compute("TRANSPOSE", [data[n, c]], [out[c, n]], target="SIMD")
            cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")


    cdlt.add_compilation_param("LEVEL1_hint", f"splits['N'] == 1")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def tensor_transpose4d(hag: ArchitectureNode):
    #

    # # TODO: Add option to create operand
    # THIS ASSUMES THE AXIS IS THE OUTERMOST AXIS. IN THE FUTURE, NEED TO ADAPT TO DIFFERENT AXES
    with CodeletTemplate("tensor_transpose4d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, H, W], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        # Change this to be the reciprocal as a FXP value

        # axis = cdlt.dummy_op("axis", cdlt.node.kwargs['axes'][0])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        cdlt.configure("start", "SIMD")

        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                cdlt.transfer(data, ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM2")
                cdlt.compute("TRANSPOSE", [data[n, c]], [out[c, n]], target="SIMD")
            cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")


    cdlt.add_compilation_param("LEVEL1_hint", f"splits['N'] == 1")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt

def clip(hag: ArchitectureNode):

    with CodeletTemplate("elem_clip") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])
        cdlt.set_outputs([out])
        minval = cdlt.dummy_op("min", cdlt.node.kwargs['minval'], dtype="FXP32")
        maxval = cdlt.dummy_op("max", cdlt.node.kwargs['maxval'], dtype="FXP32")
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=minval, index=0)
        min_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        cdlt.configure("start", "IMM", immediate_value=maxval, index=1)
        max_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        # temp_res = cdlt.create_operand_template("partial", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        temp_res = cdlt.create_temp_operand([N, C, H, W], "VMEM1")

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:

                        cdlt.transfer(op1, ["DRAM", "VMEM1"])
                        temp_res.set_write_destination("VMEM1")
                        cdlt.compute("MAX", [op1[n, c, h, w], max_op], [temp_res[n, c, h, w]], target="SIMD")
                        out.set_write_destination("VMEM2")
                        cdlt.compute("MIN", [temp_res[n, c, h, w], min_op], [out[n, c, h, w]], target="SIMD")
                        cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")

    cdlt = add_simd_constraint(hag, cdlt, "C")
    return cdlt

UNARY_CODELETS = {
    "coarse_flatten": coarse_flatten,
    "coarse_flatten2d": coarse_flatten2d,
    "elem_tanh": partial(elem_unary_op, "elem_tanh", "TANH", 16),
    "elem_tanh2d": elem_tanh2d,
    # TODO: Check if this needs to be 'sigmoid'
    "elem_sigmoid": partial(elem_unary_op, "elem_sigmoid", "SIGMOID", 16),
    "leaky_relu": leaky_relu,
    "elem_clip": clip,
    "elem_ceil2d": elem_ceil2d,
    "elem_pow2d": elem_pow2d,
    "elem_exp": elem_exp,
    "relu": partial(elem_unary_op, "relu", "RELU", 16),
    "relu2d": relu2d,
    'tensor_transpose2d': tensor_transpose2d,
    "reduce_mean2d": reduce_mean2d,
    "reduce_min2d": reduce_min2d,
    "reduce_sum": reduce_sum,
    'elem_cast': elem_cast,
    'elem_cast2d': elem_cast2d,
    "inv_sqrt": inv_sqrt,
}