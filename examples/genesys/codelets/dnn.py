from codelets.adl.graph import ArchitectureNode
from codelets.templates.codelet_template import CodeletTemplate
from examples.genesys import OP_DTYPES, FXP_CONFIGS

from . import range_from_cfg

def depthwise_conv(hag: ArchitectureNode):
    # TODO: De-duplicate replicated outer loops for a given VMEM
    # TODO: Add zero constant
    # TODO: Replicate inner loops on a per-operand basis, and use the same offset from the previous tile
    # TODO: Make sure the output operands use 0 for it's offset
    # TODO: Need to figure out how to change the memory layout
    with CodeletTemplate("depthwise_conv") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        ONE = cdlt.dummy_op("ONE", cdlt.node.inputs[1].shape[1])
        KH = cdlt.dummy_op("KH", cdlt.node.inputs[1].shape[2])
        KW = cdlt.dummy_op("KW", cdlt.node.inputs[1].shape[3])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])
        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, IH, IW], default_dtype=OP_DTYPES[2])
        weight = cdlt.create_operand_template("weight", OP_DTYPES, [C, ONE, KH, KW], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, OH, OW], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data, weight])
        cdlt.set_outputs([out])

        stride = cdlt.dummy_op("stride", cdlt.node.stride)
        pad = cdlt.dummy_op("pad", cdlt.node.pad)
        # OS ->
        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=0, index=0)

        with cdlt.loop(ONE) as one:
            with cdlt.loop(N) as n:
                with cdlt.loop(C) as c:
                    with cdlt.loop(OH) as y:
                        with cdlt.loop(OW) as x:
                            with cdlt.loop(KH) as kh:
                                with cdlt.loop(KW) as kw:

                                    cdlt.transfer(weight[c, one, kh, kw], ["DRAM", "VMEM1"])
                                    cdlt.transfer(data[n, c, y*stride + kh, x*stride + kw], ["DRAM", "VMEM2"])
                                    cdlt.transfer(out[n, c, y, x], ["DRAM", "VMEM2"])
                                    out.set_write_destination("VMEM2")
                                    cdlt.compute("MACC", [data, weight, out], [out], target="SIMD")
                                    cdlt.transfer(out[n, c, y, x], ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")

    simd_dims = hag.get_subgraph_node("pe_array").dimensions
    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")

    return cdlt


def batch_norm(hag: ArchitectureNode):

    with CodeletTemplate("batch_norm") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        #
        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        scale = cdlt.create_operand_template("scale", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])
        offset = cdlt.create_operand_template("offset", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])
        mean = cdlt.create_operand_template("mean", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])
        istd = cdlt.create_operand_template("istd", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])

        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data, scale, offset, mean, istd])
        cdlt.set_outputs([out])

        numer = cdlt.create_operand_template("numer", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        denom = cdlt.create_operand_template("denom", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.add_temp_operand(numer)
        cdlt.add_temp_operand(denom)

        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(data[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(out[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(mean[c], ["DRAM", "VMEM2"])
                        cdlt.transfer(istd[c], ["DRAM", "VMEM2"])
                        cdlt.transfer(scale[c], ["DRAM", "VMEM2"])
                        cdlt.transfer(offset[c], ["DRAM", "VMEM2"])
                        numer.set_write_destination("VMEM1")
                        denom.set_write_destination("VMEM2")
                        out.set_write_destination("VMEM1")
                        cdlt.compute("SUB", [data, mean], [numer], target="SIMD")
                        cdlt.compute("MUL", [numer, istd], [out], target="SIMD")
                        cdlt.compute("MUL", [out, scale], [out], target="SIMD")
                        cdlt.compute("ADD", [out, offset], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")

    return cdlt


def mean_var(hag: ArchitectureNode):
    with CodeletTemplate("mean_var") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        mean = cdlt.create_operand_template("mean", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])
        istd = cdlt.create_operand_template("istd", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])
        temp1 = cdlt.create_operand_template("temp1", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])
        temp2 = cdlt.create_operand_template("temp2", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])
        temp3 = cdlt.create_operand_template("temp3", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])
        cdlt.add_temp_operand(temp1)
        cdlt.add_temp_operand(temp2)
        cdlt.add_temp_operand(temp3)
        temp1.start_location = "VMEM1"
        temp2.start_location = "VMEM2"
        temp3.start_location = "VMEM2"
        temp1.set_write_destination("VMEM1")
        temp2.set_write_destination("VMEM1")
        temp3.set_write_destination("VMEM1")

        cdlt.set_inputs([data])
        cdlt.set_outputs([mean, istd])
        denom = cdlt.dummy_op("denom", cdlt.node.inputs[0].shape[0]*cdlt.node.inputs[0].shape[2]*cdlt.node.inputs[0].shape[3])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        denom_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        eps_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        cdlt.configure("start", "IMM", immediate_value=denom, index=0)
        cdlt.configure("start", "IMM", immediate_value=0.0001, index=1)


        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(data[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(mean[c], ["DRAM", "VMEM1"])
                        cdlt.transfer(istd[c], ["DRAM", "VMEM2"])
                        data.set_write_destination("VMEM1")
                        mean.set_write_destination("VMEM1")
                        istd.set_write_destination("VMEM2")
                        cdlt.compute("ADD", [data, mean], [mean], target="SIMD")
                        cdlt.compute("MUL", [data, data], [temp1], target="SIMD")
                        cdlt.compute("ADD", [istd, temp1], [istd], target="SIMD")
            cdlt.compute("MUL", [mean, mean], [temp2], target="SIMD")
            cdlt.compute("DIV", [temp2, denom_op], [temp3], target="SIMD")
            cdlt.compute("SUB", [istd, temp3], [istd], target="SIMD")
            cdlt.compute("DIV", [istd, denom_op], [istd], target="SIMD")
            cdlt.compute("ADD", [istd, eps_op], [istd], target="SIMD")
            cdlt.compute("INV_SQRT", [istd], [istd], target="SIMD")
            cdlt.compute("DIV", [mean, denom_op], [mean], target="SIMD")
            cdlt.transfer(mean[c], ["VMEM1", "DRAM"])
            cdlt.transfer(istd[c], ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt


def cross_entropy_loss(hag: ArchitectureNode):

    with CodeletTemplate("cross_entropy_loss") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        res = cdlt.create_operand_template("res", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        target = cdlt.create_operand_template("target", OP_DTYPES, [N], default_dtype=OP_DTYPES[2])
        loss = cdlt.create_operand_template("loss", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([res, target])
        cdlt.set_outputs([loss])
        temp1 = cdlt.create_operand_template("temp1", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        temp1.start_location = "VMEM1"
        temp1.set_write_destination("VMEM1")
        cdlt.add_temp_operand(temp1)
        cdlt.configure("start", "SIMD")

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(res[n, c], ["DRAM", "VMEM1"])
                cdlt.transfer(loss[n, c], ["DRAM", "VMEM1"])
                loss.set_write_destination("VMEM1")
                cdlt.compute("EXP", [res], [temp1], target="SIMD")
                cdlt.compute("ADD", [temp1, loss], [loss], target="SIMD")
            cdlt.compute("DIV", [temp1, loss], [loss], target="SIMD")
            cdlt.transfer(loss[n, c], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt


def maxpool2d(hag: ArchitectureNode):
    #

    # # TODO: Add option to create operand
    with CodeletTemplate("max_pool") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        KH = cdlt.dummy_op("KH", cdlt.node.kernel_size[0])
        KW = cdlt.dummy_op("KW", cdlt.node.kernel_size[1])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])
        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, IH, IW], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, OH, OW], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        min_val, _ = range_from_cfg(FXP_CONFIGS[str(OP_DTYPES[2])])

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=min_val, index=0)
        pad = cdlt.dummy_op("pad", cdlt.node.pad[0])
        sy = cdlt.dummy_op("sy", cdlt.node.stride[0])
        sx = cdlt.dummy_op("sx", cdlt.node.stride[1])
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(KH) as kh:
                    with cdlt.loop(KW) as kw:
                        with cdlt.loop(OH) as y:
                            with cdlt.loop(OW) as x:
                                cdlt.transfer(data[n, c, y*sy + kh, x*sx + kw], ["DRAM", "VMEM1"])
                                # TODO: Initialize output as negative infinity at compile time
                                cdlt.transfer(out[n, c, y, x], ["DRAM", "VMEM2"])
                                out.set_write_destination("VMEM2")
                                cdlt.compute("MAX", [data, out], [out], target="SIMD")
                                cdlt.transfer(out[n, c, y, x], ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions
    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt


def averagepool2d(hag: ArchitectureNode):

    with CodeletTemplate("avg_pool") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        KH = cdlt.dummy_op("KH", cdlt.node.kernel_size[0])
        KW = cdlt.dummy_op("KW", cdlt.node.kernel_size[1])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])
        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])
        denom = cdlt.dummy_op("denom", cdlt.node.inputs[0].shape[2]*cdlt.node.inputs[0].shape[3])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, IH, IW], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, OH, OW], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        denom_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        cdlt.configure("start", "SIMD")
        # denom = IH*IW
        cdlt.configure("start", "IMM", immediate_value=denom, index=0)
        cdlt.configure("start", "IMM", immediate_value=0, index=1)
        sy = cdlt.dummy_op("sy", cdlt.node.stride[0])
        sx = cdlt.dummy_op("sx", cdlt.node.stride[1])
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(KH) as kh:
                    with cdlt.loop(KW) as kw:
                        with cdlt.loop(OH) as y:
                            with cdlt.loop(OW) as x:
                                cdlt.transfer(data[n, c, y*sy + kh, x*sx + kw], ["DRAM", "VMEM1"])
                                # TODO: Initialize output as negative infinity at compile time
                                cdlt.transfer(out[n, c, y, x], ["DRAM", "VMEM2"])
                                out.set_write_destination("VMEM2")
                                cdlt.compute("ADD", [data, out], [out], target="SIMD")
                cdlt.compute("MUL", [out, denom_op], [out], target="SIMD")
                cdlt.transfer(out[n, c, y, x], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    # cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt


def global_avg_pool(hag: ArchitectureNode):
    #

    # # TODO: Add option to create operand
    with CodeletTemplate("global_avg_pool") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, IH, IW], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, OH, OW], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        # Change this to be the reciprocal as a FXP value

        denom = cdlt.dummy_op("denom", 1/(cdlt.node.inputs[0].shape[2]*cdlt.node.inputs[0].shape[3]), dtype="FXP32")
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        cdlt.configure("start", "SIMD")
        ## IMPORTANT: The configure index needs to correspond to the order in which the corresponding temporary is created
        # This is a temporary hotfix to enable IMM value indexing during instruction generation
        cdlt.configure("start", "IMM", immediate_value=0, index=0)
        zero_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        cdlt.configure("start", "IMM", immediate_value=denom, index=1)
        denom_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        with cdlt.loop(OH) as oy:
            with cdlt.loop(OW) as ox:
                with cdlt.loop(IH) as iy:
                    with cdlt.loop(IW) as ix:
                        with cdlt.loop(N) as n:
                            with cdlt.loop(C) as c:
                                cdlt.transfer(data[n, c, iy, ix], ["DRAM", "VMEM1"])
                                # TODO: Zero out output data at compile time
                                cdlt.transfer(out[n, c, oy, ox], ["DRAM", "VMEM2"])
                                out.set_write_destination("VMEM2")
                                cdlt.compute("ADD", [data, out], [out], target="SIMD")
                                cdlt.compute("MUL", [out, denom_op], [out], target="SIMD")
                        cdlt.transfer(out[n, c, oy, ox], ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")

    simd_dims = hag.get_subgraph_node("pe_array").dimensions
    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt


DNN_CDLTS = {
"avg_pool": averagepool2d,
    "batch_norm": batch_norm,
    "cross_entropy_loss": cross_entropy_loss,
    "depthwise_conv": depthwise_conv,
    "global_avg_pool": global_avg_pool,
    "max_pool": maxpool2d,
    "mean_var": mean_var,
}