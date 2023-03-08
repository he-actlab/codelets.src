from codelets.adl.graph import ArchitectureNode
from codelets.templates.codelet_template import CodeletTemplate
from codelets.examples.genesys import FXP_CONFIGS, QUANT_SCALE, SIGN_SHIFT, DTYPE_MAP, OP_DTYPES
import numpy as np
from . import range_from_cfg, add_simd_constraint, create_immediate_with_operand, \
    add_simd_tile_constraint, add_scale_and_cast_op, CAST_FUNC



def depthwise_conv(hag: ArchitectureNode):
    # TODO: De-duplicate replicated outer loops for a given VMEM
    # TODO: Add zero constant
    # TODO: Replicate inner loops on a per-operand basis, and use the same offset from the previous tile
    # TODO: Make sure the output operands use 0 for it's offset
    # TODO: Need to figure out how to change the memory layout
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
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

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, IH, IW], default_dtype=acc_dtype)
        weight = cdlt.create_operand_template("weight", OP_DTYPES, [C, ONE, KH, KW], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, OH, OW], default_dtype=acc_dtype)
        cdlt.set_inputs([data, weight])
        cdlt.set_outputs([out])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        stride = cdlt.dummy_op("stride", cdlt.node.stride)
        pad = cdlt.dummy_op("pad", cdlt.node.pad_int)
        zero_op = cdlt.dummy_op('init', 0)
        zero = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="init")

        # OS ->
        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=zero_op)

        with cdlt.loop(ONE) as one:
            with cdlt.loop(N) as n:
                with cdlt.loop(C) as c:
                    with cdlt.loop(OH) as y:
                        with cdlt.loop(OW) as x:
                            with cdlt.loop(KH) as kh:
                                with cdlt.loop(KW) as kw:
                                    cdlt.transfer(weight, ["DRAM", "VMEM1"])
                                    cdlt.transfer(data, ["DRAM", "VMEM2"])
                                    cdlt.transfer(out, ["DRAM", "VMEM1"])
                                    out.set_write_destination("VMEM1")
                                    cdlt.compute("MACC", [data[n, c, y * stride + kh, x * stride + kw], weight[c, one, kh, kw], out[n, c, y, x]], [out[n, c, y, x]], target="SIMD")
                                    cdlt.transfer(out, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")

    cdlt = add_simd_constraint(hag, cdlt, "C")
    cdlt = add_simd_tile_constraint(hag, cdlt, ["KH", "KW"])

    return cdlt


def depthwise_conv_bias(hag: ArchitectureNode):
    # TODO: De-duplicate replicated outer loops for a given VMEM
    # TODO: Add zero constant
    # TODO: Replicate inner loops on a per-operand basis, and use the same offset from the previous tile
    # TODO: Make sure the output operands use 0 for it's offset
    # TODO: Need to figure out how to change the memory layout
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate("depthwise_conv_bias") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        ONE = cdlt.dummy_op("ONE", cdlt.node.inputs[1].shape[1])
        KH = cdlt.dummy_op("KH", cdlt.node.inputs[1].shape[2])
        KW = cdlt.dummy_op("KW", cdlt.node.inputs[1].shape[3])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])
        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, IH, IW], default_dtype=acc_dtype)
        weight = cdlt.create_operand_template("weight", OP_DTYPES, [C, ONE, KH, KW], default_dtype=acc_dtype)
        bias = cdlt.create_operand_template("bias", OP_DTYPES, [C], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, OH, OW], default_dtype=acc_dtype)
        cdlt.set_inputs([data, weight, bias])
        cdlt.set_outputs([out])

        stride = cdlt.dummy_op("stride", cdlt.node.stride)
        pad = cdlt.dummy_op("pad", cdlt.node.pad_int)
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        zero_op = cdlt.dummy_op('init', 0)
        zero = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="init")


        # OS ->
        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=zero_op)

        with cdlt.loop(ONE) as one:
            with cdlt.loop(N) as n:
                with cdlt.loop(C) as c:
                    with cdlt.loop(OH) as y:
                        with cdlt.loop(OW) as x:
                            with cdlt.loop(KH) as kh:
                                with cdlt.loop(KW) as kw:
                                    cdlt.transfer(bias, ["DRAM", "VMEM1"])
                                    cdlt.transfer(weight, ["DRAM", "VMEM1"])
                                    cdlt.transfer(data, ["DRAM", "VMEM2"])
                                    cdlt.transfer(out, ["DRAM", "VMEM2"])
                                    out.set_write_destination("VMEM2")
                                    cdlt.compute("MACC", [data[n, c, y * stride + kh, x * stride + kw], weight[c, one, kh, kw], out[n, c, y, x]], [out[n, c, y, x]], target="SIMD")
                                    cdlt.compute("ADD", [out[n, c, y, x], bias[c]], [out[n, c, y, x]], target="SIMD")
                                    cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")

    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt



def batch_norm(hag: ArchitectureNode):
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate("batch_norm") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        #
        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        scale = cdlt.create_operand_template("scale", OP_DTYPES, [C], default_dtype=acc_dtype)
        offset = cdlt.create_operand_template("offset", OP_DTYPES, [C], default_dtype=acc_dtype)
        mean = cdlt.create_operand_template("mean", OP_DTYPES, [C], default_dtype=acc_dtype)
        istd = cdlt.create_operand_template("istd", OP_DTYPES, [C], default_dtype=acc_dtype)

        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_inputs([data, scale, offset, mean, istd])
        cdlt.set_outputs([out])


        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(data, ["DRAM", "VMEM1"])
                        cdlt.transfer(scale, ["DRAM", "VMEM1"])
                        cdlt.transfer(offset, ["DRAM", "VMEM1"])
                        cdlt.transfer(mean, ["DRAM", "VMEM2"])
                        cdlt.transfer(istd, ["DRAM", "VMEM2"])

                        data.set_write_destination("VMEM1")
                        out.set_write_destination("VMEM2")
                        cdlt.compute("SUB", [data[n, c, h, w], mean[c]], [data[n, c, h, w]], target="SIMD")
                        cdlt.compute("MUL", [data[n, c, h, w], istd[c]], [out[n, c, h, w]], target="SIMD")
                        cdlt.compute("MUL", [out[n, c, h, w], scale[c]], [out[n, c, h, w]], target="SIMD")
                        cdlt.compute("ADD", [out[n, c, h, w], offset[c]], [out[n, c, h, w]], target="SIMD")
                        cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")

    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt

def bias_add(hag: ArchitectureNode):
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate("bias_add") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        bias = cdlt.create_operand_template("bias", OP_DTYPES, [C], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.set_inputs([data, bias])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")

        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(data, ["DRAM", "VMEM1"])
                        cdlt.transfer(bias, ["DRAM", "VMEM2"])
                        out.set_write_destination("VMEM1")
                        cdlt.compute("ADD", [data[n, c, h, w], bias[c]], [out[n, c, h, w]], target="SIMD")
                        cdlt.transfer(out, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")

    cdlt = add_simd_constraint(hag, cdlt, "C")
    cdlt.update_compilation_param("LEVEL1_hint", "splits['W'] == 1 and splits['H'] == 1")

    return cdlt

def mean_var(hag: ArchitectureNode):
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate("mean_var") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        mean = cdlt.create_operand_template("mean", OP_DTYPES, [C], default_dtype=acc_dtype)
        istd = cdlt.create_operand_template("istd", OP_DTYPES, [C], default_dtype=acc_dtype)
        temp1 = cdlt.create_operand_template("temp1", OP_DTYPES, [C], default_dtype=acc_dtype)
        temp2 = cdlt.create_operand_template("temp2", OP_DTYPES, [C], default_dtype=acc_dtype)
        temp3 = cdlt.create_operand_template("temp3", OP_DTYPES, [C], default_dtype=acc_dtype)
        cdlt.add_temp_operand(temp1)
        cdlt.add_temp_operand(temp2)
        cdlt.add_temp_operand(temp3)
        temp1.start_location = "VMEM1"
        temp2.start_location = "VMEM2"
        temp3.start_location = "VMEM2"
        temp1.set_write_destination("VMEM1")
        temp2.set_write_destination("VMEM2")
        temp3.set_write_destination("VMEM2")

        cdlt.set_inputs([data])
        cdlt.set_outputs([mean, istd])
        denom = cdlt.dummy_op("denom", cdlt.node.inputs[0].shape[0]*cdlt.node.inputs[0].shape[2]*cdlt.node.inputs[0].shape[3], dtype=acc_dtype_name)
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        denom_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="denom")
        eps = cdlt.dummy_op('eps', 0.0001, dtype=acc_dtype_name)
        eps_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="eps")
        cdlt.configure("start", "IMM", immediate_value=denom)
        cdlt.configure("start", "IMM", immediate_value=eps)


        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(data, ["DRAM", "VMEM1"])
                        cdlt.transfer(mean, ["DRAM", "VMEM1"])
                        cdlt.transfer(istd, ["DRAM", "VMEM2"])
                        mean.set_write_destination("VMEM1")
                        istd.set_write_destination("VMEM2")
                        cdlt.compute("ADD", [data[n, c, h, w], mean[c]], [mean[c]], target="SIMD")
                        cdlt.compute("MUL", [data[n, c, h, w], data[n, c, h, w]], [temp1[c]], target="SIMD")
                        cdlt.compute("ADD", [istd[c], temp1[c]], [istd[c]], target="SIMD")
            cdlt.compute("MUL", [mean[c], mean[c]], [temp2[c]], target="SIMD")
            cdlt.compute("DIV", [temp2[c], denom_op], [temp3[c]], target="SIMD")
            cdlt.compute("SUB", [istd[c], temp3[c]], [istd[c]], target="SIMD")
            cdlt.compute("DIV", [istd[c], denom_op], [istd[c]], target="SIMD")
            cdlt.compute("ADD", [istd[c], eps_op], [istd[c]], target="SIMD")
            cdlt.compute("INV_SQRT", [istd[c]], [istd[c]], target="SIMD")
            cdlt.compute("DIV", [mean[c], denom_op], [mean[c]], target="SIMD")
            cdlt.transfer(mean, ["VMEM1", "DRAM"])
            cdlt.transfer(istd, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt

def cross_entropy_loss(hag: ArchitectureNode):
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate("cross_entropy_loss") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        res = cdlt.create_operand_template("res", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        target = cdlt.create_operand_template("target", OP_DTYPES, [N], default_dtype=acc_dtype)
        loss = cdlt.create_operand_template("loss", OP_DTYPES, [N], default_dtype=acc_dtype)
        cdlt.set_inputs([res, target])
        cdlt.set_outputs([loss])

        cdlt.configure("start", "SIMD")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(res, ["DRAM", "VMEM1"])
                cdlt.transfer(target, ["DRAM", "VMEM2"])
                res.set_write_destination("VMEM1")
                loss.set_write_destination("VMEM2")
                cdlt.compute("EXP", [res[n, c]], [res[n, c]], target="SIMD")
                cdlt.compute("ADD", [res[n, c], target[n]], [loss[n]], target="SIMD")
                cdlt.compute("DIV", [res[n, c], loss[n]], [loss[n]], target="SIMD")
            cdlt.transfer(loss, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")

    cdlt = add_simd_constraint(hag, cdlt, "N")
    cdlt.add_compilation_param("LEVEL1_hint", f"splits['C'] == 1")
    return cdlt

def maxpool2d(hag: ArchitectureNode):
    #
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]

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

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, IH, IW], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, OH, OW], default_dtype=acc_dtype)

        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        min_val, _ = range_from_cfg(FXP_CONFIGS[acc_dtype_name])
        min_val_op = cdlt.dummy_op('init', min_val)
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        min_val_temp = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="init")

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=min_val_op)
        pad = cdlt.dummy_op("pad", cdlt.node.pad[0])
        sy = cdlt.dummy_op("sy", cdlt.node.stride[0])
        sx = cdlt.dummy_op("sx", cdlt.node.stride[1])
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(KH) as kh:
                    with cdlt.loop(KW) as kw:
                        with cdlt.loop(OH) as y:
                            with cdlt.loop(OW) as x:
                                cdlt.transfer(data, ["DRAM", "VMEM1"])
                                # TODO: Initialize output as negative infinity at compile time
                                cdlt.transfer(out, ["DRAM", "VMEM2"])
                                out.set_write_destination("VMEM2")
                                cdlt.compute("MAX", [data[n, c, y*sy + kh, x*sx + kw], out[n, c, y, x]], [out[n, c, y, x]], target="SIMD")
                                cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    return cdlt


def averagepool2d(hag: ArchitectureNode):
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate("avg_pool") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        KH = cdlt.dummy_op("KH", cdlt.node.kernel_size[0])
        KW = cdlt.dummy_op("KW", cdlt.node.kernel_size[1])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])
        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, IH, IW], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, OH, OW], default_dtype=acc_dtype)
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        denom = cdlt.dummy_op("denom", 1./(cdlt.node.kernel_size[0]*cdlt.node.kernel_size[1]), dtype=acc_dtype_name)
        denom_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='denom')
        zero_op = cdlt.dummy_op('init', 0)
        zero = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name="init")
        cdlt.configure("start", "SIMD")
        # denom = IH*IW
        cdlt.configure("start", "IMM", immediate_value=denom)
        cdlt.configure("start", "IMM", immediate_value=zero_op)
        sy = cdlt.dummy_op("sy", cdlt.node.stride[0])
        sx = cdlt.dummy_op("sx", cdlt.node.stride[1])
        with cdlt.loop(N) as n:
            with cdlt.loop(KH) as kh:
                with cdlt.loop(KW) as kw:
                    with cdlt.loop(OH) as y:
                        with cdlt.loop(OW) as x:
                            with cdlt.loop(C) as c:
                                cdlt.transfer(data, ["DRAM", "VMEM1"])
                                # TODO: Initialize output as negative infinity at compile time
                                cdlt.transfer(out, ["DRAM", "VMEM2"])
                                out.set_write_destination("VMEM2")
                                cdlt.compute("ADD", [data[n, c, y*sy + kh, x*sx + kw], out[n, c, y, x]], [out[n, c, y, x]], target="SIMD")
                                cdlt.compute("MUL", [out[n, c, y, x], denom_op], [out[n, c, y, x]], target="SIMD")
                                cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    cdlt = add_simd_tile_constraint(hag, cdlt, ["OH", "OW"])

    return cdlt


def global_avg_pool(hag: ArchitectureNode):
    #
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    # # TODO: Add option to create operand
    with CodeletTemplate("global_avg_pool") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, IH, IW], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, OH, OW], default_dtype=acc_dtype)

        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        # Change this to be the reciprocal as a FXP value

        denom = cdlt.dummy_op("denom", 1/(cdlt.node.inputs[0].shape[2]*cdlt.node.inputs[0].shape[3]), dtype=acc_dtype_name)
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        zero_op = cdlt.dummy_op('init', 0)
        zero = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='init')
        denom_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM", name='denom')

        cdlt.configure("start", "SIMD")
        ## IMPORTANT: The configure index needs to correspond to the order in which the corresponding temporary is created
        # This is a temporary hotfix to enable IMM value indexing during instruction generation
        cdlt.configure("start", "IMM", immediate_value=zero_op)

        cdlt.configure("start", "IMM", immediate_value=denom)

        with cdlt.loop(OH) as oy:
            with cdlt.loop(OW) as ox:
                with cdlt.loop(IH) as iy:
                    with cdlt.loop(IW) as ix:
                        with cdlt.loop(N) as n:
                            with cdlt.loop(C) as c:
                                cdlt.transfer(data, ["DRAM", "VMEM1"])
                                cdlt.transfer(out, ["DRAM", "VMEM2"])
                                out.set_write_destination("VMEM2")
                                cdlt.compute("ADD", [data[n, c, iy, ix], out[n, c, oy, ox]],
                                             [out[n, c, oy, ox]], target="SIMD")
                                cdlt.compute("MUL", [out[n, c, oy, ox], denom_op],
                                             [out[n, c, oy, ox]],
                                             target="SIMD")
                        cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    cdlt = add_simd_tile_constraint(hag, cdlt, ["OH", "OW", "IH", "IW"])

    return cdlt



def softmax(hag):
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate("softmax") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C], default_dtype=acc_dtype)

        simd_size = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])

        cdlt.configure("start", "SIMD")
        a, b, c = 0.3585, 1.353, 0.344
        s = 0.6
        qln2 = create_immediate_with_operand(cdlt, 'qln2', CAST_FUNC(np.log(2.)/s, acc_dtype_name), simd_size=simd_size)
        qb = create_immediate_with_operand(cdlt, 'qb', CAST_FUNC(b/s, acc_dtype_name), simd_size=simd_size)
        qc = create_immediate_with_operand(cdlt, 'qc', CAST_FUNC(c/a*s**2, acc_dtype_name), simd_size=simd_size)
        one = create_immediate_with_operand(cdlt, 'one', CAST_FUNC(1., acc_dtype_name), simd_size=simd_size)
        neg_one = create_immediate_with_operand(cdlt, 'neg_one', CAST_FUNC(-1., acc_dtype_name), simd_size=simd_size)
        min_val, _ = range_from_cfg(FXP_CONFIGS[acc_dtype_name])
        mval_op = cdlt.dummy_op('min_val', min_val)
        min_op = cdlt.create_temp_operand([simd_size], "IMM", name='min_val')

        cdlt.configure("start", "IMM", immediate_value=mval_op)

        # Max reduce output
        mx = cdlt.create_operand_template("mx", OP_DTYPES, [N], default_dtype=acc_dtype)
        cdlt.add_temp_operand(mx)

        # Expontential temporaries
        z = cdlt.create_operand_template("z", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        cdlt.add_temp_operand(z)

        b = cdlt.create_operand_template("b", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        cdlt.add_temp_operand(b)

        y = cdlt.create_operand_template("y", OP_DTYPES, [N, C], default_dtype=acc_dtype)
        cdlt.add_temp_operand(y)
        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                cdlt.transfer(data, ["DRAM", "VMEM1"])
                mx.start_location = "VMEM2"
                b.start_location = "VMEM2"

                out.set_write_destination("VMEM1")
                z.set_write_destination("VMEM1")
                mx.set_write_destination("VMEM2")
                y.set_write_destination("VMEM2")
                b.set_write_destination("VMEM2")


                # First, compute the max value by setting memory to the minimum possible value and subtracting
                cdlt.compute("MOVE", [min_op], [mx[n]], target="SIMD")
                cdlt.compute("MAX", [data[n, c], mx[n]], [mx[n]], target="SIMD")
                cdlt.compute("SUB", [data[n, c], mx[n]], [out[n, c]], target="SIMD")

                # Now, Comptue exp
                cdlt.compute("DIV", [out[n, c], qln2], [z[n, c]], target="SIMD")
                cdlt.compute("MUL", [z[n, c], neg_one], [z[n, c]], target="SIMD")
                cdlt.compute("MUL", [z[n, c], qln2], [y[n, c]], target="SIMD")
                cdlt.compute("ADD", [out[n, c], y[n, c]], [y[n, c]], target="SIMD")
                cdlt.compute("ADD", [out[n, c], qb], [out[n, c]], target="SIMD")
                cdlt.compute("MOVE", [out[n, c]], [b[n, c]], target="SIMD")
                cdlt.compute("MUL", [out[n, c], b[n, c]], [out[n, c]], target="SIMD")
                cdlt.compute("ADD", [out[n, c], qc], [out[n, c]], target="SIMD")
                cdlt.compute("FLOOR", [z[n, c]], [b[n, c]], target="SIMD")
                cdlt.compute("SHIFTR", [out[n, c], b[n, c]], [out[n, c]], target="SIMD")


                # Next, compute sum for denominator
                cdlt.compute("MOVE", [one], [mx[n]], target="SIMD")
                cdlt.compute("MACC", [out[n, c], one, mx[n]], [mx[n]], target="SIMD")

                # Finally, divide the exponent by the denom
                cdlt.compute("DIV", [out[n, c], mx[n]], [out[n, c]], target="SIMD")

                cdlt.transfer(out, ["VMEM1", "DRAM"])
        cdlt.configure('end', 'SIMD')

    cdlt = add_simd_constraint(hag, cdlt, "N")
    cdlt = add_simd_tile_constraint(hag, cdlt, ["C"])
    return cdlt

def softmax4d(hag):
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate("softmax4d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)

        simd_size = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])


        cdlt.configure("start", "SIMD")
        a, b, c = 0.3585, 1.353, 0.344
        s = 0.6
        # IMM 0
        qln2 = create_immediate_with_operand(cdlt, 'qln2', CAST_FUNC(np.log(2.)/s, acc_dtype_name), simd_size=simd_size)
        qb = create_immediate_with_operand(cdlt, 'qb', CAST_FUNC(b/s, acc_dtype_name), simd_size=simd_size)
        qc = create_immediate_with_operand(cdlt, 'qc', CAST_FUNC(c/a*s**2, acc_dtype_name), simd_size=simd_size)
        one = create_immediate_with_operand(cdlt, 'one', CAST_FUNC(1., acc_dtype_name), simd_size=simd_size)
        neg_one = create_immediate_with_operand(cdlt, 'neg_one', CAST_FUNC(-1., acc_dtype_name), simd_size=simd_size)

        axis = cdlt.dummy_op("axis", cdlt.node.kwargs['axis'][0])

        # axis =
        min_val, _ = range_from_cfg(FXP_CONFIGS[acc_dtype_name])
        mval_op = cdlt.dummy_op('min_val', min_val)
        min_op = cdlt.create_temp_operand([simd_size], "IMM", name='min_val')

        cdlt.configure("start", "IMM", immediate_value=mval_op)

        # Max reduce output
        mx = cdlt.create_operand_template("mx", OP_DTYPES, [N, C, W], default_dtype=acc_dtype)
        cdlt.add_temp_operand(mx)

        # Expontential temporaries
        z = cdlt.create_operand_template("z", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.add_temp_operand(z)

        b = cdlt.create_operand_template("b", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.add_temp_operand(b)

        y = cdlt.create_operand_template("y", OP_DTYPES, [N, C, H, W], default_dtype=acc_dtype)
        cdlt.add_temp_operand(y)
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(data, ["DRAM", "VMEM1"])
                        mx.start_location = "VMEM2"
                        mx.set_write_destination("VMEM2")
                        b.start_location = "VMEM2"
                        b.set_write_destination("VMEM2")
                        out.set_write_destination("VMEM1")
                        z.set_write_destination("VMEM1")
                        y.set_write_destination("VMEM2")
                        # First, compute the max value by setting memory to the minimum possible value and subtracting
                        cdlt.compute("MOVE", [min_op], [mx[n, c, w]], target="SIMD")
                        cdlt.compute("MAX", [data[n, c, h, w], mx[n, c, w]], [mx[n, c, w]], target="SIMD")
                        cdlt.compute("SUB", [data[n, c, h, w], mx[n, c, w]], [out[n, c, h, w]], target="SIMD")

                        # Now, Comptue exp
                        cdlt.compute("DIV", [out[n, c, h, w], qln2], [z[n, c, h, w]], target="SIMD")
                        cdlt.compute("MUL", [z[n, c, h, w], neg_one], [z[n, c, h, w]], target="SIMD")
                        cdlt.compute("MUL", [z[n, c, h, w], qln2], [y[n, c, h, w]], target="SIMD")
                        cdlt.compute("ADD", [out[n, c, h, w], y[n, c, h, w]], [y[n, c, h, w]], target="SIMD")
                        cdlt.compute("ADD", [out[n, c, h, w], qb], [out[n, c, h, w]], target="SIMD")
                        cdlt.compute("MOVE", [out[n, c, h, w]], [b[n, c, h, w]], target="SIMD")
                        cdlt.compute("MUL", [out[n, c, h, w], b[n, c, h, w]], [out[n, c, h, w]], target="SIMD")
                        cdlt.compute("ADD", [out[n, c, h, w], qc], [out[n, c, h, w]], target="SIMD")
                        cdlt.compute("FLOOR", [z[n, c, h, w]], [b[n, c, h, w]], target="SIMD")
                        cdlt.compute("RSHIFT", [out[n, c, h, w], b[n, c, h, w]], [out[n, c, h, w]], target="SIMD")
                        #
                        # # Next, compute sum for denominator
                        cdlt.compute("MOVE", [one], [mx[n, c, w]], target="SIMD")
                        cdlt.compute("MACC", [out[n, c, h, w], one, mx[n, c, w]], [mx[n, c, w]], target="SIMD")
                        # #
                        # # # Finally, divide the exponent by the denom
                        cdlt.compute("DIV", [out[n, c, h, w], mx[n, c, w]], [out[n, c, h, w]], target="SIMD")
                        cdlt.transfer(out, ["VMEM1", "DRAM"])

        cdlt.configure('end', 'SIMD')
    cdlt = add_simd_constraint(hag, cdlt, "W")
    cdlt = add_simd_tile_constraint(hag, cdlt, ["H"])
    return cdlt

def gelu(hag):
    acc_dtype_name = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    inpt_dtype = DTYPE_MAP[f"FXP{hag.meta_cfg['DATA_WIDTH']}"]
    acc_dtype = DTYPE_MAP[acc_dtype_name]
    with CodeletTemplate('gelu') as cdlt:
        B = cdlt.dummy_op("B", cdlt.node.inputs[0].shape[0])
        M = cdlt.dummy_op("M", cdlt.node.inputs[0].shape[1])
        P = cdlt.dummy_op("P", cdlt.node.inputs[0].shape[2])
        simd_size = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        data = cdlt.create_operand_template("data", OP_DTYPES, [B, M, P], default_dtype=acc_dtype)
        out = cdlt.create_operand_template("out", OP_DTYPES, [B, M, P], default_dtype=inpt_dtype)

        cdlt.set_inputs([data])
        cdlt.set_outputs([out])

        s_x = cdlt.create_operand_template("s_x", OP_DTYPES, [B, M, P], default_dtype=acc_dtype)
        cdlt.add_temp_operand(s_x)

        y = cdlt.create_operand_template("y", OP_DTYPES, [B, M, P], default_dtype=acc_dtype)
        cdlt.add_temp_operand(y)

        y_temp = cdlt.create_operand_template("y_temp", OP_DTYPES, [B, M, P], default_dtype=acc_dtype)
        cdlt.add_temp_operand(y_temp)

        cdlt.configure('start', 'SIMD')

        a = -0.2888
        b = 1.769
        c = 1.
        s = 0.6/np.sqrt(2)
        s_erf = a*s**2

        b_s_const = -b / s
        qb_const = b / s
        qc_const = c/a*s**2
        q1_const = 1./s_erf

        b_s = create_immediate_with_operand(cdlt, 'b_s', CAST_FUNC(b_s_const, acc_dtype_name), simd_size=simd_size)
        qb = create_immediate_with_operand(cdlt, 'qb', CAST_FUNC(qb_const, acc_dtype_name), simd_size=simd_size)
        qc = create_immediate_with_operand(cdlt, 'qc', CAST_FUNC(qc_const, acc_dtype_name), simd_size=simd_size)
        q1 = create_immediate_with_operand(cdlt, 'q1', CAST_FUNC(q1_const, acc_dtype_name), simd_size=simd_size)

        with cdlt.loop(P) as p:
            with cdlt.loop(B) as b:
                with cdlt.loop(M) as m:
                    cdlt.transfer(data, ["DRAM", "VMEM1"])
                    out.set_write_destination('VMEM1')
                    s_x.set_write_destination("VMEM1")
                    y.set_write_destination("VMEM2")
                    y.start_location = "VMEM2"

                    y_temp.set_write_destination("VMEM1")
                    y_temp.start_location = "VMEM1"

                    indices = (b, m, p)
                    # Sign
                    cdlt.compute("SIGN", [data[indices]], [s_x[indices]], target="SIMD")

                    # ABS
                    cdlt.compute("ABS", [data[indices]], [y[indices]], target="SIMD")

                    # CLIP
                    cdlt.compute("MIN", [y[indices], b_s], [y[indices]], target="SIMD")

                    # I-POLY
                    cdlt.compute("ADD", [y[indices], qb], [y[indices]], target="SIMD")

                    # (q + qb) ^ 2
                    cdlt.compute("MOVE", [y[indices]], [y_temp[indices]], target="SIMD")
                    cdlt.compute("MUL", [y[indices], y_temp[indices]], [y[indices]], target="SIMD")

                    # + qc
                    cdlt.compute("ADD", [y[indices], qc], [y[indices]], target="SIMD")

                    # Qsgn *Q
                    cdlt.compute("MUL", [y[indices], s_x[indices]], [y[indices]],
                                 target="SIMD")
                    cdlt.compute("ADD", [y[indices], q1], [y[indices]], target="SIMD")
                    cdlt.compute("MUL", [data[indices], y[indices]], [out[indices]], target="SIMD")

                    cdlt.transfer(out, ["VMEM1", "DRAM"])
        cdlt.configure('end', 'SIMD')
    cdlt = add_simd_constraint(hag, cdlt, "P")
    return cdlt

def load_dnn_cdlts(cfg):

    DNN_CDLTS = {
        "avg_pool": averagepool2d,
        "softmax": softmax,
        "softmax4d": softmax4d,
        "batch_norm": batch_norm,
        "cross_entropy_loss": cross_entropy_loss,
        "bias_add": bias_add,
        "depthwise_conv": depthwise_conv,
        "depthwise_conv_bias": depthwise_conv_bias,
        "global_avg_pool": global_avg_pool,
        "max_pool": maxpool2d,
        "mean_var": mean_var,
        "gelu": gelu,
    }
    return DNN_CDLTS