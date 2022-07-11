from codelets.adl.graph import ArchitectureNode
from codelets.templates.codelet_template import CodeletTemplate
from examples.genesys import OP_DTYPES, DTYPE_MAP
from . import add_simd_constraint


def sgd1d(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("sgd1d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes["SIMD"].dimensions[0])

        param = cdlt.create_operand_template("param", OP_DTYPES, [N], default_dtype=DTYPE_MAP[acc_dtype])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N], default_dtype=DTYPE_MAP[acc_dtype])
        updated_param = cdlt.create_operand_template("updated", OP_DTYPES, [N], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.set_inputs([param, grad])
        cdlt.set_outputs([updated_param])
        cdlt.configure("start", "SIMD")
        lr = cdlt.dummy_op("lr", cdlt.node.kwargs['lr'])
        momentum = cdlt.dummy_op("momentum", cdlt.node.kwargs['momentum'])

        cdlt.configure("start", "IMM", immediate_value=lr, index=0)
        cdlt.configure("start", "IMM", immediate_value=momentum, index=1)
        itemp1 = cdlt.create_operand_template("itemp1", OP_DTYPES, [N], default_dtype=DTYPE_MAP[acc_dtype])
        itemp2 = cdlt.create_operand_template("itemp2", OP_DTYPES, [N], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.add_temp_operand(itemp1)
        cdlt.add_temp_operand(itemp2)
        lr_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        momentum_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        with cdlt.loop(N) as n:
            cdlt.transfer(param, ["DRAM", "VMEM1"])
            cdlt.transfer(grad, ["DRAM", "VMEM2"])
            updated_param.set_write_destination("VMEM1")
            itemp1.set_write_destination("VMEM2")
            itemp2.set_write_destination("VMEM1")
            cdlt.compute("MUL", [param[n], momentum_op], [itemp1[n]], target="SIMD")
            cdlt.compute("MUL", [grad[n], lr_op], [itemp2[n]], target="SIMD")
            cdlt.compute("SUB", [itemp1[n], itemp2[n]], [updated_param[n]], target="SIMD")
            cdlt.transfer(updated_param, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
        cdlt = add_simd_constraint(hag, cdlt, "N")

    return cdlt



def sgd2d(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("sgd2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes["SIMD"].dimensions[0])

        param = cdlt.create_operand_template("param", OP_DTYPES, [N, C], default_dtype=DTYPE_MAP[acc_dtype])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C], default_dtype=DTYPE_MAP[acc_dtype])
        updated_param = cdlt.create_operand_template("updated", OP_DTYPES, [N, C], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.set_inputs([param, grad])
        cdlt.set_outputs([updated_param])
        cdlt.configure("start", "SIMD")
        lr = cdlt.dummy_op("lr", cdlt.node.kwargs['lr'])
        momentum = cdlt.dummy_op("momentum", cdlt.node.kwargs['momentum'])

        cdlt.configure("start", "IMM", immediate_value=lr, index=0)
        cdlt.configure("start", "IMM", immediate_value=momentum, index=1)
        itemp1 = cdlt.create_operand_template("itemp1", OP_DTYPES, [N, C], default_dtype=DTYPE_MAP[acc_dtype])
        itemp2 = cdlt.create_operand_template("itemp2", OP_DTYPES, [N, C], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.add_temp_operand(itemp1)
        cdlt.add_temp_operand(itemp2)
        lr_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        momentum_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(param, ["DRAM", "VMEM1"])
                cdlt.transfer(grad, ["DRAM", "VMEM2"])
                updated_param.set_write_destination("VMEM1")
                itemp1.set_write_destination("VMEM2")
                itemp2.set_write_destination("VMEM1")
                cdlt.compute("MUL", [param[n, c], momentum_op], [itemp1[n, c]], target="SIMD")
                cdlt.compute("MUL", [grad[n, c], lr_op], [itemp2[n, c]], target="SIMD")
                cdlt.compute("SUB", [itemp1[n, c], itemp2[n, c]], [updated_param[n, c]], target="SIMD")
                cdlt.transfer(updated_param, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def sgd3d(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("sgd3d") as cdlt:
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        param = cdlt.create_operand_template("param", OP_DTYPES, [C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        updated_param = cdlt.create_operand_template("updated", OP_DTYPES, [C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.set_inputs([param, grad])
        cdlt.set_outputs([updated_param])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            with cdlt.loop(H) as h:
                with cdlt.loop(W) as w:
                    cdlt.transfer(param, ["DRAM", "VMEM1"])
                    cdlt.transfer(grad, ["DRAM", "VMEM2"])
                    updated_param.set_write_destination("VMEM1")
                    cdlt.compute("ADD", [param[c, h, w], grad[c, h, w]], [updated_param[c, h, w]], target="SIMD")
                    cdlt.transfer(updated_param, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def sgd4d(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("sgd4d") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes["SIMD"].dimensions[0])

        param = cdlt.create_operand_template("param", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        updated_param = cdlt.create_operand_template("updated", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.set_inputs([param, grad])
        cdlt.set_outputs([updated_param])
        cdlt.configure("start", "SIMD")
        lr = cdlt.dummy_op("lr", cdlt.node.kwargs['lr'])
        momentum = cdlt.dummy_op("momentum", cdlt.node.kwargs['momentum'])

        cdlt.configure("start", "IMM", immediate_value=lr, index=0)
        cdlt.configure("start", "IMM", immediate_value=momentum, index=1)
        itemp1 = cdlt.create_operand_template("itemp1", OP_DTYPES, [N, C], default_dtype=DTYPE_MAP[acc_dtype])
        itemp2 = cdlt.create_operand_template("itemp2", OP_DTYPES, [N, C], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.add_temp_operand(itemp1)
        cdlt.add_temp_operand(itemp2)
        lr_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        momentum_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(param, ["DRAM", "VMEM1"])
                        cdlt.transfer(grad, ["DRAM", "VMEM2"])
                        updated_param.set_write_destination("VMEM1")
                        itemp1.set_write_destination("VMEM2")
                        itemp2.set_write_destination("VMEM1")
                        cdlt.compute("MUL", [param[n, c, h, w], momentum_op], [itemp1[n, c]], target="SIMD")
                        cdlt.compute("MUL", [grad[n, c, h, w], lr_op], [itemp2[n, c]], target="SIMD")
                        cdlt.compute("SUB", [itemp1[n, c], itemp2[n, c]], [updated_param[n, c, h, w]], target="SIMD")
                        cdlt.transfer(updated_param, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def batchnorm_grad(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("batchnorm_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        scale = cdlt.create_operand_template("scale", OP_DTYPES, [C], default_dtype=DTYPE_MAP[acc_dtype])
        offset = cdlt.create_operand_template("offset", OP_DTYPES, [C], default_dtype=DTYPE_MAP[acc_dtype])
        mean = cdlt.create_operand_template("mean", OP_DTYPES, [C], default_dtype=DTYPE_MAP[acc_dtype])
        istd = cdlt.create_operand_template("istd", OP_DTYPES, [C], default_dtype=DTYPE_MAP[acc_dtype])
        xhat = cdlt.create_operand_template("xhat", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])

        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        scale_grad = cdlt.create_operand_template("scale_grad", OP_DTYPES, [C], default_dtype=DTYPE_MAP[acc_dtype])
        offset_grad = cdlt.create_operand_template("offset_grad", OP_DTYPES, [C], default_dtype=DTYPE_MAP[acc_dtype])

        cdlt.set_inputs([data, scale, offset, mean, istd, grad])
        cdlt.set_outputs([data_grad, scale_grad, offset_grad])

        temp1 = cdlt.create_operand_template("temp1", OP_DTYPES, [C], default_dtype=DTYPE_MAP[acc_dtype])
        temp1.start_location = "VMEM1"
        temp1.set_write_destination("VMEM1")

        cdlt.add_temp_operand(temp1)

        temp2 = cdlt.create_operand_template("temp2", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        temp2.start_location = "VMEM1"
        temp2.set_write_destination("VMEM1")
        cdlt.add_temp_operand(temp2)

        temp3 = cdlt.create_operand_template("temp3", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        temp3.start_location = "VMEM1"
        temp3.set_write_destination("VMEM1")

        temp4 = cdlt.create_operand_template("temp4", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        temp4.start_location = "VMEM1"
        temp4.set_write_destination("VMEM1")

        temp5 = cdlt.create_operand_template("temp5", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        temp5.start_location = "VMEM1"
        temp5.set_write_destination("VMEM1")

        cdlt.add_temp_operand(temp3)
        cdlt.add_temp_operand(temp4)
        cdlt.add_temp_operand(temp5)

        numer = cdlt.create_operand_template("numer", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.add_temp_operand(xhat)
        cdlt.add_temp_operand(numer)
        denom = cdlt.dummy_op("denom",
                              cdlt.node.inputs[0].shape[0] * cdlt.node.inputs[0].shape[2] * cdlt.node.inputs[0].shape[
                                  3])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        denom_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        cdlt.configure("start", "IMM", immediate_value=denom, index=0)
        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            cdlt.transfer(offset_grad, ["DRAM", "VMEM2"])
            cdlt.transfer(mean, ["DRAM", "VMEM2"])
            cdlt.transfer(istd, ["DRAM", "VMEM2"])
            cdlt.transfer(scale_grad, ["DRAM", "VMEM2"])
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(data, ["DRAM", "VMEM1"])
                        cdlt.transfer(grad, ["DRAM", "VMEM2"])
                        scale_grad.set_write_destination("VMEM1")
                        offset_grad.set_write_destination("VMEM1")
                        numer.set_write_destination("VMEM1")
                        xhat.set_write_destination("VMEM1")
                        cdlt.compute("SUB", [data[n, c, h, w], mean[c]], [numer[n, c, h, w]], target="SIMD")
                        cdlt.compute("MUL", [numer[n, c, h, w], istd[c]], [xhat[n, c, h, w]], target="SIMD")
                        cdlt.compute("MUL", [xhat[n, c, h, w], grad[n, c, h, w]], [numer[n, c, h, w]], target="SIMD")
                        cdlt.compute("ADD", [scale_grad[c], numer[n, c, h, w]], [scale_grad[c]], target="SIMD")
                        cdlt.compute("ADD", [grad[n, c, h, w], offset_grad[c]], [offset_grad[c]], target="SIMD")

            with cdlt.loop(N) as n1:
                with cdlt.loop(H) as h1:
                    with cdlt.loop(W) as w1:
                        cdlt.transfer(scale, ["DRAM", "VMEM2"])
                        data_grad.set_write_destination("VMEM1")
                        cdlt.compute("MUL", [scale[c], istd[c]], [temp1[c]], target="SIMD")
                        cdlt.compute("DIV", [temp1[c], denom_op], [temp1[c]], target="SIMD")
                        cdlt.compute("MUL", [denom_op, grad[n, c, h, w]], [temp2[n, c, h, w]], target="SIMD")
                        cdlt.compute("MUL", [xhat[n1, c, h1, w1], scale_grad[c]], [temp3[n1, c, h1, w1]], target="SIMD")
                        cdlt.compute("SUB", [temp2[n1, c, h1, w1], temp3[n1, c, h1, w1]], [temp4[n1, c, h1, w1]], target="SIMD")
                        cdlt.compute("SUB", [temp4[n1, c, h1, w1], offset_grad[c]], [temp5[n1, c, h1, w1]], target="SIMD")
                        cdlt.compute("MUL", [temp1[c], temp5[n1, c, h1, w1]], [data_grad[n1, c, h1, w1]], target="SIMD")
                        cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
            cdlt.transfer(offset_grad, ["VMEM1", "DRAM"])
            cdlt.transfer(scale_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def flatten_grad(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("flatten_grad") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=DTYPE_MAP[acc_dtype])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C], default_dtype=DTYPE_MAP[acc_dtype])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([out])
        cdlt.configure("end", "SIMD")

    return cdlt


def relu_grad(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("relu_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])
        cdlt.configure("start", "SIMD")
        # cdlt.configure("start", "VMEM")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(data, ["DRAM", "VMEM1"])
                        cdlt.transfer(grad, ["DRAM", "VMEM2"])
                        data_grad.set_write_destination("VMEM1")
                        cdlt.compute("RELU", [data[n, c, h, w], grad[n, c, h, w]], [data_grad[n, c, h, w]], target="SIMD")
                        cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def relu_grad2d(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("relu_grad2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=DTYPE_MAP[acc_dtype])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C], default_dtype=DTYPE_MAP[acc_dtype])
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(data, ["DRAM", "VMEM1"])
                cdlt.transfer(grad, ["DRAM", "VMEM1"])
                data_grad.set_write_destination("VMEM1")
                cdlt.compute("RELU", [data[n, c], grad[n, c]], [data_grad[n, c]], target="SIMD")
                cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def elem_tanh_grad(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("elem_tanh_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C, H, W], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])
        temp1 = cdlt.create_operand_template("temp1", OP_DTYPES, [SIMD_SIZE], default_dtype=DTYPE_MAP[acc_dtype])
        temp1.start_location = "VMEM1"
        cdlt.add_temp_operand(temp1)
        one_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=1, index=0)
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(data, ["DRAM", "VMEM1"])
                        cdlt.transfer(grad, ["DRAM", "VMEM1"])
                        data.set_write_destination("VMEM1")
                        data_grad.set_write_destination("VMEM1")
                        cdlt.compute("MUL", [data[n, c, h, w], data[n, c, h, w]], [data[n, c, h, w]], target="SIMD")
                        one_op.set_write_destination("VMEM1")
                        temp1.set_write_destination("VMEM1")
                        cdlt.compute("SUB", [one_op, data[n, c, h, w]], [temp1], target="SIMD")
                        cdlt.compute("MUL", [grad[n, c, h, w], temp1], [data_grad[n, c, h, w]], target="SIMD")
                        cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def elem_tanh_grad2d(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("elem_tanh_grad2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=DTYPE_MAP[acc_dtype])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C], default_dtype=DTYPE_MAP[acc_dtype])
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])
        one_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        temp1 = cdlt.create_operand_template("temp1", OP_DTYPES, [SIMD_SIZE], default_dtype=DTYPE_MAP[acc_dtype])
        temp1.start_location = "VMEM1"

        cdlt.add_temp_operand(temp1)

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=1, index=0)

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(data, ["DRAM", "VMEM1"])
                cdlt.transfer(grad, ["DRAM", "VMEM1"])
                data.set_write_destination("VMEM1")
                data_grad.set_write_destination("VMEM1")
                cdlt.compute("MUL", [data[n, c], data[n, c]], [data[n, c]], target="SIMD")
                one_op.set_write_destination("VMEM1")
                temp1.set_write_destination("VMEM1")
                cdlt.compute("SUB", [one_op, data[n, c]], [temp1], target="SIMD")
                cdlt.compute("MUL", [grad, temp1], [data_grad[n, c]], target="SIMD")
                cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def max_pool_grad(hag: ArchitectureNode):
    #
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    # # TODO: Add option to create operand
    with CodeletTemplate("max_pool_grad") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        KH = cdlt.dummy_op("KH", cdlt.node.kernel_size[0])
        KW = cdlt.dummy_op("KW", cdlt.node.kernel_size[1])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])

        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])
        sy = cdlt.dummy_op("sy", cdlt.node.stride[0])
        sx = cdlt.dummy_op("sx", cdlt.node.stride[1])
        data = cdlt.create_operand_template("max_pool_data", OP_DTYPES, [N, C, IH, IW], default_dtype=DTYPE_MAP[acc_dtype])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, OH, OW], default_dtype=DTYPE_MAP[acc_dtype])
        data_grad = cdlt.create_operand_template("max_pool_data_grad", OP_DTYPES, [N, C, IH, IW], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])
        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=0, index=0)
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(KH) as kh:
                    with cdlt.loop(KW) as kw:
                        with cdlt.loop(OH) as y:
                            with cdlt.loop(OW) as x:
                                cdlt.transfer(data, ["DRAM", "VMEM1"])
                                cdlt.transfer(grad, ["DRAM", "VMEM1"])
                                data_grad.set_write_destination("VMEM1")
                                cdlt.compute("MAX", [data[n, c, y*sy + kh, x*sx + kw], grad[n,c,y,x]], [data_grad[n, c, y*sy + kh, x*sx + kw]], target="SIMD")
                                cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def average_pool_grad(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    # # TODO: Add option to create operand
    with CodeletTemplate("average_pool_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        KH = cdlt.dummy_op("KH", cdlt.node.kernel_size[0])
        KW = cdlt.dummy_op("KW", cdlt.node.kernel_size[1])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])
        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("avg_pool_data", OP_DTYPES, [N, C, IH, IW], default_dtype=DTYPE_MAP[acc_dtype])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, OH, OW], default_dtype=DTYPE_MAP[acc_dtype])
        #
        data_grad = cdlt.create_operand_template("avg_pool_data_grad", OP_DTYPES, [N, C, IH, IW], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])


        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=0, index=0)
        sy = cdlt.dummy_op('sy', cdlt.node.stride[0])
        sx = cdlt.dummy_op('sx', cdlt.node.stride[1])
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(KH) as kh:
                    with cdlt.loop(KW) as kw:
                        with cdlt.loop(OH) as y:
                            with cdlt.loop(OW) as x:
                                cdlt.transfer(data, ["DRAM", "VMEM1"])
                                cdlt.transfer(grad, ["DRAM", "VMEM1"])
                                data_grad.set_write_destination("VMEM1")
                                cdlt.compute("MAX", [data[n, c, y*sy + kh, x*sx + kw],
                                                     grad[n, c, y, x]],
                                             [data_grad[n, c, y*sy + kh, x*sx + kw]], target="SIMD")
                                cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def global_average_pool_grad(hag: ArchitectureNode):

    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    # # TODO: Add option to create operand
    with CodeletTemplate("global_average_pool_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, IH, IW], default_dtype=DTYPE_MAP[acc_dtype])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, OH, OW], default_dtype=DTYPE_MAP[acc_dtype])
        #
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C, IH, IW], default_dtype=DTYPE_MAP[acc_dtype])
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=0, index=0)

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(IH) as iy:
                    with cdlt.loop(IW) as ix:
                        with cdlt.loop(OH) as oy:
                            with cdlt.loop(OW) as ox:
                                cdlt.transfer(data, ["DRAM", "VMEM1"])
                                cdlt.transfer(grad, ["DRAM", "VMEM1"])
                                data_grad.set_write_destination("VMEM1")
                                cdlt.compute("MEAN", [data[n, c, iy, ix], grad[n, c, oy, ox]], [data_grad[n, c, iy, ix]], target="SIMD")
                                cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt


def cross_entropy_loss_grad(hag: ArchitectureNode):
    inpt_dtype = f"FXP{hag.meta_cfg['DATA_WIDTH']}"
    acc_dtype = f"FXP{hag.meta_cfg['ACC_WIDTH']}"
    with CodeletTemplate("cross_entropy_loss_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=DTYPE_MAP[acc_dtype])
        target = cdlt.create_operand_template("target", OP_DTYPES, [N], default_dtype=DTYPE_MAP[acc_dtype])
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C], default_dtype=DTYPE_MAP[acc_dtype])

        cdlt.set_inputs([data, target])
        cdlt.set_outputs([data_grad])

        cdlt.configure("start", "SIMD")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(data, ["DRAM", "VMEM1"])
                cdlt.transfer(target, ["DRAM", "VMEM2"])
                data_grad.set_write_destination("VMEM1")
                cdlt.compute("SUB", [data[n, c], target[n]], [data_grad[n, c]], target="SIMD")
                cdlt.transfer(data_grad, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")

    return cdlt

def load_gradient_cdlts(cfg):

    GRADIENT_CDLTS = {
        # 'average_pool_grad': average_pool_grad,
        # "batchnorm_grad": batchnorm_grad,
        # "cross_entropy_loss_grad": cross_entropy_loss_grad,
        # 'elem_tanh_grad': elem_tanh_grad,
        # 'elem_tanh_grad2d': elem_tanh_grad2d,
        # "flatten_grad": flatten_grad,
        # 'global_average_pool_grad': global_average_pool_grad,
        # 'max_pool_grad': max_pool_grad,
        # 'relu_grad2d': relu_grad2d,
        # 'relu_grad': relu_grad,
        # "sgd1d": sgd1d,
        # "sgd2d": sgd2d,
        # "sgd3d": sgd3d,
        # "sgd4d": sgd4d,
    }
    return GRADIENT_CDLTS