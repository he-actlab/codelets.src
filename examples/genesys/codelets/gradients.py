from codelets.adl.graph import ArchitectureNode
from codelets.templates.codelet_template import CodeletTemplate
from examples.genesys import OP_DTYPES


def sgd1d(hag: ArchitectureNode):

    with CodeletTemplate("sgd1d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes["SIMD"].dimensions[0])

        param = cdlt.create_operand_template("param", OP_DTYPES, [N], default_dtype=OP_DTYPES[2])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N], default_dtype=OP_DTYPES[2])
        updated_param = cdlt.create_operand_template("updated", OP_DTYPES, [N], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([param, grad])
        cdlt.set_outputs([updated_param])
        cdlt.configure("start", "SIMD")
        lr = cdlt.dummy_op("lr", cdlt.node.kwargs['lr'])
        momentum = cdlt.dummy_op("momentum", cdlt.node.kwargs['momentum'])

        cdlt.configure("start", "IMM", immediate_value=lr, index=0)
        cdlt.configure("start", "IMM", immediate_value=momentum, index=1)
        itemp1 = cdlt.create_operand_template("itemp1", OP_DTYPES, [N], default_dtype=OP_DTYPES[2])
        itemp2 = cdlt.create_operand_template("itemp2", OP_DTYPES, [N], default_dtype=OP_DTYPES[2])
        cdlt.add_temp_operand(itemp1)
        cdlt.add_temp_operand(itemp2)
        lr_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        momentum_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        with cdlt.loop(N) as n:
            cdlt.transfer(param[n], ["DRAM", "VMEM1"])
            cdlt.transfer(grad[n], ["DRAM", "VMEM2"])
            updated_param.set_write_destination("VMEM1")
            itemp1.set_write_destination("VMEM2")
            itemp2.set_write_destination("VMEM1")
            cdlt.compute("MUL", [param, momentum_op], [itemp1], target="SIMD")
            cdlt.compute("MUL", [grad, lr_op], [itemp2], target="SIMD")
            cdlt.compute("SUB", [itemp1, itemp2], [updated_param], target="SIMD")
            cdlt.transfer(updated_param[n], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    return cdlt


def sgd2d(hag: ArchitectureNode):

    with CodeletTemplate("sgd2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes["SIMD"].dimensions[0])

        param = cdlt.create_operand_template("param", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        updated_param = cdlt.create_operand_template("updated", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([param, grad])
        cdlt.set_outputs([updated_param])
        cdlt.configure("start", "SIMD")
        lr = cdlt.dummy_op("lr", cdlt.node.kwargs['lr'])
        momentum = cdlt.dummy_op("momentum", cdlt.node.kwargs['momentum'])

        cdlt.configure("start", "IMM", immediate_value=lr, index=0)
        cdlt.configure("start", "IMM", immediate_value=momentum, index=1)
        itemp1 = cdlt.create_operand_template("itemp1", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        itemp2 = cdlt.create_operand_template("itemp2", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.add_temp_operand(itemp1)
        cdlt.add_temp_operand(itemp2)
        lr_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        momentum_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(param[n, c], ["DRAM", "VMEM1"])
                cdlt.transfer(grad[n, c], ["DRAM", "VMEM2"])
                updated_param.set_write_destination("VMEM1")
                itemp1.set_write_destination("VMEM2")
                itemp2.set_write_destination("VMEM1")
                cdlt.compute("MUL", [param, momentum_op], [itemp1], target="SIMD")
                cdlt.compute("MUL", [grad, lr_op], [itemp2], target="SIMD")
                cdlt.compute("SUB", [itemp1, itemp2], [updated_param], target="SIMD")
                cdlt.transfer(updated_param[n, c], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt


def sgd3d(hag: ArchitectureNode):

    with CodeletTemplate("sgd3d") as cdlt:
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        param = cdlt.create_operand_template("param", OP_DTYPES, [C, H, W], default_dtype=OP_DTYPES[2])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [C, H, W], default_dtype=OP_DTYPES[2])
        updated_param = cdlt.create_operand_template("updated", OP_DTYPES, [C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([param, grad])
        cdlt.set_outputs([updated_param])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            with cdlt.loop(H) as h:
                with cdlt.loop(W) as w:
                    cdlt.transfer(param[c, h, w], ["DRAM", "VMEM1"])
                    cdlt.transfer(grad[c, h, w], ["DRAM", "VMEM2"])
                    updated_param.set_write_destination("VMEM1")
                    cdlt.compute("ADD", [param, grad], [updated_param], target="SIMD")
                    cdlt.transfer(updated_param[c, h, w], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt


def sgd4d(hag: ArchitectureNode):

    with CodeletTemplate("sgd4d") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes["SIMD"].dimensions[0])

        param = cdlt.create_operand_template("param", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        updated_param = cdlt.create_operand_template("updated", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([param, grad])
        cdlt.set_outputs([updated_param])
        cdlt.configure("start", "SIMD")
        lr = cdlt.dummy_op("lr", cdlt.node.kwargs['lr'])
        momentum = cdlt.dummy_op("momentum", cdlt.node.kwargs['momentum'])

        cdlt.configure("start", "IMM", immediate_value=lr, index=0)
        cdlt.configure("start", "IMM", immediate_value=momentum, index=1)
        itemp1 = cdlt.create_operand_template("itemp1", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        itemp2 = cdlt.create_operand_template("itemp2", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.add_temp_operand(itemp1)
        cdlt.add_temp_operand(itemp2)
        lr_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        momentum_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(param[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(grad[n, c, h, w], ["DRAM", "VMEM2"])
                        updated_param.set_write_destination("VMEM1")
                        itemp1.set_write_destination("VMEM2")
                        itemp2.set_write_destination("VMEM1")
                        cdlt.compute("MUL", [param, momentum_op], [itemp1], target="SIMD")
                        cdlt.compute("MUL", [grad, lr_op], [itemp2], target="SIMD")
                        cdlt.compute("SUB", [itemp1, itemp2], [updated_param], target="SIMD")
                        cdlt.transfer(updated_param[n, c, h, w], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt


def batchnorm_grad(hag: ArchitectureNode):

    with CodeletTemplate("batchnorm_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        scale = cdlt.create_operand_template("scale", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])
        offset = cdlt.create_operand_template("offset", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])
        mean = cdlt.create_operand_template("mean", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])
        istd = cdlt.create_operand_template("istd", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])
        xhat = cdlt.create_operand_template("xhat", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])

        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        scale_grad = cdlt.create_operand_template("scale_grad", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])
        offset_grad = cdlt.create_operand_template("offset_grad", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([data, scale, offset, mean, istd, grad])
        cdlt.set_outputs([data_grad, scale_grad, offset_grad])

        temp1 = cdlt.create_operand_template("temp1", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])
        temp1.start_location = "VMEM1"
        temp1.set_write_destination("VMEM1")

        cdlt.add_temp_operand(temp1)

        temp2 = cdlt.create_operand_template("temp2", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        temp2.start_location = "VMEM1"
        temp2.set_write_destination("VMEM1")
        cdlt.add_temp_operand(temp2)

        temp3 = cdlt.create_operand_template("temp3", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        temp3.start_location = "VMEM1"
        temp3.set_write_destination("VMEM1")

        temp4 = cdlt.create_operand_template("temp4", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        temp4.start_location = "VMEM1"
        temp4.set_write_destination("VMEM1")

        temp5 = cdlt.create_operand_template("temp5", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        temp5.start_location = "VMEM1"
        temp5.set_write_destination("VMEM1")

        cdlt.add_temp_operand(temp3)
        cdlt.add_temp_operand(temp4)
        cdlt.add_temp_operand(temp5)

        numer = cdlt.create_operand_template("numer", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
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
            cdlt.transfer(offset_grad[c], ["DRAM", "VMEM2"])
            cdlt.transfer(mean[c], ["DRAM", "VMEM2"])
            cdlt.transfer(istd[c], ["DRAM", "VMEM2"])
            cdlt.transfer(scale_grad[c], ["DRAM", "VMEM2"])
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(data[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(grad[n, c, h, w], ["DRAM", "VMEM2"])
                        scale_grad.set_write_destination("VMEM1")
                        offset_grad.set_write_destination("VMEM1")
                        numer.set_write_destination("VMEM1")
                        xhat.set_write_destination("VMEM1")
                        cdlt.compute("SUB", [data, mean], [numer], target="SIMD")
                        cdlt.compute("MUL", [numer, istd], [xhat], target="SIMD")
                        cdlt.compute("MUL", [xhat, grad], [numer], target="SIMD")
                        cdlt.compute("ADD", [scale_grad, numer], [scale_grad], target="SIMD")
                        cdlt.compute("ADD", [grad, offset_grad], [offset_grad], target="SIMD")

            with cdlt.loop(N) as n1:
                with cdlt.loop(H) as h1:
                    with cdlt.loop(W) as w1:
                        cdlt.transfer(scale[c], ["DRAM", "VMEM2"])
                        data_grad.set_write_destination("VMEM1")
                        cdlt.compute("MUL", [scale, istd], [temp1], target="SIMD")
                        cdlt.compute("DIV", [temp1, denom_op], [temp1], target="SIMD")
                        cdlt.compute("MUL", [denom_op, grad], [temp2], target="SIMD")
                        cdlt.compute("MUL", [xhat, scale_grad], [temp3], target="SIMD")
                        cdlt.compute("SUB", [temp2, temp3], [temp4], target="SIMD")
                        cdlt.compute("SUB", [temp4, offset_grad], [temp5], target="SIMD")
                        cdlt.compute("MUL", [temp1, temp5], [data_grad], target="SIMD")
                        cdlt.transfer(data_grad[n1, c, h1, w1], ["VMEM1", "DRAM"])
            cdlt.transfer(offset_grad[c], ["VMEM1", "DRAM"])
            cdlt.transfer(scale_grad[c], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions
    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")

    return cdlt


def flatten_grad(hag: ArchitectureNode):

    with CodeletTemplate("flatten_grad") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([out])
        cdlt.configure("end", "SIMD")

    return cdlt


def relu_grad(hag: ArchitectureNode):

    with CodeletTemplate("relu_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])
        cdlt.configure("start", "SIMD")
        # cdlt.configure("start", "VMEM")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(data[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(grad[n, c, h, w], ["DRAM", "VMEM1"])
                        data_grad.set_write_destination("VMEM1")
                        cdlt.compute("RELU", [data, grad], [data_grad], target="SIMD")
                        cdlt.transfer(data_grad[n, c, h, w], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    return cdlt


def relu_grad2d(hag: ArchitectureNode):

    with CodeletTemplate("relu_grad2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])
        cdlt.configure("start", "SIMD")
        # cdlt.configure("start", "VMEM")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(data[n, c], ["DRAM", "VMEM1"])
                cdlt.transfer(grad[n, c], ["DRAM", "VMEM1"])
                data_grad.set_write_destination("VMEM1")
                cdlt.compute("RELU", [data, grad], [data_grad], target="SIMD")
                cdlt.transfer(data_grad[n, c], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt


def elem_tanh_grad(hag: ArchitectureNode):

    with CodeletTemplate("elem_tanh_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])
        temp1 = cdlt.create_operand_template("temp1", OP_DTYPES, [SIMD_SIZE], default_dtype=OP_DTYPES[2])
        temp1.start_location = "VMEM1"
        cdlt.add_temp_operand(temp1)
        one_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=1, index=0)
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(data[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(grad[n, c, h, w], ["DRAM", "VMEM1"])
                        data.set_write_destination("VMEM1")
                        data_grad.set_write_destination("VMEM1")
                        cdlt.compute("MUL", [data, data], [data], target="SIMD")
                        one_op.set_write_destination("VMEM1")
                        temp1.set_write_destination("VMEM1")
                        cdlt.compute("SUB", [one_op, data], [temp1], target="SIMD")
                        cdlt.compute("MUL", [grad, temp1], [data_grad], target="SIMD")
                        cdlt.transfer(data_grad[n, c, h, w], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt


def elem_tanh_grad2d(hag: ArchitectureNode):

    with CodeletTemplate("elem_tanh_grad2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data, grad])
        cdlt.set_outputs([data_grad])
        one_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        temp1 = cdlt.create_operand_template("temp1", OP_DTYPES, [SIMD_SIZE], default_dtype=OP_DTYPES[2])
        temp1.start_location = "VMEM1"

        cdlt.add_temp_operand(temp1)

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=1, index=0)

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(data[n, c], ["DRAM", "VMEM1"])
                cdlt.transfer(grad[n, c], ["DRAM", "VMEM1"])
                data.set_write_destination("VMEM1")
                data_grad.set_write_destination("VMEM1")
                cdlt.compute("MUL", [data, data], [data], target="SIMD")
                one_op.set_write_destination("VMEM1")
                temp1.set_write_destination("VMEM1")
                cdlt.compute("SUB", [one_op, data], [temp1], target="SIMD")
                cdlt.compute("MUL", [grad, temp1], [data_grad], target="SIMD")
                cdlt.transfer(data_grad[n, c], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt


def max_pool_grad(hag: ArchitectureNode):
    #

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
        data = cdlt.create_operand_template("max_pool_data", OP_DTYPES, [N, C, IH, IW], default_dtype=OP_DTYPES[2])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, OH, OW], default_dtype=OP_DTYPES[2])
        data_grad = cdlt.create_operand_template("max_pool_data_grad", OP_DTYPES, [N, C, IH, IW], default_dtype=OP_DTYPES[2])
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
                                cdlt.transfer(data[n, c, y*sy + kh, x*sx + kw], ["DRAM", "VMEM1"])
                                cdlt.transfer(grad[n, c, y, x], ["DRAM", "VMEM1"])
                                data_grad.set_write_destination("VMEM1")
                                cdlt.compute("MAX", [data, grad], [data_grad], target="SIMD")
                                cdlt.transfer(data_grad[n, c, y*sy + kh, x*sx + kw], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt


def average_pool_grad(hag: ArchitectureNode):

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

        data = cdlt.create_operand_template("avg_pool_data", OP_DTYPES, [N, C, IH, IW], default_dtype=OP_DTYPES[2])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, OH, OW], default_dtype=OP_DTYPES[2])
        #
        data_grad = cdlt.create_operand_template("avg_pool_data_grad", OP_DTYPES, [N, C, IH, IW], default_dtype=OP_DTYPES[2])
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
                                cdlt.transfer(data[n, c, y*sy + kh, x*sx + kw], ["DRAM", "VMEM1"])
                                cdlt.transfer(grad[n, c, y, x], ["DRAM", "VMEM1"])
                                data_grad.set_write_destination("VMEM1")
                                cdlt.compute("MAX", [data, grad], [data_grad], target="SIMD")
                                cdlt.transfer(data_grad[n, c, y*sy + kh, x*sx + kw], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt


def global_average_pool_grad(hag: ArchitectureNode):


    # # TODO: Add option to create operand
    with CodeletTemplate("global_average_pool_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, IH, IW], default_dtype=OP_DTYPES[2])
        grad = cdlt.create_operand_template("grad", OP_DTYPES, [N, C, OH, OW], default_dtype=OP_DTYPES[2])
        #
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C, IH, IW], default_dtype=OP_DTYPES[2])
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
                                cdlt.transfer(data[n, c, iy, ix], ["DRAM", "VMEM1"])
                                cdlt.transfer(grad[n, c, oy, ox], ["DRAM", "VMEM1"])
                                data_grad.set_write_destination("VMEM1")
                                cdlt.compute("MEAN", [data, grad], [data_grad], target="SIMD")
                                cdlt.transfer(data_grad[n, c, iy, ix], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt


def cross_entropy_loss_grad(hag: ArchitectureNode):

    with CodeletTemplate("cross_entropy_loss_grad") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        target = cdlt.create_operand_template("target", OP_DTYPES, [N], default_dtype=OP_DTYPES[2])
        data_grad = cdlt.create_operand_template("data_grad", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([data, target])
        cdlt.set_outputs([data_grad])

        cdlt.configure("start", "SIMD")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(data[n, c], ["DRAM", "VMEM1"])
                cdlt.transfer(target[n], ["DRAM", "VMEM2"])
                data_grad.set_write_destination("VMEM1")
                cdlt.compute("SUB", [data, target], [data_grad], target="SIMD")
                cdlt.transfer(data_grad[n, c], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt

GRADIENT_CDLTS = {
    'average_pool_grad': average_pool_grad,
    "batchnorm_grad": batchnorm_grad,
    "cross_entropy_loss_grad": cross_entropy_loss_grad,
    'elem_tanh_grad': elem_tanh_grad,
    'elem_tanh_grad2d': elem_tanh_grad2d,
    "flatten_grad": flatten_grad,
    'global_average_pool_grad': global_average_pool_grad,
    'max_pool_grad': max_pool_grad,
    'relu_grad2d': relu_grad2d,
    'relu_grad': relu_grad,
    "sgd1d": sgd1d,
    "sgd2d": sgd2d,
    "sgd3d": sgd3d,
    "sgd4d": sgd4d,
}