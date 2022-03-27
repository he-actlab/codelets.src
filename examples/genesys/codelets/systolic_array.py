from codelets.adl.graph import ArchitectureNode
from codelets.templates.codelet_template import CodeletTemplate
from examples.genesys import OP_DTYPES
from . import add_conv_constraints, add_gemm_constraints



def gemm(hag: ArchitectureNode):


    with CodeletTemplate("gemm") as cdlt:

        P = cdlt.dummy_op("P", cdlt.node.inputs[2].shape[0])
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[1])
        M = cdlt.dummy_op("M", cdlt.node.inputs[0].shape[0])

        data = cdlt.create_operand_template("data", OP_DTYPES, [M, N], default_dtype=OP_DTYPES[0])
        weight = cdlt.create_operand_template("weight", OP_DTYPES, [N, P], default_dtype=OP_DTYPES[0])
        bias = cdlt.create_operand_template("bias", OP_DTYPES, [P], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [M, P], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([data, weight, bias])
        cdlt.set_outputs([out])

        cdlt.configure("start", "systolic_array")
        cdlt.configure("start", "WBUF")
        cdlt.configure("start", "IBUF")
        cdlt.configure("start", "BBUF")
        cdlt.configure("start", "OBUF")
        #
        with cdlt.loop(P) as p:
            with cdlt.loop(N) as n:
                with cdlt.loop(M) as m:
                    cdlt.transfer(data, ["DRAM", "IBUF"])
                    cdlt.transfer(weight, ["DRAM", "WBUF"])
                    cdlt.transfer(bias, ["DRAM", "BBUF"])
                    cdlt.transfer(out, ["DRAM", "OBUF"])
                    out.set_write_destination("OBUF")
                    cdlt.compute("MVMUL", [data[m, n], weight[n, p], bias[p], out[m,p]], [out[m,p]], target="pe_array")

                    cdlt.transfer(out, ["OBUF", "DRAM"])

        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "BBUF")
        cdlt.configure("end", "systolic_array")

    cdlt = add_gemm_constraints(hag, cdlt)

    return cdlt


def gemm_no_bias(hag: ArchitectureNode):

    with CodeletTemplate("gemm_no_bias") as cdlt:
        P = cdlt.dummy_op("P", cdlt.node.inputs[1].shape[1])
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[1])
        M = cdlt.dummy_op("M", cdlt.node.inputs[0].shape[0])
        data = cdlt.create_operand_template("data", OP_DTYPES, [M, N], default_dtype=OP_DTYPES[0])
        weight = cdlt.create_operand_template("weight", OP_DTYPES, [N, P], default_dtype=OP_DTYPES[0])
        out = cdlt.create_operand_template("out", OP_DTYPES, [M, P], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([data, weight])
        cdlt.set_outputs([out])

        cdlt.configure("start", "systolic_array")
        cdlt.configure("start", "WBUF")
        cdlt.configure("start", "IBUF")
        cdlt.configure("start", "OBUF")

        with cdlt.loop(M) as m:
            with cdlt.loop(N) as n:
                with cdlt.loop(P) as p:
                    cdlt.transfer(data, ["DRAM", "IBUF"])
                    cdlt.transfer(weight, ["DRAM", "WBUF"])
                    cdlt.transfer(out, ["DRAM", "OBUF"])
                    out.set_write_destination("OBUF")
                    cdlt.compute("MVMUL", [data[m, n], weight[n, p], out[m,p]], [out[m,p]], target="pe_array")
                    cdlt.transfer(out, ["OBUF", "DRAM"])

        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "systolic_array")
    cdlt = add_gemm_constraints(hag, cdlt)

    return cdlt


def matmul(hag: ArchitectureNode):

    with CodeletTemplate("matmul") as cdlt:
        P = cdlt.dummy_op("P", cdlt.node.outputs[0].shape[1])
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[1])
        M = cdlt.dummy_op("M", cdlt.node.inputs[0].shape[0])
        data = cdlt.create_operand_template("data", OP_DTYPES, [M, N], default_dtype=OP_DTYPES[0])
        weight = cdlt.create_operand_template("weight", OP_DTYPES, [N, P], default_dtype=OP_DTYPES[0])
        out = cdlt.create_operand_template("out", OP_DTYPES, [M, P], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([data, weight])
        cdlt.set_outputs([out])

        # cdlt.configure("start", "systolic_array")
        # cdlt.configure("start", "WBUF")
        # cdlt.configure("start", "IBUF")
        # cdlt.configure("start", "OBUF")

        with cdlt.loop(P) as p:
            with cdlt.loop(N) as n:
                with cdlt.loop(M) as m:
                    cdlt.transfer(data, ["DRAM", "IBUF"])
                    cdlt.transfer(weight, ["DRAM", "WBUF"])
                    cdlt.transfer(out, ["DRAM", "OBUF"])
                    out.set_write_destination("OBUF")
                    cdlt.compute("MVMUL", [data[m, n], weight[n, p], out[m,p]], [out[m,p]], target="pe_array")
                    cdlt.transfer(out, ["OBUF", "DRAM"])

        # TODO: Add store off chip
        # cdlt.configure("end", "WBUF")
        # cdlt.configure("end", "IBUF")
        # cdlt.configure("end", "OBUF")
        # cdlt.configure("end", "systolic_array")
    cdlt = add_gemm_constraints(hag, cdlt)

    return cdlt


def conv2d(hag: ArchitectureNode):
    # TODO: Need to figure out how to change the memory layout
    with CodeletTemplate("conv") as cdlt:
        stride = cdlt.dummy_op("stride", cdlt.node.stride)
        pad = cdlt.dummy_op("pad", cdlt.node.pad)
        OC = cdlt.dummy_op("OC", cdlt.node.outputs[0].shape[1])
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        IC = cdlt.dummy_op("IC", cdlt.node.inputs[0].shape[1])
        KH = cdlt.dummy_op("KH", cdlt.node.inputs[1].shape[2])
        KW = cdlt.dummy_op("KW", cdlt.node.inputs[1].shape[3])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])
        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, IC, IH, IW], default_dtype=OP_DTYPES[0])
        weight = cdlt.create_operand_template("weight", OP_DTYPES, [OC, IC, KH, KW], default_dtype=OP_DTYPES[0])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data, weight])
        cdlt.set_outputs([out])
        cdlt.configure("start", "systolic_array")
        cdlt.configure("start", "WBUF")
        cdlt.configure("start", "BBUF")

        cdlt.configure("start", "IBUF")
        cdlt.configure("start", "OBUF")

        # OS ->

        with cdlt.loop(OC) as oc:
            with cdlt.loop(N) as n:
                with cdlt.loop(IC) as ic:
                    with cdlt.loop(KH) as kh:
                        with cdlt.loop(KW) as kw:
                            with cdlt.loop(OH) as y:
                                with cdlt.loop(OW) as x:
                                    cdlt.transfer(weight, ["DRAM", "WBUF"])
                                    cdlt.transfer(data, ["DRAM", "IBUF"])
                                    cdlt.transfer(out, ["DRAM", "OBUF"])
                                    out.set_write_destination("OBUF")
                                    cdlt.compute("MVMUL", [data[n, ic, y*stride + kh, x*stride + kw], weight[oc, ic, kh, kw], out[n, oc, y, x]], [out[n, oc, y, x]], target="pe_array")
                                    cdlt.transfer(out, ["OBUF", "DRAM"])

        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "BBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "systolic_array")

    cdlt = add_conv_constraints(hag, cdlt)
    return cdlt



def conv2d_bias(hag: ArchitectureNode):
    # TODO: Need to figure out how to change the memory layout

    required_params = {}

    with CodeletTemplate("conv_bias") as cdlt:
        stride = cdlt.dummy_op("stride", cdlt.node.stride)
        pad = cdlt.dummy_op("pad", cdlt.node.pad)
        OC = cdlt.dummy_op("OC", cdlt.node.outputs[0].shape[1])
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        IC = cdlt.dummy_op("IC", cdlt.node.inputs[0].shape[1])
        KH = cdlt.dummy_op("KH", cdlt.node.inputs[1].shape[2])
        KW = cdlt.dummy_op("KW", cdlt.node.inputs[1].shape[3])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])
        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, IC, IH, IW], default_dtype=OP_DTYPES[0])
        weight = cdlt.create_operand_template("weight", OP_DTYPES, [OC, IC, KH, KW], default_dtype=OP_DTYPES[0])
        bias = cdlt.create_operand_template("bias", OP_DTYPES, [OC], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([data, weight, bias])
        cdlt.set_outputs([out])

        cdlt.configure("start", "systolic_array")
        cdlt.configure("start", "WBUF")
        cdlt.configure("start", "BBUF")
        cdlt.configure("start", "IBUF")
        cdlt.configure("start", "OBUF")
        with cdlt.loop(OC) as oc:
            with cdlt.loop(N) as n:
                with cdlt.loop(IC) as ic:
                    with cdlt.loop(KH) as kh:
                        with cdlt.loop(KW) as kw:
                            with cdlt.loop(OH) as y:
                                with cdlt.loop(OW) as x:
                                    cdlt.transfer(weight, ["DRAM", "WBUF"])
                                    cdlt.transfer(bias, ["DRAM", "BBUF"])
                                    cdlt.transfer(data, ["DRAM", "IBUF"])
                                    cdlt.transfer(out, ["DRAM", "OBUF"])
                                    out.set_write_destination("OBUF")
                                    cdlt.compute("MVMUL", [data[n, ic, y*stride + kh, x*stride + kw], weight[oc, ic, kh, kw], bias[oc], out[n, oc, y, x]], [out[n, oc, y, x]], target="pe_array")
                                    cdlt.transfer(out, ["OBUF", "DRAM"])

        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "BBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "systolic_array")

    cdlt = add_conv_constraints(hag, cdlt)

    return cdlt


SA_CDLTS = {
    "conv_bias": conv2d_bias,
    "conv": conv2d,
    "gemm": gemm,
    'gemm_no_bias': gemm_no_bias,
    "matmul": matmul,
}