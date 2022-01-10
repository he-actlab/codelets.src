from codelets.templates.codelet_template import CodeletTemplate
from codelets.adl.flex_param import FlexParam
from codelets.adl.graph import ArchitectureNode
from . import OP_DTYPES, ASIC_CONFIG

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
        # with cdlt.loop(M) as m:
        #     with cdlt.loop(N) as n:
        #         with cdlt.loop(P) as p:
                    cdlt.transfer(data[m, n], ["DRAM", "IBUF"])
                    cdlt.transfer(weight[n, p], ["DRAM", "WBUF"])
                    cdlt.transfer(bias[p], ["DRAM", "BBUF"])
                    cdlt.transfer(out[m, p], ["DRAM", "OBUF"])
                    out.set_write_destination("OBUF")
                    cdlt.compute("MVMUL", [data, weight, bias], [out], target="pe_array")
                    cdlt.transfer(out[m, p], ["OBUF", "DRAM"])

        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "BBUF")
        cdlt.configure("end", "systolic_array")
    sys_array_dims = hag.get_subgraph_node("pe_array").dimensions
    # cdlt.add_compilation_param("N_hint2", f"size == {sys_array_dims[0]}")
    # cdlt.add_compilation_param("P_hint2", f"size == {sys_array_dims[1]}")

    wbuf_elements = hag.get_subgraph_node("WBUF").addressable_elements
    obuf_elements = hag.get_subgraph_node("OBUF").addressable_elements
    wbuf_index_size = f"sizes['N']*sizes['P']"
    obuf_index_size = f"sizes['M']*sizes['P']"
    constraint = f"np.prod(list(splits.values())) > 1"

    if not ASIC_CONFIG:
        sg_edge = hag.get_subgraph_edge('DRAM', 'IBUF')
        bandwidth = sg_edge.bandwidth
        fpga_constr = f"{wbuf_index_size} <= {wbuf_elements} and " \
                      f"{obuf_index_size} <= {obuf_elements} and " \
                      f"sizes['N']*{OP_DTYPES[0].bits()} % {bandwidth} == 0"
        constraint = f"{constraint} and {fpga_constr}"
    cdlt.add_compilation_param("LEVEL1_hint", constraint)

    ## DRAM to buffers
    cdlt.add_compilation_param("N_hint1", f"size % {sys_array_dims[0]} == 0")
    cdlt.add_compilation_param("P_hint1", f"size % {sys_array_dims[1]} == 0")

    ## Buffer to systolic array
    cdlt.add_compilation_param("N_hint0", f"size % {sys_array_dims[0]} == 0")
    cdlt.add_compilation_param("P_hint0", f"size % {sys_array_dims[1]} == 0")

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

        # with cdlt.loop(P) as p:
        #     with cdlt.loop(N) as n:
        #         with cdlt.loop(M) as m:
        with cdlt.loop(M) as m:
            with cdlt.loop(N) as n:
                with cdlt.loop(P) as p:
                    cdlt.transfer(data[m, n], ["DRAM", "IBUF"])
                    cdlt.transfer(weight[n, p], ["DRAM", "WBUF"])
                    cdlt.transfer(out[m, p], ["DRAM", "OBUF"])
                    out.set_write_destination("OBUF")
                    cdlt.compute("MVMUL", [data, weight], [out], target="pe_array")
                    cdlt.transfer(out[m, p], ["OBUF", "DRAM"])

        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "systolic_array")
    sys_array_dims = hag.get_subgraph_node("pe_array").dimensions
    cdlt.add_compilation_param("N_hint2", f"size == {sys_array_dims[0]}")
    cdlt.add_compilation_param("P_hint2", f"size == {sys_array_dims[1]}")
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
                    cdlt.transfer(data[m, n], ["DRAM", "IBUF"])
                    cdlt.transfer(weight[n, p], ["DRAM", "WBUF"])
                    cdlt.transfer(out[m, p], ["DRAM", "OBUF"])
                    out.set_write_destination("OBUF")
                    cdlt.compute("MVMUL", [data, weight], [out], target="pe_array")
                    cdlt.transfer(out[m, p], ["OBUF", "DRAM"])

        # TODO: Add store off chip
        # cdlt.configure("end", "WBUF")
        # cdlt.configure("end", "IBUF")
        # cdlt.configure("end", "OBUF")
        # cdlt.configure("end", "systolic_array")
    sys_array_dims = hag.get_subgraph_node("pe_array").dimensions
    cdlt.add_compilation_param("N_hint2", f"size == {sys_array_dims[0]}")
    cdlt.add_compilation_param("P_hint2", f"size == {sys_array_dims[1]}")
    return cdlt

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

        with cdlt.loop(ONE) as one:
            with cdlt.loop(N) as n:
                with cdlt.loop(C) as c:
                    with cdlt.loop(KH) as kh:
                        with cdlt.loop(KW) as kw:
                            with cdlt.loop(OH) as y:
                                with cdlt.loop(OW) as x:

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

def conv2d(hag: ArchitectureNode):
    # TODO: Need to figure out how to change the memory layout
    with CodeletTemplate("conv") as cdlt:
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
        stride = cdlt.dummy_op("stride", cdlt.node.stride)
        pad = cdlt.dummy_op("pad", cdlt.node.pad)
        # OS ->

        with cdlt.loop(OC) as oc:
            with cdlt.loop(N) as n:
                with cdlt.loop(IC) as ic:
                    with cdlt.loop(KH) as kh:
                        with cdlt.loop(KW) as kw:
                            with cdlt.loop(OH) as y:
                                with cdlt.loop(OW) as x:
                                    cdlt.transfer(weight[oc, ic, kh, kw], ["DRAM", "WBUF"])
                                    cdlt.transfer(data[n, ic, y*stride + kh, x*stride + kw], ["DRAM", "IBUF"])
                                    cdlt.transfer(out[n, oc, y, x], ["DRAM", "OBUF"])
                                    out.set_write_destination("OBUF")
                                    cdlt.compute("MVMUL", [data, weight], [out], target="pe_array")
                                    cdlt.transfer(out[n, oc, y, x], ["OBUF", "DRAM"])

        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "BBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "systolic_array")
    sys_array_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("LOOP_TILE_ORDER", ["KH", "KW", "OC", "IC", "N", "OH", "OW"])

    wbuf_elements = hag.get_subgraph_node("WBUF").addressable_elements
    obuf_elements = hag.get_subgraph_node("OBUF").addressable_elements
    wbuf_index_size = f"sizes['KH']*sizes['KW']*sizes['IC']*sizes['OC']"
    obuf_index_size = f"sizes['N']*sizes['OH']*sizes['OW']*sizes['OC']"

    gt_one_tiles = f"np.prod(list(splits.values())) > 1"
    ic_tiling = f"(splits['IC'] == 1 or any([splits['KH'] > 1, splits['KW'] > 1, splits['OH'] > 1, splits['OW'] > 1]))"
    constraint = f"{gt_one_tiles} and {ic_tiling}"
    if not ASIC_CONFIG:
        sg_edge = hag.get_subgraph_edge('DRAM', 'IBUF')
        bandwidth = sg_edge.bandwidth
        ic_hint = f"sizes['IC']*{OP_DTYPES[0].bits()} % {bandwidth} == 0"
        constraint = f"{constraint} and {ic_hint} and {wbuf_index_size} <= {wbuf_elements} and {obuf_index_size} <= {obuf_elements}"

    cdlt.add_compilation_param("LEVEL1_hint", constraint)


    ## DRAM to buffers
    cdlt.add_compilation_param("IC_hint1", f"size % {sys_array_dims[0]} == 0")
    cdlt.add_compilation_param("OC_hint1", f"size % {sys_array_dims[1]} == 0")
    cdlt.add_compilation_param("KH_hint1", f"split == 1")
    cdlt.add_compilation_param("KW_hint1", f"split == 1")

    ## Buffer to systolic array
    cdlt.add_compilation_param("IC_hint0", f"size % {sys_array_dims[0]} == 0")
    cdlt.add_compilation_param("OC_hint0", f"size % {sys_array_dims[1]} == 0")
    cdlt.add_compilation_param("KH_hint0", f"size == 1")
    cdlt.add_compilation_param("KW_hint0", f"size == 1")
    ####

    return cdlt

def conv2d_bias(hag: ArchitectureNode):
    # TODO: Need to figure out how to change the memory layout

    required_params = {}

    with CodeletTemplate("conv_bias") as cdlt:

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
        stride = cdlt.dummy_op("stride", cdlt.node.stride)
        pad = cdlt.dummy_op("pad", cdlt.node.pad)


        with cdlt.loop(OC) as oc:
            with cdlt.loop(N) as n:
                with cdlt.loop(IC) as ic:
                    with cdlt.loop(KH) as kh:
                        with cdlt.loop(KW) as kw:
                            with cdlt.loop(OH) as y:
                                with cdlt.loop(OW) as x:
                                    cdlt.transfer(weight[oc, ic, kh, kw], ["DRAM", "WBUF"])
                                    cdlt.transfer(bias[oc], ["DRAM", "BBUF"])
                                    cdlt.transfer(data[n, ic, y*stride + kh, x*stride + kw], ["DRAM", "IBUF"])
                                    cdlt.transfer(out[n, oc, y, x], ["DRAM", "OBUF"])
                                    out.set_write_destination("OBUF")
                                    cdlt.compute("MVMUL", [data, weight, bias], [out], target="pe_array")
                                    # cdlt.compute("MVMUL", [data[n, ic, y*"stride" + kh, x*"stride" + kw], weight[oc, ic, kh, kw], bias[oc]], [out[n, oc, y, x]], target="pe_array")
                                    cdlt.transfer(out[n, oc, y, x], ["OBUF", "DRAM"])

        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "BBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "systolic_array")
    sys_array_dims = hag.get_subgraph_node("pe_array").dimensions
    cdlt.add_compilation_param("LOOP_TILE_ORDER", ["KH", "KW", "OC", "IC", "N", "OH", "OW"])

    wbuf_elements = hag.get_subgraph_node("WBUF").addressable_elements
    obuf_elements = hag.get_subgraph_node("OBUF").addressable_elements
    wbuf_index_size = f"sizes['KH']*sizes['KW']*sizes['IC']*sizes['OC']"
    obuf_index_size = f"sizes['N']*sizes['OH']*sizes['OW']*sizes['OC']"
    gt_one_tiles = f"np.prod(list(splits.values())) > 1"

    ic_tiling = f"(splits['IC'] == 1 or any([splits['KH'] > 1, splits['KW'] > 1, splits['OH'] > 1, splits['OW'] > 1]))"
    t = f""
    # t = f" and splits['OC'] == 1 and splits['IC'] > 1"
    # t = f" and splits['OC'] > 1 and splits['IC'] == 1"
    # t = f" and splits['OC'] > 1 and splits['IC'] > 1"
    constraint = f"{gt_one_tiles} and {ic_tiling}{t}"
    if not ASIC_CONFIG:
        sg_edge = hag.get_subgraph_edge('DRAM', 'IBUF')
        bandwidth = sg_edge.bandwidth
        ic_hint = f"sizes['IC']*{OP_DTYPES[0].bits()} % {bandwidth} == 0"

        constraint = f"{constraint} and {ic_hint} and {wbuf_index_size} <= {wbuf_elements} and " \
                     f"{obuf_index_size} <= {obuf_elements}"
    cdlt.add_compilation_param("LEVEL1_hint", constraint)
    # bw/sys array width*bits

    ## DRAM to buffers
    cdlt.add_compilation_param("IC_hint1", f"size % {sys_array_dims[0]} == 0")
    cdlt.add_compilation_param("OC_hint1", f"size % {sys_array_dims[1]} == 0")
    cdlt.add_compilation_param("KH_hint1", f"split == 1")
    cdlt.add_compilation_param("KW_hint1", f"split == 1")

    # Buffer to systolic array
    cdlt.add_compilation_param("IC_hint0", f"size % {sys_array_dims[0]} == 0")
    cdlt.add_compilation_param("OC_hint0", f"size % {sys_array_dims[1]} == 0")
    cdlt.add_compilation_param("KH_hint0", f"size == 1")
    cdlt.add_compilation_param("KW_hint0", f"size == 1")


    return cdlt

def elem_add(hag: ArchitectureNode):

    with CodeletTemplate("elem_add") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        op2 = cdlt.create_operand_template("op2", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("add_out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([op1, op2])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(op2[n, c, h, w], ["DRAM", "VMEM2"])
                        out.set_write_destination("VMEM1")
                        # out.set_write_destination("OBUF")
                        cdlt.compute("ADD", [op1, op2], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM1", "DRAM"])
                        # cdlt.transfer(out[n, c, h, w], ["OBUF", "DRAM"])
        cdlt.configure("end", "SIMD")

        simd_dims = hag.get_subgraph_node("pe_array").dimensions

        cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt




def elem_mul(hag: ArchitectureNode):

    with CodeletTemplate("elem_mul") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        op2 = cdlt.create_operand_template("op2", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("mul_out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([op1, op2])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(op2[n, c, h, w], ["DRAM", "VMEM2"])
                        out.set_write_destination("VMEM1")
                        cdlt.compute("MUL", [op1, op2], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
        simd_dims = hag.get_subgraph_node("pe_array").dimensions

        cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt

def elem_lt(hag: ArchitectureNode):

    with CodeletTemplate("elem_lt") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        op2 = cdlt.create_operand_template("op2", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("lt_out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([op1, op2])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(op2[n, c, h, w], ["DRAM", "VMEM2"])
                        out.set_write_destination("VMEM1")
                        cdlt.compute("LT", [op1, op2], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
        simd_dims = hag.get_subgraph_node("pe_array").dimensions

        cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt

def elem_eq(hag: ArchitectureNode):

    with CodeletTemplate("elem_eq") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        op2 = cdlt.create_operand_template("op2", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("eq_out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([op1, op2])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(op2[n, c, h, w], ["DRAM", "VMEM2"])
                        out.set_write_destination("VMEM1")
                        cdlt.compute("EQ", [op1, op2], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
        simd_dims = hag.get_subgraph_node("pe_array").dimensions
        cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt

def elem_sub(hag: ArchitectureNode):

    with CodeletTemplate("elem_sub") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        op2 = cdlt.create_operand_template("op2", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("sub_out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([op1, op2])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(op2[n, c, h, w], ["DRAM", "VMEM2"])
                        out.set_write_destination("VMEM1")
                        # out.set_write_destination("OBUF")
                        cdlt.compute("SUB", [op1, op2], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM1", "DRAM"])
                        # cdlt.transfer(out[n, c, h, w], ["OBUF", "DRAM"])
        cdlt.configure("end", "SIMD")
        simd_dims = hag.get_subgraph_node("pe_array").dimensions

        cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt


def elem_div(hag: ArchitectureNode):

    with CodeletTemplate("elem_div") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        op2 = cdlt.create_operand_template("op2", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("sub_out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([op1, op2])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        cdlt.transfer(op2[n, c, h, w], ["DRAM", "VMEM2"])
                        out.set_write_destination("VMEM1")
                        # out.set_write_destination("OBUF")
                        cdlt.compute("DIV", [op1, op2], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM1", "DRAM"])
                        # cdlt.transfer(out[n, c, h, w], ["OBUF", "DRAM"])
        cdlt.configure("end", "SIMD")
        simd_dims = hag.get_subgraph_node("pe_array").dimensions

        cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt


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

def coarse_flatten(hag: ArchitectureNode):

    with CodeletTemplate("coarse_flatten") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        cdlt.configure("end", "SIMD")
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

def tensor_reshape(hag: ArchitectureNode):

    with CodeletTemplate("tensor_reshape") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        cdlt.configure("end", "SIMD")
    return cdlt

def tensor_pad(hag: ArchitectureNode):

    with CodeletTemplate("tensor_pad") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        cdlt.configure("end", "SIMD")
    return cdlt

def tensor_flip(hag: ArchitectureNode):

    with CodeletTemplate("tensor_flip") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        cdlt.configure("end", "SIMD")

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
                cdlt.transfer(data[n, c], ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM1")
                cdlt.compute("ADD", [data, data], [out], target="SIMD")
                cdlt.transfer(out[c], ["VMEM1", "DRAM"])
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

def relu(hag):

    with CodeletTemplate("relu") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])

        out = cdlt.create_operand_template("out_relu", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        # fix C dim to array size
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:

                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM2")
                        cdlt.compute("RELU", [op1], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    # cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")

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

                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM2")
                        cdlt.compute("RELU", [op1], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")

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
                cdlt.transfer(op1[n, c], ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM1")
                cdlt.compute("RELU", [op1], [out], target="SIMD")
                cdlt.transfer(out[n, c], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt

def sigmoid(hag):

    with CodeletTemplate("elem_sigmoid") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])

        out = cdlt.create_operand_template("out_relu", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        # fix C dim to array size
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:

                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM2")
                        cdlt.compute("SIGMOID", [op1], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")

    return cdlt

def exp(hag):

    with CodeletTemplate("exp") as cdlt:
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

                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM2")
                        cdlt.compute("EXP", [op1], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")

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

                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM2")
                        cdlt.compute("LEAKY_RELU", [op1], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")

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
                cdlt.transfer(op1[n, c], ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM1")
                cdlt.compute("RELU", [op1], [out], target="SIMD")
                cdlt.transfer(out[n, c], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
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
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM1")
                        cdlt.compute("TANH", [op1], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt.add_compilation_param("LOOP_TILE_ORDER", ["N", "C", "H", "W"])

    return cdlt

def elem_tanh2d(hag: ArchitectureNode):

    with CodeletTemplate("elem_tanh2d") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out_tanh", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                cdlt.transfer(op1[n, c], ["DRAM", "VMEM1"])
                out.set_write_destination("VMEM1")
                cdlt.compute("TANH", [op1], [out], target="SIMD")
                cdlt.transfer(out[n, c], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt.add_compilation_param("LOOP_TILE_ORDER", ["N", "C"])
    simd_dims = hag.get_subgraph_node("pe_array").dimensions

    cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
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

# TODO: Implement valid operation sequence
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

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=0, index=0)
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

# TODO: Implement valid operation sequence
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


    # cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
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
        temp_res = cdlt.create_temp_operand([N, C, H, W], "VMEM2")

        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        # cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        # out.set_write_destination("VMEM2")
                        # cdlt.compute("MAX", [op1, max_op], [out], target="SIMD")
                        # cdlt.compute("MIN", [out, min_op], [out], target="SIMD")
                        # cdlt.transfer(out[n, c, h, w], ["VMEM2", "DRAM"])

                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        temp_res.set_write_destination("VMEM2")
                        cdlt.compute("MAX", [op1, max_op], [temp_res], target="SIMD")
                        out.set_write_destination("VMEM2")
                        cdlt.compute("MIN", [temp_res, min_op], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM2", "DRAM"])

        cdlt.configure("end", "SIMD")
        simd_dims = hag.get_subgraph_node("pe_array").dimensions

        cdlt.add_compilation_param("C_hint2", f"size == {simd_dims[0]}")
    return cdlt

GENESYS_CODELETS = {
    "max_pool": maxpool2d,
    "avg_pool": averagepool2d,
    "global_avg_pool": global_avg_pool,
    "coarse_flatten": coarse_flatten,
    "coarse_flatten2d": coarse_flatten2d,
    "conv_bias": conv2d_bias,
    "conv": conv2d,
    "depthwise_conv": depthwise_conv,
    "gemm": gemm,
    "matmul": matmul,
    "elem_add": elem_add,
    "elem_sub": elem_sub,
    "elem_div": elem_div,
    "elem_tanh": elem_tanh,
    "elem_mul": elem_mul,
    "elem_lt": elem_lt,
    "elem_clip": clip,
    "exp": exp,
    "sigmoid": sigmoid,
    "elem_tanh2d": elem_tanh2d,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "relu2d": relu2d,
    "batch_norm": batch_norm,
    "mean_var": mean_var,
    "batchnorm_grad": batchnorm_grad,
    "cross_entropy_loss": cross_entropy_loss,
    "cross_entropy_loss_grad": cross_entropy_loss_grad,
    "flatten_grad": flatten_grad,
    "reduce_sum": reduce_sum,
    "sgd1d": sgd1d,
    "sgd2d": sgd2d,
    "sgd3d": sgd3d,
    "sgd4d": sgd4d,
    'elem_tanh_grad': elem_tanh_grad,
    'elem_cast': elem_cast,
    'elem_cast2d': elem_cast2d,
    'elem_tanh_grad2d': elem_tanh_grad2d,
    'average_pool_grad': average_pool_grad,
    'max_pool_grad': max_pool_grad,
    'gemm_no_bias': gemm_no_bias,
    'relu_grad': relu_grad,
    'relu_grad2d': relu_grad2d,
    'global_average_pool_grad': global_average_pool_grad,
    'tensor_transpose': tensor_transpose,
    'tensor_reshape': tensor_reshape,
    'tensor_flip': tensor_flip,
    'tensor_pad': tensor_pad,
}
