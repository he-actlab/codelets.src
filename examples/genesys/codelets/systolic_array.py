from codelets.adl.graph import ArchitectureNode
from codelets.templates.codelet_template import CodeletTemplate
from examples.genesys import OP_DTYPES, ASIC_CONFIG


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
                                    cdlt.transfer(weight[oc, ic, kh, kw], ["DRAM", "WBUF"])
                                    cdlt.transfer(bias[oc], ["DRAM", "BBUF"])
                                    cdlt.transfer(data[n, ic, y*stride + kh, x*stride + kw], ["DRAM", "IBUF"])
                                    cdlt.transfer(out[n, oc, y, x], ["DRAM", "OBUF"])
                                    out.set_write_destination("OBUF")
                                    cdlt.compute("MVMUL", [data, weight, bias], [out], target="pe_array")
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


SA_CDLTS = {
    "conv_bias": conv2d_bias,
    "conv": conv2d,
    "gemm": gemm,
    'gemm_no_bias': gemm_no_bias,
    "matmul": matmul,
}