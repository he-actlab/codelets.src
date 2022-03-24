from codelets.adl.graph import ArchitectureNode
from codelets.templates.codelet_template import CodeletTemplate
from examples.genesys import OP_DTYPES, ASIC_CONFIG


def conv_relu(hag: ArchitectureNode):
    with CodeletTemplate("conv_bias_relu") as cdlt:
        stride = cdlt.dummy_op("stride", cdlt.node.stride)
        pad = cdlt.dummy_op("pad", cdlt.node.pad)
        OC = cdlt.dummy_op("OC", cdlt.node.outputs[0].shape[1])
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])
        IC = cdlt.dummy_op("IC", cdlt.node.inputs[0].shape[1])
        KH = cdlt.dummy_op("KH", cdlt.node.inputs[1].shape[2])
        KW = cdlt.dummy_op("KW", cdlt.node.inputs[1].shape[3])

        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, IC, IH, IW], default_dtype=OP_DTYPES[0])
        weight = cdlt.create_operand_template("weight", OP_DTYPES, [OC, IC, KH, KW], default_dtype=OP_DTYPES[0])
        bias = cdlt.create_operand_template("bias", OP_DTYPES, [OC], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([data, weight, bias])
        cdlt.set_outputs([out])

        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        param = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        conv_out = cdlt.create_operand_template("conv_out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])
        cdlt.add_temp_operand(conv_out)
        # SA Config
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
                                    cdlt.transfer(data[n, ic, y * stride + kh, x * stride + kw], ["DRAM", "IBUF"])
                                    cdlt.transfer(conv_out[n, oc, y, x], ["DRAM", "OBUF"])
                                    conv_out.set_write_destination("OBUF")
                                    cdlt.compute("MVMUL", [data, weight, bias], [conv_out], target="pe_array")
        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "BBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "systolic_array")

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=16, index=0)

        with cdlt.loop(OC) as oc:
            with cdlt.loop(N) as n:
                with cdlt.loop(OH) as y:
                    with cdlt.loop(OW) as x:
                        out.set_write_destination("VMEM2")
                        cdlt.compute("RELU", [conv_out, param], [out], target="SIMD")
                        cdlt.transfer(out[n, oc, y, x], ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")

    sys_array_dims = hag.get_subgraph_node("pe_array").dimensions

    wbuf_elements = hag.get_subgraph_node("WBUF").addressable_elements
    obuf_elements = hag.get_subgraph_node("OBUF").addressable_elements
    wbuf_index_size = f"sizes['KH']*sizes['KW']*sizes['IC']*sizes['OC']"
    obuf_index_size = f"sizes['N']*sizes['OH']*sizes['OW']*sizes['OC']"
    gt_one_tiles = f"np.prod(list(splits.values())) > 1"

    ic_tiling = f"(splits['IC'] == 1)"

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


def conv_relu_max_pool(hag: ArchitectureNode):
    # with CodeletTemplate("conv_bias_relu") as cdlt:
    with CodeletTemplate("conv_bias_relu_max_pool") as cdlt:
        OC = cdlt.dummy_op("OC", cdlt.node.outputs[0].shape[1])
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])
        IC = cdlt.dummy_op("IC", cdlt.node.inputs[0].shape[1])
        KH = cdlt.dummy_op("KH", cdlt.node.inputs[1].shape[2])
        KW = cdlt.dummy_op("KW", cdlt.node.inputs[1].shape[3])
        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, IC, IH, IW], default_dtype=OP_DTYPES[0])
        weight = cdlt.create_operand_template("weight", OP_DTYPES, [OC, IC, KH, KW], default_dtype=OP_DTYPES[0])
        bias = cdlt.create_operand_template("bias", OP_DTYPES, [OC], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([data, weight, bias])
        cdlt.set_outputs([out])

        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        param = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        conv_out = cdlt.create_operand_template("conv_out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])
        cdlt.add_temp_operand(conv_out)
        # SA Config
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
                                    cdlt.transfer(data[n, ic, y * stride + kh, x * stride + kw], ["DRAM", "IBUF"])
                                    cdlt.transfer(conv_out[n, oc, y, x], ["DRAM", "OBUF"])
                                    conv_out.set_write_destination("OBUF")
                                    cdlt.compute("MVMUL", [data, weight, bias], [conv_out], target="pe_array")
        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "BBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "systolic_array")
        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=16, index=0)

        with cdlt.loop(OC) as oc:
            with cdlt.loop(N) as n:
                with cdlt.loop(OH) as y:
                    with cdlt.loop(OW) as x:
                        out.set_write_destination("VMEM2")
                        cdlt.compute("RELU", [conv_out, param], [out], target="SIMD")
                        cdlt.transfer(out[n, oc, y, x], ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")

    sys_array_dims = hag.get_subgraph_node("pe_array").dimensions

    wbuf_elements = hag.get_subgraph_node("WBUF").addressable_elements
    obuf_elements = hag.get_subgraph_node("OBUF").addressable_elements
    wbuf_index_size = f"sizes['KH']*sizes['KW']*sizes['IC']*sizes['OC']"
    obuf_index_size = f"sizes['N']*sizes['OH']*sizes['OW']*sizes['OC']"
    gt_one_tiles = f"np.prod(list(splits.values())) > 1"

    ic_tiling = f"(splits['IC'] == 1)"

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


def conv_add_relu(hag: ArchitectureNode):
    # MP padding:
    # iw1' = ((ow_conv + 2*p_mp) - 1) * stride + kw_conv
    # P1_update = (iw' - iw1)/2
    with CodeletTemplate("conv_bias_elem_add_relu") as cdlt:
        stride = cdlt.dummy_op("stride", cdlt.node.stride)
        pad = cdlt.dummy_op("pad", cdlt.node.pad)
        OC = cdlt.dummy_op("OC", cdlt.node.outputs[0].shape[1])
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        OH = cdlt.dummy_op("OH", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.outputs[0].shape[3])
        IC = cdlt.dummy_op("IC", cdlt.node.inputs[0].shape[1])
        KH = cdlt.dummy_op("KH", cdlt.node.inputs[1].shape[2])
        KW = cdlt.dummy_op("KW", cdlt.node.inputs[1].shape[3])

        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, IC, IH, IW], default_dtype=OP_DTYPES[0])
        weight = cdlt.create_operand_template("weight", OP_DTYPES, [OC, IC, KH, KW], default_dtype=OP_DTYPES[0])
        bias = cdlt.create_operand_template("bias", OP_DTYPES, [OC], default_dtype=OP_DTYPES[2])
        add_lhs = cdlt.create_operand_template("add_lhs", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([data, weight, bias, add_lhs])
        cdlt.set_outputs([out])

        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        param = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        conv_out = cdlt.create_operand_template("conv_out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])
        add_out = cdlt.create_operand_template("add_out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])
        add_out.start_location = "VMEM2"


        cdlt.add_temp_operand(conv_out)
        cdlt.add_temp_operand(add_out)
        # SA Config
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
                                    cdlt.transfer(data[n, ic, y * stride + kh, x * stride + kw], ["DRAM", "IBUF"])
                                    cdlt.transfer(conv_out[n, oc, y, x], ["DRAM", "OBUF"])
                                    conv_out.set_write_destination("OBUF")
                                    cdlt.compute("MVMUL", [data, weight, bias], [conv_out], target="pe_array")
        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "BBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "systolic_array")

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=16, index=0)

        with cdlt.loop(OC) as oc:
            with cdlt.loop(N) as n:
                with cdlt.loop(OH) as y:
                    with cdlt.loop(OW) as x:
                        cdlt.transfer(add_lhs[n, oc, y, x], ["DRAM", "VMEM1"])
                        add_out.set_write_destination("VMEM2")
                        cdlt.compute("ADD", [add_lhs, conv_out], [add_out], target="SIMD")
                        out.set_write_destination("VMEM1")
                        cdlt.compute("RELU", [add_out, param], [out], target="SIMD")
                        cdlt.transfer(out[n, oc, y, x], ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")

    sys_array_dims = hag.get_subgraph_node("pe_array").dimensions

    wbuf_elements = hag.get_subgraph_node("WBUF").addressable_elements
    obuf_elements = hag.get_subgraph_node("OBUF").addressable_elements
    wbuf_index_size = f"sizes['KH']*sizes['KW']*sizes['IC']*sizes['OC']"
    obuf_index_size = f"sizes['N']*sizes['OH']*sizes['OW']*sizes['OC']"
    gt_one_tiles = f"np.prod(list(splits.values())) > 1"

    ic_tiling = f"(splits['IC'] == 1)"

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


def conv_clip_dw_conv(hag: ArchitectureNode):
    # MP padding:
    # iw1' = ((ow_conv + 2*p_mp) - 1) * stride + kw_conv
    # P1_update = (iw' - iw1)/2
    with CodeletTemplate("conv_bias_elem_clip_depthwise_conv") as cdlt:
        stride = cdlt.dummy_op("stride", cdlt.node.stride)
        pad = cdlt.dummy_op("pad", cdlt.node.pad)
        OC = cdlt.dummy_op("OC", cdlt.node.outputs[0].shape[1])
        ONE = cdlt.dummy_op("ONE", cdlt.node.inputs[3].shape[1])
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        OH = cdlt.dummy_op("OH", cdlt.node.conv_output.shape[2])
        OW = cdlt.dummy_op("OW", cdlt.node.conv_output.shape[3])

        OH1 = cdlt.dummy_op("OH1", cdlt.node.outputs[0].shape[2])
        OW1 = cdlt.dummy_op("OW1", cdlt.node.outputs[0].shape[3])

        IC = cdlt.dummy_op("IC", cdlt.node.inputs[0].shape[1])
        KH = cdlt.dummy_op("KH", cdlt.node.inputs[1].shape[2])
        KW = cdlt.dummy_op("KW", cdlt.node.inputs[1].shape[3])

        KH1 = cdlt.dummy_op("KH1", cdlt.node.inputs[3].shape[2])
        KW1 = cdlt.dummy_op("KW1", cdlt.node.inputs[3].shape[3])

        IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
        IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, IC, IH, IW], default_dtype=OP_DTYPES[0])
        weight = cdlt.create_operand_template("weight", OP_DTYPES, [OC, IC, KH, KW], default_dtype=OP_DTYPES[0])
        bias = cdlt.create_operand_template("bias", OP_DTYPES, [OC], default_dtype=OP_DTYPES[2])
        dw_conv_weights = cdlt.create_operand_template("dw_conv_weights", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])

        cdlt.set_inputs([data, weight, bias, dw_conv_weights])
        cdlt.set_outputs([out])

        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        param = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        conv_out = cdlt.create_operand_template("conv_out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])
        clip_out = cdlt.create_operand_template("clip_out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])
        clip_out.start_location = "VMEM2"


        cdlt.add_temp_operand(conv_out)
        cdlt.add_temp_operand(clip_out)
        # SA Config
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
                                    cdlt.transfer(data[n, ic, y * stride + kh, x * stride + kw], ["DRAM", "IBUF"])
                                    cdlt.transfer(conv_out[n, oc, y, x], ["DRAM", "OBUF"])
                                    conv_out.set_write_destination("OBUF")
                                    cdlt.compute("MVMUL", [data, weight, bias], [conv_out], target="pe_array")
        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "BBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "systolic_array")

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=16, index=0)

        with cdlt.loop(ONE) as one:
            with cdlt.loop(N) as n:
                with cdlt.loop(OC) as c:
                    with cdlt.loop(OH1) as y:
                        with cdlt.loop(OW1) as x:
                            with cdlt.loop(KH1) as kh:
                                with cdlt.loop(KW1) as kw:
                                    cdlt.transfer(weight[c, one, kh, kw], ["DRAM", "VMEM1"])
                                    cdlt.transfer(out[n, c, y, x], ["DRAM", "VMEM2"])
                                    out.set_write_destination("VMEM2")
                                    cdlt.compute("MACC", [clip_out, weight, out], [out], target="SIMD")
                                    cdlt.transfer(out[n, c, y, x], ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")

    sys_array_dims = hag.get_subgraph_node("pe_array").dimensions

    wbuf_elements = hag.get_subgraph_node("WBUF").addressable_elements
    obuf_elements = hag.get_subgraph_node("OBUF").addressable_elements
    wbuf_index_size = f"sizes['KH']*sizes['KW']*sizes['IC']*sizes['OC']"
    obuf_index_size = f"sizes['N']*sizes['OH']*sizes['OW']*sizes['OC']"
    gt_one_tiles = f"np.prod(list(splits.values())) > 1"

    ic_tiling = f"(splits['IC'] == 1)"

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


FUSION_CODELETS = {
    "conv_bias_relu": conv_relu,
    "conv_bias_elem_add_relu": conv_add_relu
}

FUSION_OP_INFO = {
    'conv_bias_relu': {
        'cdlt': conv_relu,
        'seq': ['Conv', 'Relu']
    },
    'conv_bias_elem_add_relu': {
        'cdlt': conv_add_relu,
        'seq': ['Conv', 'Add', 'Relu'],

    },
    'single_layer_info':
        {
            'Conv' : {'inputs': 3, 'outputs': 1},
            'Relu' : {'inputs': 1, 'outputs': 1},
            'Add' : {'inputs': 2, 'outputs': 1}
        }
}