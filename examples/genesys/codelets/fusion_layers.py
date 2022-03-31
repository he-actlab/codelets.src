from codelets.adl.graph import ArchitectureNode
from codelets.templates.codelet_template import CodeletTemplate
from examples.genesys import OP_DTYPES, ASIC_CONFIG, FXP_CONFIGS, QUANT_SCALE, SIGN_SHIFT
from . import add_conv_constraints, range_from_cfg, \
    add_simd_constraint, create_immediate_with_operand, add_scale_op

def create_systolic_args(cdlt):
    params = {}
    stride = cdlt.dummy_op("stride", cdlt.node.stride)
    pad = cdlt.dummy_op("pad", cdlt.node.pad)
    OC = cdlt.dummy_op("OC", cdlt.node.conv_output.shape[1])
    N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
    OH = cdlt.dummy_op("OH", cdlt.node.conv_output.shape[2])
    OW = cdlt.dummy_op("OW", cdlt.node.conv_output.shape[3])
    IC = cdlt.dummy_op("IC", cdlt.node.inputs[0].shape[1])
    KH = cdlt.dummy_op("KH", cdlt.node.inputs[1].shape[2])
    KW = cdlt.dummy_op("KW", cdlt.node.inputs[1].shape[3])
    IH = cdlt.dummy_op("IH", cdlt.node.inputs[0].shape[2])
    IW = cdlt.dummy_op("IW", cdlt.node.inputs[0].shape[3])

    params['stride'] = stride
    params['pad'] = pad
    params['OC'] = OC
    params['N'] = N
    params['IC'] = IC
    params['OH'] = OH
    params['OW'] = OW
    params['KH'] = KH
    params['KW'] = KW
    params['IH'] = IH
    params['IW'] = IW

    data = cdlt.create_operand_template("data", OP_DTYPES, [N, IC, IH, IW], default_dtype=OP_DTYPES[0])
    weight = cdlt.create_operand_template("weight", OP_DTYPES, [OC, IC, KH, KW], default_dtype=OP_DTYPES[0])
    bias = cdlt.create_operand_template("bias", OP_DTYPES, [OC], default_dtype=OP_DTYPES[2])
    cdlt.set_inputs([data, weight, bias])
    return cdlt, params


def create_systolic_func(cdlt, params):
    data = cdlt.inputs[0]
    weight = cdlt.inputs[1]
    bias = cdlt.inputs[2]
    stride = params['stride']
    pad = params['pad']
    OC = params['OC']
    N = params['N']
    IC = params['IC']
    OH = params['OH']
    OW = params['OW']
    KH = params['KH']
    KW = params['KW']
    IH = params['IH']
    IW = params['IW']

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
                                cdlt.transfer(weight, ["DRAM", "WBUF"])
                                cdlt.transfer(bias, ["DRAM", "BBUF"])
                                cdlt.transfer(data, ["DRAM", "IBUF"])
                                cdlt.transfer(conv_out, ["DRAM", "OBUF"])
                                conv_out.set_write_destination("OBUF")
                                cdlt.compute("MVMUL",
                                             [data[n, ic, y * stride + kh, x * stride + kw], weight[oc, ic, kh, kw],
                                              bias[oc], conv_out[n, oc, y, x]], [conv_out[n, oc, y, x]],
                                             target="pe_array")
    # TODO: Add store off chip
    cdlt.configure("end", "WBUF")
    cdlt.configure("end", "BBUF")
    cdlt.configure("end", "IBUF")
    cdlt.configure("end", "OBUF")
    cdlt.configure("end", "systolic_array")
    return cdlt, conv_out

def conv_relu(hag: ArchitectureNode):
    with CodeletTemplate("conv_bias_relu") as cdlt:

        cdlt, params = create_systolic_args(cdlt)
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        param = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        cdlt, conv_out = create_systolic_func(cdlt, params)

        OC, N, OH, OW = params['OC'], params['N'], params['OH'], params['OW']
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])

        cdlt.configure("start", "SIMD")
        m0 = create_immediate_with_operand(cdlt, QUANT_SCALE, simd_size=SIMD_SIZE)
        nshift = create_immediate_with_operand(cdlt, QUANT_SCALE, simd_size=SIMD_SIZE)
        cdlt.configure("start", "IMM", immediate_value=16, index=0)
        with cdlt.loop(OC) as oc:
            with cdlt.loop(N) as n:
                with cdlt.loop(OH) as y:
                    with cdlt.loop(OW) as x:
                        out.set_write_destination("VMEM2")
                        indices = (n, oc, y, x)
                        add_scale_op(cdlt, conv_out, out, m0, nshift, indices)
                        # cdlt.compute("RELU", [conv_out[n, oc, y, x], param], [out[n, oc, y, x]], target="SIMD")
                        cdlt.compute("RELU", [out[n, oc, y, x], param], [out[n, oc, y, x]], target="SIMD")
                        cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")

    cdlt = add_conv_constraints(hag, cdlt, is_fusion=True)

    return cdlt


def conv_leaky_relu(hag: ArchitectureNode):
    with CodeletTemplate("conv_bias_leaky_relu") as cdlt:

        cdlt, params = create_systolic_args(cdlt)
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        alpha = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        cdlt, conv_out = create_systolic_func(cdlt, params)

        OC, N, OH, OW = params['OC'], params['N'], params['OH'], params['OW']
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])

        cdlt.configure("start", "SIMD")
        alphaval = cdlt.dummy_op("alpha", cdlt.node.alpha, dtype="FXP32")
        cdlt.configure("start", "IMM", immediate_value=alphaval, index=0)
        with cdlt.loop(OC) as oc:
            with cdlt.loop(N) as n:
                with cdlt.loop(OH) as y:
                    with cdlt.loop(OW) as x:
                        out.set_write_destination("VMEM2")
                        cdlt.compute("LEAKY_RELU", [conv_out[n, oc, y, x], alpha], [out[n, oc, y, x]], target="SIMD")
                        cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")

    cdlt = add_conv_constraints(hag, cdlt, is_fusion=True)

    return cdlt


def conv_relu_max_pool(hag: ArchitectureNode):
    with CodeletTemplate("conv_bias_relu_max_pool") as cdlt:
        # Setup conv arguments
        cdlt, params = create_systolic_args(cdlt)
        # Add parameter for relu
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        param = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        # Create systolic loop nest
        cdlt, conv_out = create_systolic_func(cdlt, params)

        # parameters for max pool
        C, N = params['OC'], params['N']
        OH = cdlt.dummy_op("OH1", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW1", cdlt.node.outputs[0].shape[3])
        KH = cdlt.dummy_op("KH1", cdlt.node.kernel_size[0])
        KW = cdlt.dummy_op("KW1", cdlt.node.kernel_size[1])

        # Create outputs
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, OH, OW], default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])

        # Create temporary storage for relu output
        relu_out = cdlt.create_operand_template("relu_out", OP_DTYPES, [N, C, params['OH'], params['OW']], default_dtype=OP_DTYPES[2])
        relu_out.start_location = "VMEM1"
        cdlt.add_temp_operand(relu_out)


        min_val, _ = range_from_cfg(FXP_CONFIGS[str(OP_DTYPES[2])])
        mp_pad = cdlt.dummy_op("max_pool_pad", cdlt.node.max_pool_pad[0])
        sy = cdlt.dummy_op("sy", cdlt.node.max_pool_stride[0])
        sx = cdlt.dummy_op("sx", cdlt.node.max_pool_stride[1])
        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=min_val, index=0)
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(KH) as kh:
                    with cdlt.loop(KW) as kw:
                        with cdlt.loop(OH) as y:
                            with cdlt.loop(OW) as x:
                                cdlt.transfer(out, ["DRAM", "VMEM2"])
                                out.set_write_destination("VMEM2")
                                relu_out.set_write_destination("VMEM1")

                                cdlt.compute("RELU", [conv_out[n, c, y * sy + kh, x * sx + kw], param],
                                             [relu_out[n, c, y * sy + kh, x * sx + kw]], target="SIMD")
                                cdlt.compute("MAX", [relu_out[n, c, y * sy + kh, x * sx + kw],
                                                     out[n, c, y, x]],
                                             [out[n, c, y, x]], target="SIMD")
                                cdlt.transfer(out, ["VMEM2", "DRAM"])
        cdlt.configure("end", "SIMD")
    cdlt = add_conv_constraints(hag, cdlt, is_fusion=True)
    cdlt = add_simd_constraint(hag, cdlt, "OC")
    return cdlt


def conv_add_relu(hag: ArchitectureNode):
    # MP padding:
    # iw1' = ((ow_conv + 2*p_mp) - 1) * stride + kw_conv
    # P1_update = (iw' - iw1)/2
    # Halo effect
    # constraint =
    with CodeletTemplate("conv_bias_elem_add_relu") as cdlt:
        cdlt, params = create_systolic_args(cdlt)

        # Use initial params to setup subsequent operation details
        OC, N, OH, OW = params['OC'], params['N'], params['OH'], params['OW']
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        param = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        add_lhs = cdlt.create_operand_template("add_lhs", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])
        cdlt.add_input(add_lhs)
        cdlt.set_outputs([out])

        add_out = cdlt.create_operand_template("add_out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])
        add_out.start_location = "VMEM2"
        cdlt.add_temp_operand(add_out)

        # Add the convolution
        cdlt, conv_out = create_systolic_func(cdlt, params)

        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=16, index=0)
        with cdlt.loop(OC) as oc:
            with cdlt.loop(N) as n:
                with cdlt.loop(OH) as y:
                    with cdlt.loop(OW) as x:
                        cdlt.transfer(add_lhs, ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM1")
                        add_out.set_write_destination("VMEM2")
                        cdlt.compute("ADD", [add_lhs[n, oc, y, x], conv_out[n, oc, y, x]], [add_out[n, oc, y, x]], target="SIMD")
                        cdlt.compute("RELU", [add_out[n, oc, y, x], param], [out[n, oc, y, x]], target="SIMD")

                        # cdlt.compute("ADD", [add_lhs[n, oc, y, x], conv_out[n, oc, y, x]], [out[n, oc, y, x]], target="SIMD")
                        # cdlt.compute("RELU", [out[n, oc, y, x], param], [out[n, oc, y, x]], target="SIMD")

                        cdlt.transfer(out, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")

    cdlt = add_conv_constraints(hag, cdlt, is_fusion=True)

    return cdlt


def conv_add_leaky_relu(hag: ArchitectureNode):
    # MP padding:
    # iw1' = ((ow_conv + 2*p_mp) - 1) * stride + kw_conv
    # P1_update = (iw' - iw1)/2
    # Halo effect
    # constraint =
    with CodeletTemplate("conv_bias_elem_add_leaky_relu") as cdlt:
        cdlt, params = create_systolic_args(cdlt)

        # Use initial params to setup subsequent operation details
        OC, N, OH, OW = params['OC'], params['N'], params['OH'], params['OW']
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])
        alpha = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        add_lhs = cdlt.create_operand_template("add_lhs", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])
        cdlt.add_input(add_lhs)
        cdlt.set_outputs([out])

        add_out = cdlt.create_operand_template("add_out", OP_DTYPES, [N, OC, OH, OW], default_dtype=OP_DTYPES[2])
        add_out.start_location = "VMEM2"
        cdlt.add_temp_operand(add_out)

        # Add the convolution
        cdlt, conv_out = create_systolic_func(cdlt, params)

        cdlt.configure("start", "SIMD")
        alphaval = cdlt.dummy_op("alpha", cdlt.node.alpha, dtype="FXP32")
        cdlt.configure("start", "IMM", immediate_value=alphaval, index=0)
        with cdlt.loop(OC) as oc:
            with cdlt.loop(N) as n:
                with cdlt.loop(OH) as y:
                    with cdlt.loop(OW) as x:
                        cdlt.transfer(add_lhs, ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM1")
                        add_out.set_write_destination("VMEM2")
                        cdlt.compute("ADD", [add_lhs[n, oc, y, x], conv_out[n, oc, y, x]], [add_out[n, oc, y, x]], target="SIMD")
                        cdlt.compute("LEAKY_RELU", [add_out[n, oc, y, x], alpha], [out[n, oc, y, x]], target="SIMD")

                        # cdlt.compute("ADD", [add_lhs[n, oc, y, x], conv_out[n, oc, y, x]], [out[n, oc, y, x]], target="SIMD")
                        # cdlt.compute("RELU", [out[n, oc, y, x], param], [out[n, oc, y, x]], target="SIMD")

                        cdlt.transfer(out, ["VMEM1", "DRAM"])
        cdlt.configure("end", "SIMD")

    cdlt = add_conv_constraints(hag, cdlt, is_fusion=True)

    return cdlt

def conv_bias_elem_clip_depthwise_conv_bias(hag: ArchitectureNode):
    # MP padding:
    # iw1' = ((ow_conv + 2*p_mp) - 1) * stride + kw_conv
    # P1_update = (iw' - iw1)/2
    with CodeletTemplate("conv_bias_elem_clip_depthwise_conv_bias") as cdlt:
        # Setup conv arguments
        cdlt, params = create_systolic_args(cdlt)
        # Add parameter for clip
        C, N = params['OC'], params['N']
        ONE = cdlt.dummy_op("ONE", cdlt.node.inputs[3].shape[1])
        OH = cdlt.dummy_op("OH1", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW1", cdlt.node.outputs[0].shape[3])
        KH = cdlt.dummy_op("KH1", cdlt.node.inputs[3].shape[2])
        KW = cdlt.dummy_op("KW1", cdlt.node.inputs[3].shape[3])

        # Add dw conv inputs
        weight = cdlt.create_operand_template("dw_conv_wgt", OP_DTYPES, [C, ONE, KH, KW], default_dtype=OP_DTYPES[2])
        bias = cdlt.create_operand_template("dw_conv_bias", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])
        cdlt.add_input(weight)
        cdlt.add_input(bias)

        s = cdlt.dummy_op("s2", cdlt.node.depthwise_conv_bias_stride)
        pad = cdlt.dummy_op("p2", cdlt.node.depthwise_conv_bias_pad)


        # Create outputs
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, OH, OW], default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])

        # Create temporary storage
        clip_out1 = cdlt.create_operand_template("clip_out1", OP_DTYPES, [N, C, params['OH'], params['OW']], default_dtype=OP_DTYPES[2])
        cdlt.add_temp_operand(clip_out1)

        # clip_out2 = cdlt.create_operand_template("clip_out2", OP_DTYPES, [N, C, params['OH'], params['OW']], default_dtype=OP_DTYPES[2])


        # Setup min/max params
        minval = cdlt.dummy_op("min", cdlt.node.kwargs['minval'], dtype="FXP32")
        maxval = cdlt.dummy_op("max", cdlt.node.kwargs['maxval'], dtype="FXP32")
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        min_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        max_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        cdlt, conv_out = create_systolic_func(cdlt, params)
        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=0, index=0)
        cdlt.configure("start", "IMM", immediate_value=minval, index=1)
        cdlt.configure("start", "IMM", immediate_value=maxval, index=2)
        with cdlt.loop(ONE) as one:
            with cdlt.loop(N) as n:
                with cdlt.loop(C) as c:
                    with cdlt.loop(OH) as y:
                        with cdlt.loop(OW) as x:
                            with cdlt.loop(KH) as kh:
                                with cdlt.loop(KW) as kw:
                                    cdlt.transfer(weight, ["DRAM", "VMEM1"])
                                    cdlt.transfer(bias, ["DRAM", "VMEM2"])
                                    cdlt.transfer(out, ["DRAM", "VMEM1"])
                                    clip_out1.set_write_destination("VMEM2")

                                    cdlt.compute("MAX", [conv_out[n, c, y * s + kh, x * s + kw], max_op],
                                                 [clip_out1[n, c, y * s + kh, x * s + kw]
                                                  ],
                                                 target="SIMD")

                                    cdlt.compute("MIN", [clip_out1[n, c, y * s + kh, x * s + kw], min_op],
                                                 [clip_out1[n, c, y * s + kh, x * s + kw]],
                                                 target="SIMD")

                                    out.set_write_destination("VMEM1")

                                    cdlt.compute("MACC",
                                                 [clip_out1[n, c, y * s + kh, x * s + kw], weight[c, one, kh, kw],
                                                  out[n, c, y, x]], [out[n, c, y, x]],
                                                 target="SIMD")

                                    cdlt.compute("ADD", [out[n, c, y, x], bias[c]], [out[n,c,y,x]], target="SIMD")

                                    cdlt.transfer(out, ["VMEM1", "DRAM"])

        cdlt.configure("end", "SIMD")
    cdlt = add_conv_constraints(hag, cdlt, is_fusion=True)

    cdlt = add_simd_constraint(hag, cdlt, "OC")
    return cdlt


def conv_bias_elem_clip_depthwise_conv_bias_elem_clip(hag: ArchitectureNode):
    # MP padding:
    # iw1' = ((ow_conv + 2*p_mp) - 1) * stride + kw_conv
    # P1_update = (iw' - iw1)/2
    with CodeletTemplate("conv_bias_elem_clip_depthwise_conv_bias_elem_clip") as cdlt:
        # Setup conv arguments
        cdlt, params = create_systolic_args(cdlt)
        # Add parameter for clip
        C, N = params['OC'], params['N']
        ONE = cdlt.dummy_op("ONE", cdlt.node.inputs[3].shape[1])
        OH = cdlt.dummy_op("OH1", cdlt.node.outputs[0].shape[2])
        OW = cdlt.dummy_op("OW1", cdlt.node.outputs[0].shape[3])
        KH = cdlt.dummy_op("KH1", cdlt.node.inputs[3].shape[2])
        KW = cdlt.dummy_op("KW1", cdlt.node.inputs[3].shape[3])

        # Add dw conv inputs
        weight = cdlt.create_operand_template("dw_conv_wgt", OP_DTYPES, [C, ONE, KH, KW], default_dtype=OP_DTYPES[2])
        bias = cdlt.create_operand_template("dw_conv_bias", OP_DTYPES, [C], default_dtype=OP_DTYPES[2])
        cdlt.add_input(weight)
        cdlt.add_input(bias)

        s = cdlt.dummy_op("s2", cdlt.node.depthwise_conv_bias_stride)
        pad = cdlt.dummy_op("p2", cdlt.node.depthwise_conv_bias_pad)


        # Create outputs
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, OH, OW], default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])

        # Create temporary storage
        clip_out1 = cdlt.create_operand_template("clip_out1", OP_DTYPES, [N, C, params['OH'], params['OW']], default_dtype=OP_DTYPES[2])
        cdlt.add_temp_operand(clip_out1)

        # dw_conv_out = cdlt.create_operand_template("dw_conv_out", OP_DTYPES, [N, C, OH, OW], default_dtype=OP_DTYPES[2])
        # cdlt.add_temp_operand(dw_conv_out)

        # clip_out2 = cdlt.create_operand_template("clip_out2", OP_DTYPES, [N, C, params['OH'], params['OW']], default_dtype=OP_DTYPES[2])


        # Setup min/max params
        minval = cdlt.dummy_op("min", cdlt.node.kwargs['minval'], dtype="FXP32")
        maxval = cdlt.dummy_op("max", cdlt.node.kwargs['maxval'], dtype="FXP32")
        SIMD_SIZE = cdlt.dummy_op("SIMD_SIZE", cdlt.hag.all_subgraph_nodes['SIMD'].dimensions[0])

        min_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")
        max_op = cdlt.create_temp_operand([SIMD_SIZE], "IMM")

        cdlt, conv_out = create_systolic_func(cdlt, params)
        cdlt.configure("start", "SIMD")
        cdlt.configure("start", "IMM", immediate_value=0, index=0)
        cdlt.configure("start", "IMM", immediate_value=minval, index=1)
        cdlt.configure("start", "IMM", immediate_value=maxval, index=2)
        with cdlt.loop(ONE) as one:
            with cdlt.loop(N) as n:
                with cdlt.loop(C) as c:
                    with cdlt.loop(OH) as y:
                        with cdlt.loop(OW) as x:
                            with cdlt.loop(KH) as kh:
                                with cdlt.loop(KW) as kw:
                                    cdlt.transfer(weight, ["DRAM", "VMEM1"])
                                    cdlt.transfer(bias, ["DRAM", "VMEM2"])
                                    cdlt.transfer(out, ["DRAM", "VMEM1"])
                                    clip_out1.set_write_destination("VMEM2")

                                    cdlt.compute("MAX", [conv_out[n, c, y * s + kh, x * s + kw], max_op],
                                                 [clip_out1[n, c, y * s + kh, x * s + kw]
                                                  ],
                                                 target="SIMD")

                                    cdlt.compute("MIN", [clip_out1[n, c, y * s + kh, x * s + kw], min_op],
                                                 [clip_out1[n, c, y * s + kh, x * s + kw]],
                                                 target="SIMD")

                                    out.set_write_destination("VMEM1")

                                    cdlt.compute("MACC",
                                                 [clip_out1[n, c, y * s + kh, x * s + kw], weight[c, one, kh, kw],
                                                  out[n, c, y, x]], [out[n, c, y, x]],
                                                 target="SIMD")

                                    cdlt.compute("ADD", [out[n, c, y, x], bias[c]], [out[n,c,y,x]], target="SIMD")

                                    cdlt.compute("MAX", [out[n, c, y, x], max_op],
                                                 [out[n, c, y, x]
                                                  ],
                                                 target="SIMD")

                                    cdlt.compute("MIN", [out[n, c, y, x], min_op],
                                                 [out[n, c, y, x]],
                                                 target="SIMD")
                                    # dw_conv_out.set_write_destination("VMEM1")
                                    #
                                    # cdlt.compute("MAX", [out[n, c, y, x], max_op],
                                    #              [dw_conv_out[n, c, y, x]
                                    #               ],
                                    #              target="SIMD")
                                    #
                                    # cdlt.compute("MIN", [dw_conv_out[n, c, y, x], min_op],
                                    #              [out[n, c, y, x]],
                                    #              target="SIMD")

                                    cdlt.transfer(out, ["VMEM1", "DRAM"])

        cdlt.configure("end", "SIMD")
    cdlt = add_conv_constraints(hag, cdlt, is_fusion=True)

    cdlt = add_simd_constraint(hag, cdlt, "OC")
    return cdlt

def inv_sqrt(hag: ArchitectureNode):
    with CodeletTemplate("elem_sqrt_reciprocal") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])

        cdlt.configure("start", "SIMD")
        with cdlt.loop(C) as c:
            with cdlt.loop(N) as n:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(data, ["DRAM", "VMEM1"])
                        out.set_write_destination('VMEM1')
                        cdlt.compute('INV_SQRT', [data[n, c, h, w]], [out[n, c, h, w]], target='SIMD')
                        cdlt.transfer(out, ['VMEM1', 'DRAM'])
        cdlt.configure("end", "SIMD")
    cdlt = add_simd_constraint(hag, cdlt, "C")
    return cdlt

FUSION_OP_INFO = {
    'conv_bias_relu': {
        'cdlt': conv_relu,
        'seq': ['Conv', 'Relu']
    },
    'conv_bias_elem_add_relu': {
        'cdlt': conv_add_relu,
        'seq': ['Conv', 'Add', 'Relu'],
    },
    'conv_bias_leaky_relu': {
        'cdlt': conv_leaky_relu,
        'seq': ['Conv', 'LeakyRelu']
    },
    'conv_bias_elem_add_leaky_relu': {
        'cdlt': conv_add_leaky_relu,
        'seq': ['Conv', 'Add', 'LeakyRelu'],
    },
    'conv_bias_elem_clip_depthwise_conv_bias': {
        'cdlt': conv_bias_elem_clip_depthwise_conv_bias,
        'seq': ['Conv', 'Clip', 'DepthwiseConv'],

    },
    'conv_bias_elem_clip_depthwise_conv_bias_elem_clip': {
        'cdlt': conv_bias_elem_clip_depthwise_conv_bias_elem_clip,
        'seq': ['Conv', 'Clip', 'DepthwiseConv', 'Clip'],

    },
    'single_layer_info':
        {
            'Conv' : {'inputs': 3, 'outputs': 1},
            'Relu' : {'inputs': 1, 'outputs': 1},
            'LeakyRelu' : {'inputs': 1, 'outputs': 1},
            'Add' : {'inputs': 2, 'outputs': 1},
            'MaxPool': {'inputs': 1, 'outputs': 1}
        }
}

FUSION_CODELETS = {k : v['cdlt'] for k,v in FUSION_OP_INFO.items() if k != 'single_layer_info'}
