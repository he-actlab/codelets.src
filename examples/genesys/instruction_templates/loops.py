from codelets.adl.graph import ArchitectureNode
from examples.genesys import ASIC_CONFIG
LOOPS_PER_LEVEL = 7


def outer_simd_loops(hag: ArchitectureNode):
    instructions = []
    # program, cdlt, op, hag
    ld_stride_str = f"operand.get_offset(cdlt, 1, op.loop_id, hag, outer_loop=True)*operand.dtype.bytes()"
    st_stride_str = f"operand.get_offset(cdlt, 1, op.loop_id, hag, outer_loop=True, movement_type='down')*operand.dtype.bytes()"
    # operand_str = f"cdlt.used_inputs"
    ld_operand_names = f"list(set([o.name for o in cdlt.read_operands]))"
    st_operand_names = f"list(set([o.name for o in cdlt.write_operands]))"
    ld_operand_str = f"[cdlt.get_operand(n) for n in {ld_operand_names}]"
    st_operand_str = f"[cdlt.get_operand(n) for n in {st_operand_names}]"
    simd_target = f'cdlt.is_loop_node_target(op, "SIMD")'
    not_dir_loop_dep = f'not cdlt.is_direct_loop_dep(op, "SIMD")'
    off_chip_operand = f'"DRAM" in operand.data_path'
    loop_conds = [simd_target, not_dir_loop_dep, off_chip_operand]
    ld_conds = loop_conds + [f"operand not in cdlt.outputs"]

    block_iter = ('operand', f'{ld_operand_str}')
    block_cond = " and ".join(ld_conds)

    macro_instr = hag.get_primitive_template("LD_CONFIG_BASE_LOOP_ITER")
    macro_instr.add_iterable(*block_iter)
    macro_instr.add_condition(block_cond)
    macro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    macro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    macro_instr.set_field_flex_param("NUM_ITERS", f"op.iter_count - 1")

    micro_instr = hag.get_primitive_template("LD_CONFIG_BASE_LOOP_STRIDE")
    micro_instr.add_iterable(*block_iter)
    micro_instr.add_condition(block_cond)
    micro_instr.set_field_by_name("LSB_MSB", "LSB")
    micro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    micro_instr.set_field_flex_param("STRIDE",
                                     f"program.extract_bits({ld_stride_str}, 16, 0)"
                                     )
    macro_instr.add_base_instruction(micro_instr)

    micro_instr = hag.get_primitive_template("LD_CONFIG_BASE_LOOP_STRIDE")
    micro_instr.add_iterable(*block_iter)
    micro_instr.add_condition(block_cond)

    micro_instr.set_field_by_name("LSB_MSB", "MSB")
    micro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    micro_instr.set_field_flex_param("STRIDE",
                                     f"program.extract_bits({ld_stride_str}, 16, 16)"
                                     )
    macro_instr.add_base_instruction(micro_instr)
    instructions.append(macro_instr)

    macro_instr = hag.get_primitive_template("ST_CONFIG_BASE_LOOP_ITER")
    macro_instr.add_iterable('operand', f'{st_operand_str}')
    macro_instr.add_condition(" and ".join(loop_conds))

    macro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    macro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    macro_instr.set_field_flex_param("NUM_ITERS", f"op.iter_count - 1")

    micro_instr = hag.get_primitive_template("ST_CONFIG_BASE_LOOP_STRIDE")
    micro_instr.add_iterable('operand', f'{st_operand_str}')
    # micro_instr.add_condition(f'cdlt.is_loop_node_target(op, "SIMD") and not cdlt.is_direct_loop_dep(op, "SIMD")')
    micro_instr.add_condition(" and ".join(loop_conds))

    micro_instr.set_field_by_name("LSB_MSB", "LSB")
    micro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    micro_instr.set_field_flex_param("STRIDE", f"program.extract_bits({st_stride_str}, 16, 0)")
    macro_instr.add_base_instruction(micro_instr)

    micro_instr = hag.get_primitive_template("ST_CONFIG_BASE_LOOP_STRIDE")
    micro_instr.add_iterable('operand', f'{st_operand_str}')
    # micro_instr.add_condition(f'cdlt.is_loop_node_target(op, "SIMD") and not cdlt.is_direct_loop_dep(op, "SIMD")')
    micro_instr.add_condition(" and ".join(loop_conds))
    micro_instr.set_field_by_name("LSB_MSB", "MSB")
    micro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    micro_instr.set_field_flex_param("STRIDE", f"program.extract_bits({st_stride_str}, 16, 16)")
    macro_instr.add_base_instruction(micro_instr)
    instructions.append(macro_instr)



    return instructions


def inner_simd_loops(hag: ArchitectureNode):
    instructions = []
    is_direct_loop_dep = f"cdlt.is_direct_loop_dep(loop_op, 'SIMD')"

    return instructions


def outer_sa_loops(hag: ArchitectureNode):
    instructions = []
    loop_cond_str = 'cdlt.is_loop_node_target(op, "pe_array") and not cdlt.is_direct_loop_dep(op, "pe_array")'
    reduction_loop_cond = '("conv" in cdlt.op_name and cdlt.loop_param_map[op.op_str] in ["IC", "KH", "KW"]) ' \
                          'or ("gemm" in cdlt.op_name and cdlt.loop_param_map[op.op_str] == "N")'
    instr = hag.get_primitive_template("SA_LOOP_CFG")
    instr.add_condition(loop_cond_str)
    instr.set_field_flex_param("LOOP_ID", "op.loop_id")
    instr.set_field_flex_param("NUM_ITERATIONS", "op.iter_count - 1")
    instructions.append(instr)

    instr = hag.get_primitive_template("SA_REDUCTION_LOOP")
    instr.add_condition(f'{loop_cond_str} and ({reduction_loop_cond})')
    instr.set_field_flex_param("LOOP_DIM", "cdlt.loop_param_map[op.op_str]")
    instr.set_field_by_name("LOOP_TYPE", "OUTER")
    instr.set_field_flex_param("LOOP_ID", "op.loop_id")
    instructions.append(instr)

    denom_str = f"hag.get_subgraph_edge('DRAM', operand.get_ld_storage_location(cdlt, 1)).bandwidth"
    if ASIC_CONFIG:
        # Product of stride of inner loops * stride * operand.dtype_size / bandwidth DRAM-BUF 256 (word/line size)
        # stride_str = f"operand.get_offset(cdlt, 1, op.loop_id, hag)*op.stride"
        stride_str = f"operand.get_offset(cdlt, 1, op.loop_id, hag, outer_loop=True)*operand.dtype.bits()//{denom_str}"
    else:
        # Product of iteration of inner loops * stride * operand.dtype_size / 8
        stride_str = f"(operand.get_offset(cdlt, 1, op.loop_id, hag,outer_loop=True)*operand.dtype.bits()//8)"

    # stride_str = f"operand.get_offset_(cdlt, 'DRAM', 1, op.loop_id, hag)*op.stride"
    macro_instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    macro_instr.add_iterable('operand', f'cdlt.operands')
    macro_instr.add_condition(loop_cond_str)
    macro_instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    macro_instr.set_field_by_name("ACCESS_TYPE", "LD")
    macro_instr.set_field_flex_param("LOOP_ID", "op.loop_id")
    macro_instr.set_field_flex_param("BUFFER", f"operand.get_ld_storage_location(cdlt, 1)")
    macro_instr.set_field_flex_param("STRIDE",
                                     f"program.extract_bits({stride_str}, 16, 0)")

    micro_instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    micro_instr.add_iterable('operand', f'cdlt.operands')
    micro_instr.add_condition(loop_cond_str)
    micro_instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    micro_instr.set_field_by_name("ACCESS_TYPE", "LD")
    micro_instr.set_field_flex_param("LOOP_ID", "op.loop_id")
    micro_instr.set_field_flex_param("BUFFER", f"operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("STRIDE",
                                     f"program.extract_bits({stride_str}, 16, 16)")
    macro_instr.add_base_instruction(micro_instr)
    instructions.append(macro_instr)
    if ASIC_CONFIG:
        denom_str = f"hag.get_subgraph_edge('DRAM', cdlt.outputs[0].get_ld_storage_location(cdlt, 1)).bandwidth"

        out_stride_str = f"cdlt.outputs[0].get_offset(cdlt, 1, op.loop_id, hag, movement_type='down', outer_loop=True)*cdlt.outputs[0].dtype.bits() // {denom_str}"
        # out_stride_str = f"cdlt.outputs[0].get_offset(cdlt, 1, op.loop_id, hag)*op.stride"
    else:
        out_stride_str = f"(cdlt.outputs[0].get_offset(cdlt, 1, op.loop_id, hag, movement_type='down', outer_loop=True)*cdlt.outputs[0].dtype.bits()//8) "

    # out_stride_str = f"cdlt.outputs[0].get_offset(cdlt, 1, op.loop_id, hag)"
    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_condition(loop_cond_str)
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_flex_param("LOOP_ID", "op.loop_id")
    instr.set_field_flex_param("BUFFER", f"cdlt.outputs[0].get_ld_storage_location(cdlt, 1)")
    instr.set_field_flex_param("STRIDE", "program.extract_bits("
                                         f"{out_stride_str},"
                                         "16, 0)")

    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_condition(loop_cond_str)
    instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_flex_param("LOOP_ID", "op.loop_id")
    instr.set_field_flex_param("BUFFER", f"cdlt.outputs[0].get_ld_storage_location(cdlt, 1)")
    instr.set_field_flex_param("STRIDE", "program.extract_bits("
                                         f"{out_stride_str},"
                                         "16, 16)")
    instructions.append(instr)
    return instructions


def inner_sa_loops(hag: ArchitectureNode):
    instructions = []
    inner_loop_id_str = f"(op.loop_id % {LOOPS_PER_LEVEL}) + {LOOPS_PER_LEVEL}"
    reduction_loop_cond = '("conv" in cdlt.op_name and cdlt.loop_param_map[op.op_str] in ["IC", "KH", "KW"]) ' \
                          'or ("gemm" in cdlt.op_name and cdlt.loop_param_map[op.op_str] == "N")'
    sa_loop_cond = '("conv" in cdlt.op_name and cdlt.loop_param_map[op.op_str] in ["IC", "OC"]) ' \
                          'or ("gemm" in cdlt.op_name and cdlt.loop_param_map[op.op_str] in ["N", "P"])'
    instr = hag.get_primitive_template("SA_LOOP_CFG")
    instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array")')

    instr.set_field_flex_param("LOOP_ID", inner_loop_id_str)
    sa_constraints = hag.get_subgraph_node("pe_array").dimensions[0]
    n_iter_str = f"int(np.ceil(op.iter_count / {sa_constraints})) - 1" \
                 f"if {sa_loop_cond} else op.iter_count - 1"
    instr.set_field_flex_param("NUM_ITERATIONS", f"{n_iter_str}")
    instructions.append(instr)

    instr = hag.get_primitive_template("SA_REDUCTION_LOOP")
    instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array") and ({reduction_loop_cond})')
    instr.set_field_flex_param("LOOP_DIM", "cdlt.loop_param_map[op.op_str]")
    instr.set_field_by_name("LOOP_TYPE", "INNER")
    instr.set_field_flex_param("LOOP_ID", f"op.loop_id % (cdlt.op_id_counters['loop']//2)")
    instructions.append(instr)

    # offset_str = f"operand.get_offset(cdlt, 2, op.loop_id)//{sa_constraints} if {sa_loop_cond} else operand.get_offset(cdlt, 2, op.loop_id)"
    offset_str = f"operand.get_offset(cdlt, 2, op.loop_id, hag, outer_loop=False) * op.stride"
    macro_instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    macro_instr.add_iterable('operand', f'cdlt.operands')
    macro_instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array")')
    macro_instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    macro_instr.set_field_by_name("ACCESS_TYPE", "RD")
    macro_instr.set_field_flex_param("LOOP_ID", inner_loop_id_str)
    macro_instr.set_field_flex_param("BUFFER", f"operand.get_ld_storage_location(cdlt, 1)")
    macro_instr.set_field_flex_param("STRIDE",
                                     f"program.extract_bits({offset_str}, 16, 0)"
                                     )

    micro_instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    micro_instr.add_iterable('operand', f'cdlt.operands')
    micro_instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array")')
    micro_instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    micro_instr.set_field_by_name("ACCESS_TYPE", "RD")
    micro_instr.set_field_flex_param("LOOP_ID", inner_loop_id_str)
    micro_instr.set_field_flex_param("BUFFER", f"operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("STRIDE",
                                     # "program.extract_bits(operand.get_offset(cdlt, 2, op.loop_id, hag, outer_loop=False), 16, 16)"
                                     f"program.extract_bits({offset_str}, 16, 16)"
                                     )
    macro_instr.add_base_instruction(micro_instr)
    instructions.append(macro_instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array")')
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "WR")
    instr.set_field_flex_param("LOOP_ID", inner_loop_id_str)
    instr.set_field_flex_param("BUFFER", f"cdlt.outputs[0].get_ld_storage_location(cdlt, 1)")
    instr.set_field_flex_param("STRIDE",
                               "program.extract_bits(cdlt.outputs[0].get_offset(cdlt, 2, op.loop_id, hag, outer_loop=False, movement_type='down'), 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array")')
    instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    instr.set_field_by_name("ACCESS_TYPE", "WR")
    instr.set_field_flex_param("LOOP_ID", inner_loop_id_str)
    instr.set_field_flex_param("BUFFER", f"cdlt.outputs[0].get_ld_storage_location(cdlt, 1)")
    instr.set_field_flex_param("STRIDE",
                               "program.extract_bits(cdlt.outputs[0].get_offset(cdlt, 2, op.loop_id, hag, outer_loop=False, movement_type='down'), 16, 16)")
    instructions.append(instr)
    return instructions


def loop_template(hag: ArchitectureNode):
    instructions = []
    # SYSTOLIC ARRAY LOOP INSTR

    # OUTER LOOP
    instructions += outer_sa_loops(hag)

    # INNER LOOP
    instructions += inner_sa_loops(hag)


    # SIMD ARRAY LOOP INSTR

    # OUTER LOOP
    instructions += outer_simd_loops(hag)
    # INNER LOOP INSTR
    instructions += inner_simd_loops(hag)


    return instructions


def loop_end_template(hag: ArchitectureNode):
    instructions = []
    return instructions