from codelets.adl.graph import ArchitectureNode
VMEM_ID_MAP = {'LD': {'VMEM1': 0, 'VMEM2': 1},
                 'ST': {'VMEM1': 2, 'VMEM2': 3}
               }

def move_zero_to_mem(hag: ArchitectureNode, buffer_name):
    instructions = []
    all_loop_id = f"(len(cdlt.get_ops_by_type('loop'))//2)"
    other_buff = "VMEM2" if buffer_name == "VMEM1" else "VMEM1"
    n_banks = f"hag.get_subgraph_node('{buffer_name}').banks"
    ld_st_tabs = f"op.loop_level + 1"
    buff_name_str = f"'{buffer_name}'"

    # ns_idx = f"{all_loop_id} + op.operand.get_mem_index({buff_name_str}) * {all_loop_id}"
    ns_idx = f"op.operand.get_mem_index('{buffer_name}')"
    # ns_idx = f"0"
    imm_ns_idx = f"cdlt.get_temp_idx('init')"
    imm_base_sign_ext = f"op.operand.get_mem_offset({buff_name_str})//op.operand.dtype.bits()"
    base_sign_ext_low = f"program.extract_bits({imm_base_sign_ext}, 16, 0)"
    base_sign_ext_high = f"program.extract_bits({imm_base_sign_ext}, 16, 16)"
    bitwidth = f"len(np.binary_repr({imm_base_sign_ext})) + int(np.signbit({imm_base_sign_ext}))"
    bitwidth_cond = f"{bitwidth} <= 16"

    # First set the sign extensions for IMM
    imm_operand_cond = f"op.operand in cdlt.outputs"
    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.add_condition(imm_operand_cond)
    macro_instr.set_field_by_name('NS_ID', "IMM")
    macro_instr.set_field_flex_param('NS_INDEX_ID', f"0")
    macro_instr.set_field_flex_param('IMM', imm_ns_idx)

    micro_instr1 = hag.get_primitive_template("STRIDE_SIGN_EXT")
    micro_instr1.add_condition(imm_operand_cond)
    micro_instr1.set_field_by_name('NS_ID', "IMM")
    micro_instr1.set_field_flex_param('NS_INDEX_ID', f"0")
    micro_instr1.set_field_flex_param('IMM', f"0")
    macro_instr.add_base_instruction(micro_instr1)

    instructions.append(macro_instr)

    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.add_condition(f"{imm_operand_cond} and {bitwidth_cond}")
    macro_instr.set_field_by_name('NS_ID', buffer_name)
    macro_instr.set_field_flex_param('NS_INDEX_ID', f"{ns_idx}")
    macro_instr.set_field_flex_param('IMM', imm_base_sign_ext)
    instructions.append(macro_instr)

    low_instr = hag.get_primitive_template("BASE_LOW")
    low_instr.add_condition(f"{imm_operand_cond} and not {bitwidth_cond}")
    low_instr.set_field_by_name('NS_ID', buffer_name)
    low_instr.set_field_flex_param('NS_INDEX_ID', f"{ns_idx}")
    low_instr.set_field_flex_param('IMM', base_sign_ext_low)
    instructions.append(low_instr)

    high_instr = hag.get_primitive_template("BASE_HIGH")
    high_instr.add_condition(f"{imm_operand_cond} and not {bitwidth_cond}")
    high_instr.set_field_by_name('NS_ID', buffer_name)
    high_instr.set_field_flex_param('NS_INDEX_ID', f"{ns_idx}")
    high_instr.set_field_flex_param('IMM', base_sign_ext_high)
    instructions.append(high_instr)

    micro_instr1 = hag.get_primitive_template("STRIDE_SIGN_EXT")
    micro_instr1.add_condition(imm_operand_cond)
    micro_instr1.set_field_by_name('NS_ID', buffer_name)
    micro_instr1.set_field_flex_param('NS_INDEX_ID', f"{ns_idx}")
    micro_instr1.set_field_value('IMM', 1)
    instructions.append(micro_instr1)


    iter_instr = hag.get_primitive_template("SET_ITER")
    iter_instr.add_condition(imm_operand_cond)
    iter_instr.set_field_flex_param("LOOP_ID", imm_ns_idx)
    iter_instr.set_field_flex_param("NUM_ITER",
                                   f"op.operand.get_tile_size({buff_name_str}, 'SIMD')//{n_banks}")
    instructions.append(iter_instr)

    idx_instr = hag.get_primitive_template("SET_INDEX")
    idx_instr.add_condition(imm_operand_cond)
    idx_instr.set_field_by_name("DST_NS_ID", f"{buffer_name}")
    idx_instr.set_field_flex_param("DST_INDEX_ID", f"{ns_idx}")
    idx_instr.set_field_by_name("SRC1_NS_ID", "IMM")
    idx_instr.set_field_flex_param("SRC1_INDEX_ID", f"0")
    idx_instr.set_field_by_name("SRC2_NS_ID", other_buff)
    idx_instr.set_field_flex_param("SRC2_INDEX_ID", f"0")
    instructions.append(idx_instr)

    set_inst_instr = hag.get_primitive_template("SET_INST")
    set_inst_instr.add_condition(imm_operand_cond)
    set_inst_instr.set_field_flex_param("SINGLE_NESTED", "1")
    set_inst_instr.set_field_flex_param("NUM_INSTR", "1")
    instructions.append(set_inst_instr)

    instr = hag.get_primitive_template(f"MOVE")
    instr.add_condition(imm_operand_cond)
    instr.set_field_by_name("DST_NS_ID", f"{buffer_name}")
    instr.set_field_flex_param("DST_INDEX_ID", f"{ns_idx}")
    instr.set_field_by_name("SRC1_NS_ID", f"IMM")
    instr.set_field_flex_param("SRC1_INDEX_ID", f"0")
    instr.set_field_by_name("SRC2_NS_ID", other_buff)
    instr.set_field_flex_param("SRC2_INDEX_ID", f"0")
    instr.set_print_tabs(ld_st_tabs)
    instructions.append(instr)

    return instructions

def off_chip_transfer_simd(ld_st, buffer_name, hag: ArchitectureNode):
    instructions = []
    ### TILE LOOP
    loop_id_str = f"cdlt.op_id_counters['loop'] + {VMEM_ID_MAP['LD'][buffer_name]} + dim_info[0]"
    loop_iter_str = f"dim_info[1][1] - 1"
    n_banks = f"hag.get_subgraph_node('{buffer_name}').banks"
    data_width = f"hag.get_subgraph_node('DRAM').width"

    iterable_str = f"enumerate(zip(*op.strides_iters({data_width}, {hag.meta_cfg['MERGE_LDST_LOOPS']}, divisor={n_banks}, max_bits=32)))"
    ld_st_tabs = f"op.loop_level + len(op.sizes_for_node('{buffer_name}'))"

    req_size_str = f"op.strides_iters({data_width}, {hag.meta_cfg['MERGE_LDST_LOOPS']}, divisor={n_banks}, max_bits=32)[0][-1]"

    ####
    ## LOADS FOR INPUT OPERANDS
    # base_addr_str = f"op.operand.get_mem_offset('{buffer_name}')//(op.operand.dtype.bits()) + 1"
    base_addr_str = f"op.operand.get_mem_offset('{buffer_name}')//(hag.get_subgraph_node('{buffer_name}').data_size)"

    if ld_st == "LD":
        imm_operand_cond = f"op.operand in cdlt.outputs"
        not_imm_operand_cond = f"op.operand not in cdlt.outputs"
        base_addr_instr_lsb = hag.get_primitive_template(f"{ld_st}_CONFIG_TILE_ADDR")
        base_addr_instr_lsb.add_condition(not_imm_operand_cond)
        base_addr_instr_lsb.set_field_by_name("NS_ID", buffer_name)
        base_addr_instr_lsb.set_field_by_name("LSB_MSB", f"LSB")
        base_addr_instr_lsb.set_field_flex_param("BASE_ADDR",
                                                 f"program.extract_bits({base_addr_str}, 16,0)")
        instructions.append(base_addr_instr_lsb)

        base_addr_instr_msb = hag.get_primitive_template(f"{ld_st}_CONFIG_TILE_ADDR")
        base_addr_instr_msb.add_condition(not_imm_operand_cond)
        base_addr_instr_msb.set_field_by_name("NS_ID", buffer_name)
        base_addr_instr_msb.set_field_by_name("LSB_MSB", f"MSB")
        base_addr_instr_msb.set_field_flex_param("BASE_ADDR",
                                                 f"program.extract_bits({base_addr_str}, 16,16)")
        instructions.append(base_addr_instr_msb)

        macro_instr = hag.get_primitive_template(f"{ld_st}_CONFIG_TILE_LOOP_ITER")
        macro_instr.add_iterable('dim_info', iterable_str)
        macro_instr.add_condition(not_imm_operand_cond)
        macro_instr.set_field_by_name("NS_ID", buffer_name)
        macro_instr.set_field_flex_param("LOOP_INDEX_ID", loop_id_str)
        macro_instr.set_field_flex_param("NUM_ITERS", f"{loop_iter_str}")
        macro_instr.set_print_tabs("op.loop_level + dim_info[0]")
        stride_size_str = f"dim_info[1][0]"

        micro_instr = hag.get_primitive_template(f"{ld_st}_CONFIG_TILE_LOOP_STRIDE")
        micro_instr.add_iterable('dim_info', iterable_str)
        micro_instr.add_condition(not_imm_operand_cond)
        micro_instr.set_field_by_name("LSB_MSB", f"LSB")
        micro_instr.set_field_by_name("NS_ID", f"{buffer_name}")
        micro_instr.set_field_flex_param("LOOP_INDEX_ID", loop_id_str)
        micro_instr.set_field_flex_param("STRIDE", f"program.extract_bits({stride_size_str}, 16,0)")
        micro_instr.set_print_tabs("op.loop_level + dim_info[0]")
        macro_instr.add_base_instruction(micro_instr)

        micro_instr = hag.get_primitive_template(f"{ld_st}_CONFIG_TILE_LOOP_STRIDE")
        micro_instr.add_iterable('dim_info', iterable_str)
        micro_instr.add_condition(not_imm_operand_cond)
        micro_instr.set_field_by_name("LSB_MSB", f"MSB")
        micro_instr.set_field_by_name("NS_ID", f"{buffer_name}")
        micro_instr.set_field_flex_param("LOOP_INDEX_ID", loop_id_str)
        micro_instr.set_field_flex_param("STRIDE", f"program.extract_bits({stride_size_str}, 16,16)")
        micro_instr.set_print_tabs("op.loop_level + dim_info[0]")
        macro_instr.add_base_instruction(micro_instr)

        instructions.append(macro_instr)
        instr = hag.get_primitive_template(f"{ld_st}_START")
        instr.add_condition(not_imm_operand_cond)
        instr.set_field_by_name("NS_ID", f"{buffer_name}")
        instr.set_field_flex_param(f"{ld_st}_DATA_WIDTH", "op.operand.dtype.bits() - 1")
        instr.set_field_flex_param("REQUEST_SIZE", f"{req_size_str}//{n_banks}")
        instr.set_print_tabs(ld_st_tabs)
        instructions.append(instr)

        ### LOADS FOR IMM OPERANDS FORMERLY OUTPUT
        instructions += move_zero_to_mem(hag, buffer_name)
    else:
        base_addr_instr_lsb = hag.get_primitive_template(f"{ld_st}_CONFIG_TILE_ADDR")
        base_addr_instr_lsb.set_field_by_name("NS_ID", buffer_name)
        base_addr_instr_lsb.set_field_by_name("LSB_MSB", f"LSB")
        base_addr_instr_lsb.set_field_flex_param("BASE_ADDR",
                                                 f"program.extract_bits({base_addr_str}, 16,0)")
        instructions.append(base_addr_instr_lsb)

        base_addr_instr_msb = hag.get_primitive_template(f"{ld_st}_CONFIG_TILE_ADDR")
        base_addr_instr_msb.set_field_by_name("NS_ID", buffer_name)
        base_addr_instr_msb.set_field_by_name("LSB_MSB", f"MSB")
        base_addr_instr_msb.set_field_flex_param("BASE_ADDR",
                                                 f"program.extract_bits({base_addr_str}, 16,16)")
        instructions.append(base_addr_instr_msb)

        macro_instr = hag.get_primitive_template(f"{ld_st}_CONFIG_TILE_LOOP_ITER")
        macro_instr.add_iterable('dim_info', iterable_str)
        macro_instr.set_field_by_name("NS_ID", buffer_name)
        macro_instr.set_field_flex_param("LOOP_INDEX_ID", loop_id_str)
        macro_instr.set_field_flex_param("NUM_ITERS", f"{loop_iter_str}")
        macro_instr.set_print_tabs("op.loop_level + dim_info[0]")
        stride_size_str = f"dim_info[1][0]"

        micro_instr = hag.get_primitive_template(f"{ld_st}_CONFIG_TILE_LOOP_STRIDE")
        micro_instr.add_iterable('dim_info', iterable_str)
        micro_instr.set_field_by_name("LSB_MSB", f"LSB")
        micro_instr.set_field_by_name("NS_ID", f"{buffer_name}")
        micro_instr.set_field_flex_param("LOOP_INDEX_ID", loop_id_str)
        micro_instr.set_field_flex_param("STRIDE", f"program.extract_bits({stride_size_str}, 16,0)")
        micro_instr.set_print_tabs("op.loop_level + dim_info[0]")
        macro_instr.add_base_instruction(micro_instr)

        micro_instr = hag.get_primitive_template(f"{ld_st}_CONFIG_TILE_LOOP_STRIDE")
        micro_instr.add_iterable('dim_info', iterable_str)
        micro_instr.set_field_by_name("LSB_MSB", f"MSB")
        micro_instr.set_field_by_name("NS_ID", f"{buffer_name}")
        micro_instr.set_field_flex_param("LOOP_INDEX_ID", loop_id_str)
        micro_instr.set_field_flex_param("STRIDE", f"program.extract_bits({stride_size_str}, 16,16)")
        micro_instr.set_print_tabs("op.loop_level + dim_info[0]")
        macro_instr.add_base_instruction(micro_instr)

        instructions.append(macro_instr)
        instr = hag.get_primitive_template(f"{ld_st}_START")
        instr.set_field_by_name("NS_ID", f"{buffer_name}")
        instr.set_field_flex_param(f"{ld_st}_DATA_WIDTH", "op.operand.dtype.bits() - 1")
        instr.set_field_flex_param("REQUEST_SIZE", f"{req_size_str}//{n_banks}")
        instr.set_print_tabs(ld_st_tabs)
        instructions.append(instr)

    ####

    noop_instr = hag.get_primitive_template("NOP")
    noop_instr.set_print_tabs("op.loop_level")
    instructions.append(noop_instr)

    return instructions

