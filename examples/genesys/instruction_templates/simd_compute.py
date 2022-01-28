from codelets.adl.graph import ArchitectureNode
OP_COMPUTE_CYCLES = {
    "SIGMOID": 4,
    "RELU": 0,
    "ADD": 0,
    "SUB": 0,
    "DIV": 0,
    "LEAKY_RELU": 0,
    "MAX": 0,
    "MIN": 0,
    "MUL": 0,
    "MACC": 2,
    "RSHIFT": 0,
    "LSHIFT": 0,
    "MOVE": 0,
    "COND_MOVE_TRUE": 0,
    "COND_MOVE_FALSE": 0,
    "NOT": 0,
    "AND": 0,
    "OR": 0,
    "NOP": 0,
    "TANH": 4,
    "EXP": 0,
    "POW": 0,
    "LN": 0,
    "SQRT": 0,
    "INV_SQRT": 0,
    "LOG2": 0,
    "EQUAL": 0,
    "NEQ": 0,
    "GT": 0,
    "GTE": 0,
    "LTE": 0,
    "LT": 0,
    "32FXP_16FXP": 0,
    "32FXP_8FXP": 0,
    "32FXP_4FXP": 0,
    "16FXP_32FXP": 0,
    "8FXP_32FXP": 0,
    "4FXP_32FXP": 0,
    "32FP_16FP": 0,
    "16FP_32FP": 0,
    "32FP_162BFP": 0,
    "16BFP_32BFP": 0,
    "FLOOR": 0,
    "CEIL": 0,
}

def simd_alu_template(op_name, hag: ArchitectureNode):
    instructions = []

    loop_idx_offset = 0 if op_name != "POW" else 1
    # Params
    simd_size = hag.get_subgraph_node("SIMD").dimensions[0]
    all_loop_id = f"(len(cdlt.get_ops_by_type('loop'))//2)"

    # Loops
    operand_iter = ("operand", "op.operands_by_unique_location")
    loop_iter = ('loop_op', f'cdlt.get_ops_by_type("loop")')

    # Stride calculation
    mvmt_type = f"'up' if operand.data_path[0] == 'DRAM' else 'down'"
    loop_stride = f"operand.get_offset(cdlt, 2," \
                  f"loop_op.loop_id," \
                  f"hag, movement_type={mvmt_type}, " \
                  f"outer_loop=False)"

    # Instruction generation conditions
    conds = []
    is_dependency = f'loop_op.op_str in cdlt.all_dependencies(op.dependencies)'
    outer_loop_level = f'loop_op.loop_level < op.loop_level'
    is_direct_loop_dep = f"cdlt.is_direct_loop_dep(loop_op, 'SIMD')"
    zero_stride_cond = f"{loop_stride} == 0"

    conds.append(is_dependency)
    conds.append(outer_loop_level)
    conds.append(is_direct_loop_dep)
    # Operand location string
    op_loc = f"op.get_operand_location(operand.name)"
    # Index generation
    ns_idx = f"(loop_op.loop_id % {all_loop_id}) + (operand.get_mem_index({op_loc}) * {all_loop_id}) if op.get_operand_location(operand.name) != 'IMM' else cdlt.temps.index(operand)"
    base_sign_ext = f"operand.get_mem_offset({op_loc})//(operand.dtype.bits()) if op.get_operand_location(operand.name) " \
                    f"!= 'IMM' else cdlt.temps.index(operand)"

    ### Base and stride first
    # instructions += base_sign_ext_gen(op_name, hag)
    # macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    # macro_instr.set_print_tabs(all_loop_id)
    # macro_instr.add_iterable(*operand_iter)
    # macro_instr.add_iterable(*loop_iter)
    # macro_instr.add_condition(" and ".join(conds))
    # macro_instr.set_field_flex_param('NS_ID', op_loc)
    # macro_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    # macro_instr.set_field_flex_param('IMM', base_sign_ext)
    #
    #
    #
    # sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    # sub_instr.set_print_tabs(all_loop_id)
    # sub_instr.add_iterable(*operand_iter)
    # sub_instr.add_iterable(*loop_iter)
    # sub_instr.add_condition(" and ".join(conds))
    # sub_instr.set_field_flex_param('NS_ID', op_loc)
    # # sub_instr.set_field_flex_param('NS_INDEX_ID', f"loop_op.loop_id % {all_loop_id}")
    # sub_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    # sub_instr.set_field_flex_param('IMM', loop_stride)
    # macro_instr.add_base_instruction(sub_instr)
    #
    # instructions.append(macro_instr)

    if op_name == "POW":
        # Additional loop requires additional base/offset
        pow_ns_idx = f"({all_loop_id}) + (operand.get_mem_index({op_loc}) * {all_loop_id}) if op.get_operand_location(operand.name) != 'IMM' else cdlt.temps.index(operand)"

        macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
        macro_instr.set_print_tabs(all_loop_id)
        macro_instr.add_iterable(*operand_iter)
        macro_instr.set_field_flex_param('NS_ID', op_loc)
        macro_instr.set_field_flex_param('NS_INDEX_ID', pow_ns_idx)
        macro_instr.set_field_flex_param('IMM', base_sign_ext)
        #
        sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
        sub_instr.set_print_tabs(all_loop_id)
        sub_instr.add_iterable(*operand_iter)
        sub_instr.set_field_flex_param('NS_ID', op_loc)
        sub_instr.set_field_flex_param('NS_INDEX_ID', pow_ns_idx)
        sub_instr.set_field_flex_param('IMM', '0')
        macro_instr.add_base_instruction(sub_instr)
        instructions.append(macro_instr)

        exp_tabs = f"len(cdlt.get_ops_by_type('loop'))"
        macro_instr = hag.get_primitive_template("SET_ITER")
        macro_instr.set_print_tabs(all_loop_id)
        macro_instr.set_field_flex_param("LOOP_ID", f"0")
        macro_instr.set_field_flex_param("NUM_ITER",
                                         f"cdlt.required_params['exp'].value - 1")

        # Extra set index
        sub_instr = hag.get_primitive_template("SET_INDEX")
        set_index_fmt = "({all_loop_id}) + ({operand}.get_mem_index({op_loc}) * {all_loop_id}) if op.get_operand_location({operand}.name) != 'IMM' else cdlt.temps.index({operand})"

        sub_instr.set_print_tabs(all_loop_id)
        sub_instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
        sub_instr.set_field_flex_param("DST_INDEX_ID",
                                       set_index_fmt.format(all_loop_id=all_loop_id, operand="op.dests[0]",
                                                            op_loc="op.get_operand_location(op.dests[0].name)"))
        sub_instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
        sub_instr.set_field_flex_param("SRC1_INDEX_ID",
                                       set_index_fmt.format(all_loop_id=all_loop_id, operand="op.sources[0]",
                                                            op_loc="op.get_operand_location(op.sources[0].name)"))
        sub_instr.set_field_flex_param("SRC2_NS_ID",
                                       f"'IMM' if len(op.sources) == 1 else op.get_operand_location(op.sources[1].name)")
        src2_idx = f"0 if len(op.sources) == 1 else " + set_index_fmt.format(all_loop_id=all_loop_id,
                                                                             operand="op.sources[1]",
                                                                             op_loc="op.get_operand_location(op.sources[1].name)")
        sub_instr.set_field_flex_param("SRC2_INDEX_ID", src2_idx)
        macro_instr.add_base_instruction(sub_instr)
        instructions.append(macro_instr)

    ### iters and index
    other_constr = conds + ["loop_op.op_str in op.dependencies"]
    macro_instr = hag.get_primitive_template("SET_ITER")
    macro_instr.set_print_tabs("loop_op.loop_level")
    macro_instr.add_iterable(*loop_iter)
    macro_instr.add_condition(" and ".join(other_constr))
    macro_instr.set_field_flex_param("LOOP_ID", f"(loop_op.loop_id % {all_loop_id}) + {loop_idx_offset}")
    macro_instr.set_field_flex_param("NUM_ITER", f"loop_op.iter_count // {simd_size} if cdlt.loop_param_map[loop_op.op_str] in ['C', 'IC', 'OC'] else loop_op.iter_count")


    sub_instr = hag.get_primitive_template("SET_INDEX")
    set_index_fmt = "(loop_op.loop_id % {all_loop_id}) + ({operand}.get_mem_index({op_loc}) * {all_loop_id}) if op.get_operand_location({operand}.name) != 'IMM' else cdlt.temps.index({operand})"

    sub_instr.set_print_tabs("loop_op.loop_level")
    sub_instr.add_iterable(*loop_iter)
    sub_instr.add_condition(" and ".join(other_constr))
    sub_instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
    sub_instr.set_field_flex_param("DST_INDEX_ID",
                                   set_index_fmt.format(all_loop_id=all_loop_id, operand="op.dests[0]",
                                                        op_loc="op.get_operand_location(op.dests[0].name)"))
    sub_instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID",
                                   set_index_fmt.format(all_loop_id=all_loop_id, operand="op.sources[0]",
                                                        op_loc="op.get_operand_location(op.sources[0].name)"))
    sub_instr.set_field_flex_param("SRC2_NS_ID",
                                   f"'IMM' if len(op.sources) == 1 else op.get_operand_location(op.sources[1].name)")
    src2_idx = f"0 if len(op.sources) == 1 else " + set_index_fmt.format(all_loop_id=all_loop_id,
                                                                         operand="op.sources[1]",
                                                                         op_loc="op.get_operand_location(op.sources[1].name)")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", src2_idx)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)



    # Compute instruction
    instr = hag.get_primitive_template("SET_INST")
    instr.add_condition("cdlt.op_id_counters['compute'] - 1 > op.op_id")
    instr.set_field_flex_param("SINGLE_NESTED", "0 if op.num_loop_dependencies == 1 else 1")
    instr.set_field_flex_param("NUM_INSTR", "1")
    instructions.append(instr)
    rd_op_str = "op.sources[{IDX}].get_mem_index(op.get_operand_location(op.sources[{IDX}].name)) if op.get_operand_location(op.sources[{IDX}].name) != 'IMM' else cdlt.temps.index(op.sources[{IDX}])"

    if op_name != "POW":
        instr = hag.get_primitive_template(op_name)
        instr.add_condition("len(op.sources) > 1")
        instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
        instr.set_field_flex_param("DST_INDEX_ID", f"op.dests[0].get_mem_index(op.get_operand_location(op.dests[0].name))")
        instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
        instr.set_field_flex_param("SRC1_INDEX_ID", rd_op_str.format(IDX=0))
        instr.set_field_flex_param("SRC2_NS_ID", "op.get_operand_location(op.sources[1].name)")
        instr.set_field_flex_param("SRC2_INDEX_ID", rd_op_str.format(IDX=1))
        instructions.append(instr)

        instr = hag.get_primitive_template(op_name)
        instr.add_condition("len(op.sources) == 1")
        instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
        instr.set_field_flex_param("DST_INDEX_ID", f"op.dests[0].get_mem_index(op.get_operand_location(op.dests[0].name))")
        instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
        instr.set_field_flex_param("SRC1_INDEX_ID", rd_op_str.format(IDX=0))
        instr.set_field_flex_param("SRC2_NS_ID", "'IMM'")
        instr.set_field_value("SRC2_INDEX_ID", 0)
        instructions.append(instr)
    else:
        instr = hag.get_primitive_template("MUL")
        instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
        instr.set_field_flex_param("DST_INDEX_ID",
                                   f"op.dests[0].get_mem_index(op.get_operand_location(op.dests[0].name))")
        instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
        instr.set_field_flex_param("SRC1_INDEX_ID", rd_op_str.format(IDX=0))
        instr.set_field_flex_param("SRC2_NS_ID", "op.get_operand_location(op.sources[1].name)")
        instr.set_field_flex_param("SRC2_INDEX_ID", rd_op_str.format(IDX=1))
        instructions.append(instr)


    instructions += alu_noop(op_name, hag)

    return instructions


def base_sign_ext_gen(op_name, hag: ArchitectureNode):
    instructions = []
    # Params
    all_loop_id = f"(len(cdlt.get_ops_by_type('loop'))//2)"

    # Loops
    operand_iter = ("operand", "op.operands_by_unique_location")
    loop_iter = ('loop_op', f'cdlt.get_ops_by_type("loop")')

    # Stride calculation
    mvmt_type = f"'up' if operand.data_path[0] == 'DRAM' else 'down'"
    loop_stride = f"operand.get_offset(cdlt, 2," \
                  f"loop_op.loop_id," \
                  f"hag, movement_type={mvmt_type}, " \
                  f"outer_loop=False)"



    # Operand location string
    op_loc = f"op.get_operand_location(operand.name)"
    # Index generation
    ns_idx = f"(loop_op.loop_id % {all_loop_id}) + (operand.get_mem_index({op_loc}) * {all_loop_id}) if op.get_operand_location(operand.name) != 'IMM' else cdlt.temps.index(operand)"
    base_sign_ext = f"operand.get_mem_offset({op_loc})//(operand.dtype.bits()) if op.get_operand_location(operand.name) " \
                    f"!= 'IMM' else cdlt.temps.index(operand)"

    # Instruction generation conditions
    conds = []
    is_dependency = f'loop_op.op_str in cdlt.all_dependencies(op.dependencies)'
    outer_loop_level = f'loop_op.loop_level < op.loop_level'
    is_direct_loop_dep = f"cdlt.is_direct_loop_dep(loop_op, 'SIMD')"

    conds.append(is_dependency)
    conds.append(outer_loop_level)
    conds.append(is_direct_loop_dep)
    bitwidth = f"len(np.binary_repr({base_sign_ext})) + int(np.signbit({base_sign_ext}))"
    bitwidth_cond = f"{bitwidth} <= 16"
    single_base_ext_conds = conds + [bitwidth_cond]
    multi_base_ext_conds = conds + [f"not {bitwidth_cond}"]

    ## First, instructions for sign ext lower than 16 bits
    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.set_print_tabs(all_loop_id)
    macro_instr.add_iterable(*operand_iter)
    macro_instr.add_iterable(*loop_iter)
    macro_instr.add_condition(" and ".join(single_base_ext_conds))
    macro_instr.set_field_flex_param('NS_ID', op_loc)
    macro_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    macro_instr.set_field_flex_param('IMM', base_sign_ext)

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_print_tabs(all_loop_id)
    sub_instr.add_iterable(*operand_iter)
    sub_instr.add_iterable(*loop_iter)
    sub_instr.add_condition(" and ".join(single_base_ext_conds))
    sub_instr.set_field_flex_param('NS_ID', op_loc)
    sub_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    sub_instr.set_field_flex_param('IMM', loop_stride)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    return instructions


def alu_noop(op_name, hag):
    all_loop_id = f"(len(cdlt.get_ops_by_type('loop'))//2)"

    instructions = []
    if OP_COMPUTE_CYCLES[op_name] > 0:
        macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
        macro_instr.set_field_by_name('NS_ID', "VMEM1")
        macro_instr.set_print_tabs("op.loop_level - 1")
        macro_instr.set_field_flex_param('NS_INDEX_ID', "0")
        macro_instr.set_field_value('IMM', 0)
        #

        sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
        sub_instr.set_field_by_name('NS_ID', "VMEM1")
        sub_instr.set_print_tabs("op.loop_level - 1")
        sub_instr.set_field_flex_param('NS_INDEX_ID', "0")
        sub_instr.set_field_flex_param('IMM', f"0")
        macro_instr.add_base_instruction(sub_instr)

        noop_iters = f"{OP_COMPUTE_CYCLES[op_name]}"
        sub_instr = hag.get_primitive_template("SET_ITER")
        sub_instr.set_print_tabs("op.loop_level - 1")
        sub_instr.set_field_flex_param("LOOP_ID", f"op.loop_id % {all_loop_id}")
        sub_instr.set_field_flex_param("NUM_ITER", f"{noop_iters}")
        macro_instr.add_base_instruction(sub_instr)

        sub_instr = hag.get_primitive_template("SET_INDEX")
        sub_instr.set_print_tabs("op.loop_level + 1")
        sub_instr.set_field_by_name("DST_NS_ID", "VMEM1")
        sub_instr.set_field_flex_param("DST_INDEX_ID", "0")
        sub_instr.set_field_by_name("SRC1_NS_ID", "VMEM1")
        sub_instr.set_field_flex_param("SRC1_INDEX_ID", "0")
        sub_instr.set_field_by_name("SRC2_NS_ID", "VMEM1")
        sub_instr.set_field_flex_param("SRC2_INDEX_ID", '0')
        macro_instr.add_base_instruction(sub_instr)
        instructions.append(macro_instr)

        instr = hag.get_primitive_template("SET_INST")
        instr.set_field_flex_param("SINGLE_NESTED", "1")
        instr.set_field_flex_param("NUM_INSTR", "1")
        instructions.append(instr)

        noop_instr = hag.get_primitive_template("NOP")
        noop_instr.set_print_tabs("op.loop_level")
        instructions.append(noop_instr)
    return instructions