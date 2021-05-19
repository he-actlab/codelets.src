from codelets.adl.graph import ComputeNode, StorageNode, ArchitectureNode
from codelets.adl.flex_template import Instruction
from .genesys_instructions import DTYPE_CFG_NAMES, LOOP_OP_NAMES, ITER_CFG_NAMES, DTYPE_CAST_NAMES, \
    CMP_OP_NAMES, CALC_OP_NAMES, ALU_OP_NAMES, PLACEHOLDER_OP_NAMES


from functools import partial
BUFFER_ID_MAP = {'LD': {'IBUF': 0, 'WBUF': 1, 'OBUF': 2, 'BBUF': 3},
                 'ST': {'IBUF': 4, 'WBUF': 5, 'OBUF': 6, 'BBUF': 7},
                 }
VMEM_ID_MAP = {'LD': {'VMEM1': 0, 'VMEM2': 1},
                 'ST': {'VMEM1': 2, 'VMEM2': 3},
            }

LOOPS_PER_LEVEL = 7

SIMD_OP_NAMES = ALU_OP_NAMES + CALC_OP_NAMES + CMP_OP_NAMES + DTYPE_CAST_NAMES

BASE_ADDR_STR = "program.extract_bits(relocation_table.get_base_by_name({OPERAND_NAME}), {NUM_BITS}, {POS})"


def placeholder_alu_template(op_name, hag):

   return []


def simd_alu_template(op_name, hag: ArchitectureNode):
    instructions = []
    loop_conditions = []
    is_dependency = f'loop_op.op_str in cdlt.all_dependencies(op.dependencies)'
    is_direct_loop_dep = f"cdlt.is_direct_loop_dep(loop_op, 'SIMD')"
    single_idx = f'(instruction.name not in ["SET_INDEX", "SET_ITER"] or operand_loc_idx + 1 == len(op.unique_operand_locations))'
    outer_loop_level = f'loop_op.loop_level <= op.loop_level'
    loop_conditions.append(single_idx)
    loop_conditions.append(outer_loop_level)
    loop_conditions.append(is_dependency)
    loop_conditions.append(is_direct_loop_dep)


    ## Each compute instruction gets its own set of loops
    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.add_iterable('loop_op', f'cdlt.get_ops_by_type("loop")')
    macro_instr.add_iterable('operand_loc_idx', f'range(len(op.unique_operand_locations))')
    macro_instr.set_field_flex_param('NS_ID', "op.unique_operand_locations[operand_loc_idx]")
    macro_instr.set_print_tabs("loop_op.loop_level")
    macro_instr.add_condition(" and ".join(loop_conditions))

    macro_instr.set_field_flex_param('NS_INDEX_ID', "loop_op.loop_id")
    macro_instr.set_field_value('IMM', 0)

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.add_iterable('loop_op', f'cdlt.get_ops_by_type("loop")')
    sub_instr.add_iterable('operand_loc_idx', f'range(len(op.unique_operand_locations))')
    sub_instr.set_field_flex_param('NS_ID', "op.unique_operand_locations[operand_loc_idx]")
    sub_instr.set_print_tabs("loop_op.loop_level")
    sub_instr.add_condition(" and ".join(loop_conditions))
    sub_instr.set_field_flex_param('NS_INDEX_ID', "loop_op.loop_id")
    sub_instr.set_field_flex_param('IMM', "loop_op.stride")
    macro_instr.add_base_instruction(sub_instr)
    #
    # SET_INDEX only needs to execute once per loop
    sub_instr = hag.get_primitive_template("SET_INDEX")
    sub_instr.add_iterable('loop_op', f'cdlt.get_ops_by_type("loop")')
    sub_instr.add_iterable('operand_loc_idx', f'range(len(op.unique_operand_locations))')
    sub_instr.set_print_tabs("loop_op.loop_level")
    sub_instr.add_condition(" and ".join(loop_conditions))
    sub_instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
    sub_instr.set_field_flex_param("DST_INDEX_ID", "loop_op.loop_id")
    sub_instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID", "loop_op.loop_id")
    sub_instr.set_field_flex_param("SRC2_NS_ID",
                               "'IMM' if len(op.sources) == 1 else op.get_operand_location(op.sources[1].name)")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", 'loop_op.loop_id')
    macro_instr.add_base_instruction(sub_instr)

    # SET_ITER only needs to execute once per loop
    sub_instr = hag.get_primitive_template("SET_ITER")
    sub_instr.add_iterable('loop_op', f'cdlt.get_ops_by_type("loop")')
    sub_instr.add_iterable('operand_loc_idx', f'range(len(op.unique_operand_locations))')
    sub_instr.set_print_tabs("loop_op.loop_level")
    sub_instr.add_condition(" and ".join(loop_conditions))
    sub_instr.set_field_flex_param("LOOP_ID", "loop_op.loop_id")
    sub_instr.set_field_flex_param("NUM_ITER", "loop_op.iter_count")
    macro_instr.add_base_instruction(sub_instr)

    instructions.append(macro_instr)

    instr = hag.get_primitive_template("SET_INST")
    instr.add_condition("cdlt.op_id_counters['compute'] - 1 > op.op_id")
    instr.set_field_flex_param("SINGLE_NESTED", "0 if op.num_loop_dependencies > 0 else 1")
    instr.set_field_flex_param("NUM_INSTR", "1")
    instructions.append(instr)

    instr = hag.get_primitive_template(op_name)
    instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
    instr.set_field_value("DST_INDEX_ID", 0)
    instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
    instr.set_field_value("SRC1_INDEX_ID", 0)

    if op_name not in (DTYPE_CAST_NAMES + CALC_OP_NAMES):
        instr.set_field_flex_param("SRC2_NS_ID", "op.get_operand_location(op.sources[1].name)")
        instr.set_field_value("SRC2_INDEX_ID", 0)
    elif op_name in CALC_OP_NAMES:
        instr.set_field_by_name("SRC2_NS_ID", "IMM")
        instr.set_field_value("SRC2_INDEX_ID", 0)
    instructions.append(instr)


    return instructions

def sa_end_template(hag: ComputeNode):
    #TODO: Add conditional block end instruction
    instructions = []

    instr = hag.get_primitive_template("INST_GROUP")
    instr.set_field_by_name("COMPUTE_TARGET", "SYSTOLIC_ARRAY")
    instr.set_field_by_name("START_END", "END")
    instr.set_field_flex_param("GROUP_NUM", "cdlt.instance_id")
    # Figure out what this is
    instr.set_field_flex_param("LOOP_ID", "max([expr.loop_id for expr in cdlt.ops])")
    instr.set_field_value("NUM_INSTR", 0)
    instructions.append(instr)

    instr = hag.get_primitive_template("BLOCK_END")
    # TODO: Make sure this is evaluated after having gone through all codelets
    instr.set_field_flex_param("IS_END", "int(program.codelets[-1].instance_id == cdlt.instance_id)")
    instructions.append(instr)
    return instructions

def sa_start_template(hag: ComputeNode):

    instructions = []
    instr = hag.get_primitive_template("INST_GROUP")
    instr.set_field_by_name("COMPUTE_TARGET", "SYSTOLIC_ARRAY")
    instr.set_field_by_name("START_END", "START")
    instr.set_field_flex_param("GROUP_NUM", "cdlt.instance_id")
    # Figure out what this is
    instr.set_field_value("LOOP_ID", 0)
    instr.set_field_flex_param("NUM_INSTR", "cdlt.num_instr", lazy_eval=True)
    instructions.append(instr)


    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "IMEM")
    instr.set_field_by_name("BUFFER", "IBUF")
    # TODO: Fix relocation table imem value
    instr.set_field_flex_param("BASE_ADDR",
                             "program.extract_bits(relocation_table.get_relocation_base('INSTR_MEM', cdlt.cdlt_uid),"
                             " 16, 0)",
                               lazy_eval=True)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "IMEM")
    instr.set_field_by_name("BUFFER", "IBUF")
    instr.set_field_flex_param("BASE_ADDR",
                             "program.extract_bits(relocation_table.get_relocation_base('INSTR_MEM', cdlt.cdlt_uid),"
                             " 16, 16)",
                               lazy_eval=True)
    instructions.append(instr)

    instr = hag.get_primitive_template("LD_ST")
    instr.add_condition("cdlt.instance_id < len(program.codelets)")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("MEM_TYPE", "IMEM")
    instr.set_field_by_name("BUFFER", "IBUF")
    instr.set_field_value("LOOP_ID", 0)
    instr.set_field_flex_param("REQUEST_SIZE", "program.codelets[cdlt.instance_id].num_instr", lazy_eval=True)
    instructions.append(instr)

    return instructions

def simd_end_template(hag: ComputeNode):
    #TODO: Add conditional block end instruction
    instructions = []
    return instructions

def simd_start_template(hag: ComputeNode):

    instructions = []
    instr = hag.get_primitive_template("INST_GROUP")
    instr.set_field_by_name("COMPUTE_TARGET", "SIMD")
    instr.set_field_by_name("START_END", "START")
    instr.set_field_flex_param("GROUP_NUM", "cdlt.instance_id")
    # Figure out what this is
    instr.set_field_value("LOOP_ID", 0)
    instr.set_field_flex_param("NUM_INSTR", "cdlt.num_instr")
    instructions.append(instr)

    return instructions

def wbuf_start_template(hag: ComputeNode):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "WBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               BASE_ADDR_STR.format(OPERAND_NAME="cdlt.inputs[1].node_name", NUM_BITS="16",
                                                    POS="0"),
                               lazy_eval=True)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "WBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               BASE_ADDR_STR.format(OPERAND_NAME="cdlt.inputs[1].node_name", NUM_BITS="16",
                                                    POS="16"),
                               lazy_eval=True)
    instructions.append(instr)
    return instructions

def imm_start_template(hag: ComputeNode):
    instructions = []
    instr = hag.get_primitive_template("IMM_SIGN_EXT")
    instr.set_field_by_name("NS_ID", "IMM")
    instr.set_field_flex_param("NS_INDEX_ID", f"op.get_config_param_value('index')")
    instr.set_field_flex_param("IMM", f"op.get_config_param_value('immediate_value')")
    instructions.append(instr)

    return instructions

def imm_end_template(hag: ComputeNode):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "WBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               BASE_ADDR_STR.format(OPERAND_NAME="cdlt.inputs[1].node_name", NUM_BITS="16",
                                                    POS="0"),
                               lazy_eval=True)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "WBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               BASE_ADDR_STR.format(OPERAND_NAME="cdlt.inputs[1].node_name", NUM_BITS="16",
                                                    POS="16"),
                               lazy_eval=True)
    instructions.append(instr)
    return instructions

def ibuf_start_template(hag: ComputeNode):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "IBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               BASE_ADDR_STR.format(OPERAND_NAME="cdlt.inputs[0].node_name", NUM_BITS="16",
                                                    POS="0"),
                               lazy_eval=True)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "IBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               BASE_ADDR_STR.format(OPERAND_NAME="cdlt.inputs[0].node_name", NUM_BITS="16",
                                                    POS="16"),
                               lazy_eval=True)
    instructions.append(instr)
    return instructions

def bbuf_start_template(hag: ComputeNode):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "BBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               BASE_ADDR_STR.format(OPERAND_NAME="cdlt.inputs[2].node_name", NUM_BITS="16",
                                                    POS="0"),
                               lazy_eval=True)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "BBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               BASE_ADDR_STR.format(OPERAND_NAME="cdlt.inputs[2].node_name", NUM_BITS="16",
                                                    POS="16"),
                               lazy_eval=True)
    instructions.append(instr)
    return instructions

def obuf_start_template(hag: ComputeNode):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "OBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               BASE_ADDR_STR.format(OPERAND_NAME="cdlt.outputs[0].node_name", NUM_BITS="16", POS="0"),
                               lazy_eval=True)

    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "OBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               BASE_ADDR_STR.format(OPERAND_NAME="cdlt.outputs[0].node_name", NUM_BITS="16", POS="16"),
                               lazy_eval=True)
    instructions.append(instr)
    return instructions

def dram_buffer_template(buffer_name, hag: ComputeNode):
    instructions = []

    ### Updates to load in data
    # loop_id_str = f"(op.loop_id % {LOOPS_PER_LEVEL}) + {LOOPS_PER_LEVEL} * 2"
    loop_id_str = f"hag.util_fns.get_loop_level_id('{buffer_name}', op.loop_id, 2, 'LD')"

    # TODO: Change this back to non-integer
    req_size_str = f"int(np.ceil(hag.get_subgraph_edge('DRAM', '{buffer_name}').bandwidth / " \
                   f"(op.operand.dtype.bits() * hag.get_subgraph_node('{buffer_name}').banks)))"
    n_iter_str = f"int(op.data_transfer_sizes[-1] / ({req_size_str})/ hag.get_subgraph_node('{buffer_name}').banks)"
    req_size_str_low = f"program.extract_bits({req_size_str} << 12, 16, 0)"
    req_size_str_high = f"program.extract_bits({req_size_str} << 12, 16, 16)"
    instr = hag.get_primitive_template("SA_LOOP_CFG")
    instr.set_field_flex_param("LOOP_ID", loop_id_str)
    instr.set_field_flex_param("NUM_ITERATIONS", f"{n_iter_str} - 1")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", loop_id_str)
    instr.set_field_flex_param("STRIDE", req_size_str_low)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", loop_id_str)
    instr.set_field_flex_param("STRIDE", req_size_str_high)
    instructions.append(instr)
    ####
    # ITERS = tile_size / request_size / num_banks
    instr = hag.get_primitive_template("LD_ST")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", loop_id_str)
    instr.set_field_flex_param("REQUEST_SIZE", req_size_str)
    instructions.append(instr)
    return instructions

def buffer_dram_template(buffer_name, hag: ArchitectureNode):
    instructions = []


    ### Loading tile data iteratively
    # loop_id_str = f"(op.loop_id % {LOOPS_PER_LEVEL}) + {LOOPS_PER_LEVEL} * 2"
    loop_id_str = f"hag.util_fns.get_loop_level_id('{buffer_name}', op.loop_id, 2, 'ST')"

    req_size_str = f"int(np.ceil(hag.get_subgraph_edge('{buffer_name}', 'DRAM').bandwidth / " \
                   f"(op.operand.dtype.bits() * hag.get_subgraph_node('{buffer_name}').banks)))"
    req_size_str_low = f"program.extract_bits({req_size_str} << 12, 16, 0)"
    req_size_str_high = f"program.extract_bits({req_size_str} << 12, 16, 16)"
    # TODO: Change this back to non-integer
    n_iter_str = f"int(op.data_transfer_sizes[-1] / ({req_size_str}) / hag.get_subgraph_node('{buffer_name}').banks)"
    instr = hag.get_primitive_template("SA_LOOP_CFG")
    instr.set_field_flex_param("LOOP_ID", loop_id_str)
    instr.set_field_flex_param("NUM_ITERATIONS", f"{n_iter_str} - 1")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", loop_id_str)
    instr.set_field_flex_param("STRIDE", req_size_str_low)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", loop_id_str)
    instr.set_field_flex_param("STRIDE", req_size_str_high)
    instructions.append(instr)

    # TODO: Fix loop id for this to be the minimum
    instr = hag.get_primitive_template("LD_ST")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", loop_id_str)
    instr.set_field_flex_param("REQUEST_SIZE", req_size_str)
    instructions.append(instr)
    return instructions


def outer_simd_loops(hag: ArchitectureNode):
    instructions = []
    macro_instr = hag.get_primitive_template("LD_CONFIG_BASE_LOOP_ITER")
    macro_instr.add_iterable('operand', f'cdlt.used_inputs')
    macro_instr.add_condition(f'cdlt.is_loop_node_target(op, "SIMD") and not cdlt.is_direct_loop_dep(op, "SIMD")')
    macro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    macro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    macro_instr.set_field_flex_param("NUM_ITERS", f"op.iter_count")

    micro_instr = hag.get_primitive_template("LD_CONFIG_BASE_LOOP_STRIDE")
    micro_instr.add_iterable('operand', f'cdlt.used_inputs')
    micro_instr.add_condition(f'cdlt.is_loop_node_target(op, "SIMD") and not cdlt.is_direct_loop_dep(op, "SIMD")')
    micro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    micro_instr.set_field_flex_param("STRIDE", f"operand.get_offset(cdlt, 1, op.loop_id)")
    macro_instr.add_base_instruction(micro_instr)
    instructions.append(macro_instr)

    macro_instr = hag.get_primitive_template("LD_CONFIG_BASE_ADDR")
    macro_instr.add_iterable('operand', f'cdlt.used_inputs')
    macro_instr.add_condition(f'cdlt.is_loop_node_target(op, "SIMD") and not cdlt.is_direct_loop_dep(op, "SIMD")')
    macro_instr.set_field_by_name("LSB_MSB", "LSB")
    macro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    macro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    macro_instr.set_field_flex_param("BASE_ADDR",
                                     BASE_ADDR_STR.format(OPERAND_NAME="operand.node_name", NUM_BITS="16",
                                                          POS="0"),
                                     lazy_eval=True
                                     )

    micro_instr = hag.get_primitive_template("LD_CONFIG_BASE_ADDR")
    micro_instr.add_iterable('operand', f'cdlt.used_inputs')
    micro_instr.add_condition(f'cdlt.is_loop_node_target(op, "SIMD") and not cdlt.is_direct_loop_dep(op, "SIMD")')
    micro_instr.set_field_by_name("LSB_MSB", "MSB")
    micro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    micro_instr.set_field_flex_param("BASE_ADDR",
                                     BASE_ADDR_STR.format(OPERAND_NAME="operand.node_name", NUM_BITS="16",
                                                          POS="16"),
                                     lazy_eval=True)
    macro_instr.add_base_instruction(micro_instr)
    instructions.append(macro_instr)

    macro_instr = hag.get_primitive_template("ST_CONFIG_BASE_LOOP_ITER")
    macro_instr.add_iterable('operand', f'cdlt.used_outputs')
    macro_instr.add_condition(f'cdlt.is_loop_node_target(op, "SIMD") and not cdlt.is_direct_loop_dep(op, "SIMD")')
    macro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    macro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    macro_instr.set_field_flex_param("NUM_ITERS", f"op.iter_count")

    micro_instr = hag.get_primitive_template("ST_CONFIG_BASE_LOOP_STRIDE")
    micro_instr.add_iterable('operand', f'cdlt.used_outputs')
    micro_instr.add_condition(f'cdlt.is_loop_node_target(op, "SIMD") and not cdlt.is_direct_loop_dep(op, "SIMD")')
    micro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    micro_instr.set_field_flex_param("STRIDE", f"operand.get_offset(cdlt, 1, op.loop_id)")
    macro_instr.add_base_instruction(micro_instr)
    instructions.append(macro_instr)

    macro_instr = hag.get_primitive_template("ST_CONFIG_BASE_ADDR")
    macro_instr.add_iterable('operand', f'cdlt.used_outputs')
    macro_instr.add_condition(f'cdlt.is_loop_node_target(op, "SIMD") and not cdlt.is_direct_loop_dep(op, "SIMD")')
    macro_instr.set_field_by_name("LSB_MSB", "LSB")
    macro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    macro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    macro_instr.set_field_flex_param("BASE_ADDR",
                                     BASE_ADDR_STR.format(OPERAND_NAME="operand.node_name", NUM_BITS="16",
                                                          POS="0"),
                                     lazy_eval=True)

    micro_instr = hag.get_primitive_template("ST_CONFIG_BASE_ADDR")
    micro_instr.add_iterable('operand', f'cdlt.used_outputs')
    micro_instr.add_condition(f'cdlt.is_loop_node_target(op, "SIMD") and not cdlt.is_direct_loop_dep(op, "SIMD")')
    micro_instr.set_field_by_name("LSB_MSB", "MSB")
    micro_instr.set_field_flex_param("NS_ID", "operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    micro_instr.set_field_flex_param("BASE_ADDR",
                                     BASE_ADDR_STR.format(OPERAND_NAME="operand.node_name", NUM_BITS="16",
                                                          POS="16"),
                                     lazy_eval=True)
    macro_instr.add_base_instruction(micro_instr)
    instructions.append(macro_instr)

    return instructions

def inner_simd_loops(hag: ArchitectureNode):
    instructions = []
    return instructions

def outer_sa_loops(hag: ArchitectureNode):
    instructions = []
    loop_cond_str = 'cdlt.is_loop_node_target(op, "pe_array") and not cdlt.is_direct_loop_dep(op, "pe_array")'
    instr = hag.get_primitive_template("SA_LOOP_CFG")
    instr.add_condition(loop_cond_str)
    instr.set_field_flex_param("LOOP_ID", "op.loop_id")
    instr.set_field_flex_param("NUM_ITERATIONS", "op.iter_count - 1")
    instructions.append(instr)

    instr = hag.get_primitive_template("SA_REDUCTION_LOOP")
    instr.add_condition(f'{loop_cond_str} and cdlt.loop_param_map[op.op_str] in ["IC", "N", "KH", "KW"]')
    instr.set_field_flex_param("LOOP_DIM", "cdlt.loop_param_map[op.op_str]")
    instr.set_field_by_name("LOOP_TYPE", "OUTER")
    instr.set_field_flex_param("LOOP_ID", "op.loop_id")
    instructions.append(instr)

    macro_instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    macro_instr.add_iterable('operand', f'cdlt.operands')
    macro_instr.add_condition(loop_cond_str)
    macro_instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    macro_instr.set_field_by_name("ACCESS_TYPE", "LD")
    macro_instr.set_field_flex_param("LOOP_ID", "op.loop_id")
    macro_instr.set_field_flex_param("BUFFER", f"operand.get_ld_storage_location(cdlt, 1)")
    macro_instr.set_field_flex_param("STRIDE",
                                     "program.extract_bits(operand.get_offset(cdlt, 1, op.loop_id) << 12, 16, 0)")

    micro_instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    micro_instr.add_iterable('operand', f'cdlt.operands')
    micro_instr.add_condition(loop_cond_str)
    micro_instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    micro_instr.set_field_by_name("ACCESS_TYPE", "LD")
    micro_instr.set_field_flex_param("LOOP_ID", "op.loop_id")
    micro_instr.set_field_flex_param("BUFFER", f"operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("STRIDE",
                                     "program.extract_bits(operand.get_offset(cdlt, 1, op.loop_id) << 12, 16, 16)")
    macro_instr.add_base_instruction(micro_instr)
    instructions.append(macro_instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_condition(loop_cond_str)
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_flex_param("LOOP_ID", "op.loop_id")
    instr.set_field_flex_param("BUFFER", f"cdlt.outputs[0].get_ld_storage_location(cdlt, 1)")
    instr.set_field_flex_param("STRIDE", "program.extract_bits("
                                         "cdlt.outputs[0].get_offset(cdlt, 1, op.loop_id) << 12,"
                                         "16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_condition(loop_cond_str)
    instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_flex_param("LOOP_ID", "op.loop_id")
    instr.set_field_flex_param("BUFFER", f"cdlt.outputs[0].get_ld_storage_location(cdlt, 1)")
    instr.set_field_flex_param("STRIDE", "program.extract_bits("
                                         "cdlt.outputs[0].get_offset(cdlt, 1, op.loop_id) << 12,"
                                         "16, 16)")
    instructions.append(instr)
    return instructions

def inner_sa_loops(hag: ArchitectureNode):
    instructions = []
    inner_loop_id_str = f"(op.loop_id % {LOOPS_PER_LEVEL}) + {LOOPS_PER_LEVEL}"
    instr = hag.get_primitive_template("SA_LOOP_CFG")
    instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array")')
    instr.set_field_flex_param("LOOP_ID", inner_loop_id_str)
    sa_constraints = hag.get_subgraph_node("pe_array").dimensions[0]
    n_iter_str = f"int(np.ceil(op.iter_count / {sa_constraints})) " \
                 f"if cdlt.loop_param_map[op.op_str] in ['IC', 'OC', 'N', 'P'] else op.iter_count"
    # instr.set_field_flex_param("NUM_ITERATIONS", f"int(np.ceil(op.iter_count / {sa_constraints}))")
    instr.set_field_flex_param("NUM_ITERATIONS", f"{n_iter_str} - 1")
    instructions.append(instr)

    instr = hag.get_primitive_template("SA_REDUCTION_LOOP")
    instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array") and cdlt.loop_param_map[op.op_str] in '
                        f'["IC", "N", "KH", "KW"]')
    instr.set_field_flex_param("LOOP_DIM", "cdlt.loop_param_map[op.op_str]")
    instr.set_field_by_name("LOOP_TYPE", "INNER")
    instr.set_field_flex_param("LOOP_ID", f"op.loop_id % (cdlt.op_id_counters['loop']//2)")
    instructions.append(instr)

    macro_instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    macro_instr.add_iterable('operand', f'cdlt.operands')
    macro_instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array")')
    macro_instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    macro_instr.set_field_by_name("ACCESS_TYPE", "RD")
    macro_instr.set_field_flex_param("LOOP_ID", inner_loop_id_str)
    macro_instr.set_field_flex_param("BUFFER", f"operand.get_ld_storage_location(cdlt, 1)")
    macro_instr.set_field_flex_param("STRIDE",
                                     "program.extract_bits(operand.get_offset(cdlt, 2, op.loop_id), 16, 0)")

    micro_instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    micro_instr.add_iterable('operand', f'cdlt.operands')
    micro_instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array")')
    micro_instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    micro_instr.set_field_by_name("ACCESS_TYPE", "RD")
    micro_instr.set_field_flex_param("LOOP_ID", inner_loop_id_str)
    micro_instr.set_field_flex_param("BUFFER", f"operand.get_ld_storage_location(cdlt, 1)")
    micro_instr.set_field_flex_param("STRIDE",
                                     "program.extract_bits(operand.get_offset(cdlt, 2, op.loop_id), 16, 16)")
    macro_instr.add_base_instruction(micro_instr)
    instructions.append(macro_instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array")')
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "WR")
    instr.set_field_flex_param("LOOP_ID", inner_loop_id_str)
    instr.set_field_flex_param("BUFFER", f"cdlt.outputs[0].get_ld_storage_location(cdlt, 1)")
    instr.set_field_flex_param("STRIDE",
                               "program.extract_bits(cdlt.outputs[0].get_offset(cdlt, 2, op.loop_id), 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_condition(f'cdlt.is_direct_loop_dep(op, "pe_array")')
    instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    instr.set_field_by_name("ACCESS_TYPE", "WR")
    instr.set_field_flex_param("LOOP_ID", inner_loop_id_str)
    instr.set_field_flex_param("BUFFER", f"cdlt.outputs[0].get_ld_storage_location(cdlt, 1)")
    instr.set_field_flex_param("STRIDE",
                               "program.extract_bits(cdlt.outputs[0].get_offset(cdlt, 2, op.loop_id), 16, 16)")
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


def dram_simd_template(mem_name, hag: ArchitectureNode):
    instructions = []

    if mem_name == "VMEM1":
        inp_idx = 0
    else:
        assert mem_name == "VMEM2"
        inp_idx = 1

    ### TILE LOOP
    loop_id_str = f"cdlt.op_id_counters['loop'] + {VMEM_ID_MAP['LD'][mem_name]}"
    # TODO: Change this back to non-integer
    req_size_str = f"int(np.ceil(hag.get_subgraph_edge('DRAM', '{mem_name}').bandwidth / " \
                   f"(op.operand.dtype.bits() * hag.get_subgraph_node('{mem_name}').banks)))"
    n_iter_str = f"int(op.data_transfer_sizes[-1] / ({req_size_str})/ hag.get_subgraph_node('{mem_name}').banks)"
    # transfer size = 32
    # req_size = 1
    # banks = 32


    instr = hag.get_primitive_template("LD_CONFIG_TILE_LOOP_ITER")
    instr.set_field_by_name("NS_ID", mem_name)
    instr.set_field_flex_param("LOOP_INDEX_ID", loop_id_str)
    instr.set_field_flex_param("NUM_ITERS", n_iter_str)
    instructions.append(instr)
    ####
    # ITERS = tile_size / request_size / num_banks
    instr = hag.get_primitive_template("LD_CONFIG_TILE_LOOP_STRIDE")
    instr.set_field_by_name("NS_ID", f"{mem_name}")
    instr.set_field_flex_param("LOOP_INDEX_ID", loop_id_str)
    instr.set_field_flex_param("STRIDE", req_size_str)
    instructions.append(instr)

    instr = hag.get_primitive_template("LD_START")
    instr.set_field_by_name("NS_ID", f"{mem_name}")
    instr.set_field_flex_param("LD_DATA_WIDTH", "op.operand.dtype.bits()")
    instr.set_field_flex_param("REQUEST_SIZE", req_size_str)
    instructions.append(instr)


    return instructions

def simd_dram_template(mem_name, hag: ArchitectureNode):
    instructions = []

    ### TILE LOOP
    loop_id_str = f"cdlt.op_id_counters['loop'] + {VMEM_ID_MAP['ST'][mem_name]}"
    # TODO: Change this back to non-integer
    req_size_str = f"int(np.ceil(hag.get_subgraph_edge('{mem_name}', 'DRAM').bandwidth / " \
                   f"(op.operand.dtype.bits() * hag.get_subgraph_node('{mem_name}').banks)))"
    n_iter_str = f"int(op.data_transfer_sizes[-1] / ({req_size_str})/ hag.get_subgraph_node('{mem_name}').banks)"
    # transfer size = 32
    # req_size = 1
    # banks = 32


    instr = hag.get_primitive_template("ST_CONFIG_TILE_LOOP_ITER")
    instr.set_field_by_name("NS_ID", mem_name)
    instr.set_field_flex_param("LOOP_INDEX_ID", loop_id_str)
    instr.set_field_flex_param("NUM_ITERS", n_iter_str)
    instructions.append(instr)
    ####
    # ITERS = tile_size / request_size / num_banks
    instr = hag.get_primitive_template("ST_CONFIG_TILE_LOOP_STRIDE")
    instr.set_field_by_name("NS_ID", f"{mem_name}")
    instr.set_field_flex_param("LOOP_INDEX_ID", loop_id_str)
    instr.set_field_flex_param("STRIDE", req_size_str)
    instructions.append(instr)

    instr = hag.get_primitive_template("ST_START")
    instr.set_field_by_name("NS_ID", f"{mem_name}")
    instr.set_field_flex_param("ST_DATA_WIDTH", "op.operand.dtype.bits()")
    instr.set_field_flex_param("REQUEST_SIZE", req_size_str)
    instructions.append(instr)
    return instructions

def sa_mvmul_template(hag: ArchitectureNode):
    instructions = []
    # buffers = [('weight', 'WBUF'), ('data','IBUF'), ('bias', 'BBUF'), ('out', 'OBUF')]
    # for b in buffers:
    #     instructions += buffer_sa_template_compute(*b, hag)
    #     if b[1] == 'OBUF':
    #         instructions += sa_buffer_template_compute(*b, hag)

    return instructions

def program_start(hag: ArchitectureNode):
    instructions = []
    return instructions

def program_end(hag: ArchitectureNode):
    instructions = []
    return instructions

def codelet_start(hag: ArchitectureNode):
    instructions = []
    return instructions

def codelet_end(hag: ArchitectureNode):
    instructions = []
    return instructions

GENESYS_TEMPLATES = {
    "program": {
        "start": program_start,
        "end": program_end,
    },
    "codelet": {
        "start": codelet_start,
        "end": codelet_end,
    },
    "config": {
        "systolic_array": {
            "start": sa_start_template,
            "end": sa_end_template
        },
        "WBUF": {
            "start": wbuf_start_template,
            "end": lambda x: []
        },
        "IBUF": {
            "start": ibuf_start_template,
            "end": lambda x: []
        },
        "BBUF": {
            "start": bbuf_start_template,
            "end": lambda x: []
        },
        "OBUF": {
            "start": obuf_start_template,
            "end": lambda x: []
        },
        "SIMD": {
            "start": simd_start_template,
            "end": simd_end_template
        },
        "IMM": {
            "start": imm_start_template,
            "end": imm_end_template
        },
    },
    "transfer": {
        ("DRAM", "WBUF"): partial(dram_buffer_template, "WBUF"),
        ("DRAM", "IBUF"): partial(dram_buffer_template, "IBUF"),
        ("DRAM", "BBUF"): partial(dram_buffer_template, "BBUF"),
        ("DRAM", "OBUF"): partial(dram_buffer_template, "OBUF"),
        ("OBUF", "DRAM"): partial(buffer_dram_template, "OBUF"),
        ("IBUF", "DRAM"): partial(buffer_dram_template, "IBUF"),
        ("WBUF", "DRAM"): partial(buffer_dram_template, "WBUF"),
        ("BBUF", "DRAM"): partial(buffer_dram_template, "BBUF"),
        ("DRAM", "VMEM1"): partial(dram_simd_template, "VMEM1"),
        ("DRAM", "VMEM2"): partial(dram_simd_template, "VMEM2"),
        ("VMEM1", "DRAM"): partial(simd_dram_template, "VMEM1"),
        ("VMEM2", "DRAM"): partial(simd_dram_template, "VMEM2"),
    },
    "loop": loop_template,
    "compute": {
        ("pe_array", "MVMUL"): sa_mvmul_template,
        **{("SIMD", op_name): partial(simd_alu_template, op_name) for op_name in SIMD_OP_NAMES},
        **{("SIMD", op_name): partial(placeholder_alu_template, op_name) for op_name in PLACEHOLDER_OP_NAMES},
    }
}