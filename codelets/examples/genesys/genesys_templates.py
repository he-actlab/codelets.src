from codelets.adl.graph import ComputeNode, StorageNode
from codelets.adl.flex_template import Instruction
from .genesys_instructions import DTYPE_CFG_NAMES, LOOP_OP_NAMES, ITER_CFG_NAMES, DTYPE_CASE_NAMES, \
    CMP_OP_NAMES, CALC_OP_NAMES, ALU_OP_NAMES

from functools import partial
BUFFER_ID_MAP = {'LD': {'IBUF': 0, 'WBUF': 1, 'OBUF': 2, 'BBUF': 3},
                 'ST': {'IBUF': 4, 'WBUF': 5, 'OBUF': 6, 'BBUF': 7},
                 }

def buffer_sa_template(buffer_name, hag):
    instructions = []
    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('offset', f'op.get_src_offset("{buffer_name}", "pe_array")')
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "RD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "offset.loop_id")
    instr.set_field_flex_param("STRIDE", "offset.stride")
    instructions.append(instr)


    return instructions

def sa_buffer_template(buffer_name, hag):
    instructions = []
    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('offset', f'op.get_dst_offset("pe_array", "{buffer_name}")')
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "WR")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "offset.loop_id")
    instr.set_field_flex_param("STRIDE", "offset.stride")
    instructions.append(instr)

    return instructions

def sa_end_template(hag: ComputeNode):
    #TODO: Add conditional block end instruction
    instructions = []
    return instructions

def sa_start_template(hag: ComputeNode):

    instructions = []
    instr = hag.get_primitive_template("INST_GROUP")
    instr.set_field_by_name("COMPUTE_TARGET", "SYSTOLIC_ARRAY")
    instr.set_field_by_name("START_END", "START")
    instr.set_field_flex_param("GROUP_NUM", "cdlt.instance_id")
    # Figure out what this is
    instr.set_field_value("LOOP_ID", 0)
    # instr.set_field_value("NUM_INSTR", 0)
    instr.set_field_flex_param("NUM_INSTR", "cdlt.num_instr")
    instructions.append(instr)

    instr = hag.get_primitive_template("INST_GROUP")
    instr.set_field_by_name("COMPUTE_TARGET", "SYSTOLIC_ARRAY")
    instr.set_field_by_name("START_END", "END")
    instr.set_field_flex_param("GROUP_NUM", "cdlt.instance_id")
    # Figure out what this is
    instr.set_field_flex_param("LOOP_ID", "max([expr.loop_id for expr in cdlt.ops])")
    instr.set_field_value("NUM_INSTR", 0)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "IMEM")
    instr.set_field_by_name("BUFFER", "IBUF")
    # TODO: Fix relocation table imem value
    instr.set_field_flex_param("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.instr_mem[cdlt.instance_id].start, 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "IMEM")
    instr.set_field_by_name("BUFFER", "IBUF")
    instr.set_field_flex_param("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.instr_mem[cdlt.instance_id].start, 16, 16)")
    instructions.append(instr)

    instr = hag.get_primitive_template("LD_ST")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("MEM_TYPE", "IMEM")
    instr.set_field_by_name("BUFFER", "IBUF")
    instr.set_field_value("LOOP_ID", 0)
    instr.set_field_flex_param("REQUEST_SIZE", "cdlt.num_instr")
    instructions.append(instr)

    instr = hag.get_primitive_template("BLOCK_END")
    # TODO: Make sure this is evaluated after having gone through all codelets
    instr.set_field_flex_param("IS_END", "int(program.codelets[-1].instance_id == cdlt.instance_id)")
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
                             "hag.util_fns.extract_bits(relocation_table.state[cdlt.inputs[1].node_name].start, 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "WBUF")
    instr.set_field_flex_param("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.state[cdlt.inputs[1].node_name].start, 16, 16)")
    instructions.append(instr)
    return instructions

def imm_start_template(hag: ComputeNode):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "WBUF")
    instr.set_field_flex_param("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.state[cdlt.inputs[1].node_name].start, 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "WBUF")
    instr.set_field_flex_param("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.state[cdlt.inputs[1].node_name].start, 16, 16)")
    instructions.append(instr)
    return instructions

def imm_end_template(hag: ComputeNode):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "WBUF")
    instr.set_field_flex_param("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.state[cdlt.inputs[1].node_name].start, 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "WBUF")
    instr.set_field_flex_param("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.state[cdlt.inputs[1].node_name].start, 16, 16)")
    instructions.append(instr)
    return instructions

def ibuf_start_template(hag: ComputeNode):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "IBUF")
    instr.set_field_flex_param("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.intermediate[cdlt.inputs[0].node_name].start, 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "IBUF")
    instr.set_field_flex_param("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.intermediate[cdlt.inputs[0].node_name].start, 16, 16)")
    instructions.append(instr)
    return instructions

def bbuf_start_template(hag: ComputeNode):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "BBUF")
    instr.set_field_flex_param("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.state[cdlt.inputs[2].node_name].start, 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "BBUF")
    instr.set_field_flex_param("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.state[cdlt.inputs[2].node_name].start, 16, 16)")
    instructions.append(instr)
    return instructions

def obuf_start_template(hag: ComputeNode):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "OBUF")
    instr.set_field_flex_param("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.intermediate[cdlt.outputs[0].node_name].start, 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "OBUF")
    instr.set_field_flex_param("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.intermediate[cdlt.outputs[0].node_name].start, 16, 16)")
    instructions.append(instr)
    return instructions

def dram_buffer_template(buffer_name, hag: ComputeNode):
    instructions = []
    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('offset', f'op.get_src_offset("DRAM", "{buffer_name}")')
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "offset.loop_id")
    # instr.set_field_flex_param("STRIDE", "offset.stride")
    instr.set_field_flex_param("STRIDE", "hag.util_fns.extract_bits(offset.stride, 16, 0)")

    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('offset', f'op.get_src_offset("DRAM", "{buffer_name}")')
    instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "offset.loop_id")
    instr.set_field_flex_param("STRIDE", "hag.util_fns.extract_bits(offset.stride, 16, 16)")
    # instr.set_field_flex_param("STRIDE", "offset.stride")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('loop_dep', f'cdlt.domain_loop_map[0].keys()')
    instr.add_condition(f'loop_dep not in op.operand.dependencies and "loop" in loop_dep')
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "cdlt.op_map[loop_dep].loop_id")
    instr.set_field_flex_param("STRIDE", "0")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('loop_dep', f'cdlt.domain_loop_map[0].keys()')
    instr.add_condition(f'loop_dep not in op.operand.dependencies and "loop" in loop_dep')
    instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "cdlt.op_map[loop_dep].loop_id")
    instr.set_field_flex_param("STRIDE", "0")
    instructions.append(instr)

    ### Updates to load in data
    loop_id_str = f"cdlt.op_id_counters['loop'] + {BUFFER_ID_MAP['LD'][buffer_name]}"
    # TODO: Change this back to non-integer
    req_size_str = f"int(np.ceil(hag.get_subgraph_edge('DRAM', '{buffer_name}').bandwidth / " \
                   f"(op.operand.dtype.bytes() * hag.get_subgraph_node('{buffer_name}').banks)))"
    n_iter_str = f"int(op.data_transfer_sizes[-1] / ({req_size_str})/ hag.get_subgraph_node('{buffer_name}').banks)"
    instr = hag.get_primitive_template("SA_LOOP")
    instr.set_field_flex_param("LOOP_LEVEL", "op.loop_level")
    instr.set_field_flex_param("LOOP_ID", loop_id_str)
    instr.set_field_flex_param("NUM_ITERATIONS", n_iter_str)
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

def buffer_dram_template(buffer_name, hag):
    instructions = []
    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('offset', f'op.get_dst_offset("{buffer_name}", "DRAM")')
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "offset.loop_id")
    # instr.set_field_flex_param("STRIDE", "offset.stride")
    instr.set_field_flex_param("STRIDE", "hag.util_fns.extract_bits(offset.stride, 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('offset', f'op.get_dst_offset("{buffer_name}", "DRAM")')
    instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "offset.loop_id")
    # instr.set_field_flex_param("STRIDE", "offset.stride")
    instr.set_field_flex_param("STRIDE", "hag.util_fns.extract_bits(offset.stride, 16, 16)")
    instructions.append(instr)


    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('loop_dep', f'cdlt.domain_loop_map[0].keys()')
    instr.add_condition(f'loop_dep not in op.operand.dependencies and "loop" in loop_dep')
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "cdlt.op_map[loop_dep].loop_id")
    instr.set_field_flex_param("STRIDE", "0")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('loop_dep', f'cdlt.domain_loop_map[0].keys()')
    instr.add_condition(f'loop_dep not in op.operand.dependencies and "loop" in loop_dep')
    instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "cdlt.op_map[loop_dep].loop_id")
    instr.set_field_flex_param("STRIDE", "0")
    instructions.append(instr)

    ### Loading tile data iteratively
    loop_id_str = f"cdlt.op_id_counters['loop'] + {BUFFER_ID_MAP['ST'][buffer_name]}"
    req_size_str = f"int(np.ceil(hag.get_subgraph_edge('{buffer_name}', 'DRAM').bandwidth / " \
                   f"(op.operand.dtype.bytes() * hag.get_subgraph_node('{buffer_name}').banks)))"
    # TODO: Change this back to non-integer
    n_iter_str = f"int(op.data_transfer_sizes[-1] / ({req_size_str})/ hag.get_subgraph_node('{buffer_name}').banks)"
    instr = hag.get_primitive_template("SA_LOOP")
    instr.set_field_flex_param("LOOP_LEVEL", "op.loop_level")
    instr.set_field_flex_param("LOOP_ID", loop_id_str)
    instr.set_field_flex_param("NUM_ITERATIONS", n_iter_str)
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


def buffer_sa_template_compute(operand_name, buffer_name, hag):
    instructions = []
    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('offset', f'op.get_offset("{operand_name}")')
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "RD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "offset.loop_id")
    instr.set_field_flex_param("STRIDE", "hag.util_fns.extract_bits(offset.stride, 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('offset', f'op.get_offset("{operand_name}")')
    instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    instr.set_field_by_name("ACCESS_TYPE", "RD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "offset.loop_id")
    instr.set_field_flex_param("STRIDE", "hag.util_fns.extract_bits(offset.stride, 16, 16)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('loop_dep', f'op.dependencies')
    instr.add_condition(f'loop_dep not in op.get_offset_loops("{operand_name}") and "loop" in loop_dep')
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "RD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "cdlt.op_map[loop_dep].loop_id")
    instr.set_field_flex_param("STRIDE", "0")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('loop_dep', f'op.dependencies')
    instr.add_condition(f'loop_dep not in op.get_offset_loops("{operand_name}") and "loop" in loop_dep')
    instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    instr.set_field_by_name("ACCESS_TYPE", "RD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "cdlt.op_map[loop_dep].loop_id")
    instr.set_field_flex_param("STRIDE", "0")
    instructions.append(instr)

    return instructions

def sa_buffer_template_compute(operand_name, buffer_name, hag):
    instructions = []
    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('offset', f'op.get_offset("{operand_name}")')
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "WR")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "offset.loop_id")
    instr.set_field_flex_param("STRIDE", "hag.util_fns.extract_bits(offset.stride, 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('offset', f'op.get_offset("{operand_name}")')
    instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    instr.set_field_by_name("ACCESS_TYPE", "WR")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "offset.loop_id")
    instr.set_field_flex_param("STRIDE", "hag.util_fns.extract_bits(offset.stride, 16, 16)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('loop_dep', f'op.dependencies')
    instr.add_condition(f'loop_dep not in op.get_offset_loops("{operand_name}") and "loop" in loop_dep')
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "WR")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "cdlt.op_map[loop_dep].loop_id")
    instr.set_field_flex_param("STRIDE", "0")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('loop_dep', f'op.dependencies')
    instr.add_condition(f'loop_dep not in op.get_offset_loops("{operand_name}") and "loop" in loop_dep')
    instr.set_field_by_name("LOW_HIGH_BITS", "HIGH")
    instr.set_field_by_name("ACCESS_TYPE", "WR")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "cdlt.op_map[loop_dep].loop_id")
    instr.set_field_flex_param("STRIDE", "0")
    instructions.append(instr)

    return instructions

def loop_template(hag):
    instructions = []
    instr = hag.get_primitive_template("SA_LOOP")
    instr.add_condition(f'cdlt.is_loop_node_target(op, "pe_array")')
    instr.set_field_flex_param("LOOP_LEVEL", "op.loop_level")
    instr.set_field_flex_param("LOOP_ID", "op.loop_id")
    instr.set_field_flex_param("NUM_ITERATIONS", "op.iter_count")
    instructions.append(instr)

    # instr = hag.get_primitive_template("SA_LOOP")
    # instr.add_condition(f'cdlt.is_loop_node_target(op, "pe_array")')
    # instr.set_field_flex_param("LOOP_LEVEL", "op.loop_level")
    # instr.set_field_flex_param("LOOP_ID", "op.loop_id")
    # instr.set_field_flex_param("NUM_ITERATIONS", "op.iter_count")
    # instructions.append(instr)
    return instr

def simd_alu_template(op_name, hag):
    instructions = []
    instr = hag.get_primitive_template(op_name)
    instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
    instr.set_field_value("DST_INDEX_ID", 0)

    instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
    instr.set_field_value("SRC1_INDEX_ID", 0)

    instr.set_field_flex_param("SRC2_NS_ID", "op.get_operand_location(op.sources[1].name)")
    instr.set_field_value("SRC2_INDEX_ID", 0)
    instructions.append(instr)

    return instructions


def dram_simd_template(mem_name, hag):
    instructions = []

    if mem_name == "VMEM1":
        inp_idx = 0
    else:
        assert mem_name == "VMEM2"
        inp_idx = 1

    instr = hag.get_primitive_template("LD_CONFIG_BASE_ADDR")
    instr.set_field_by_name("LSB_MSB", "LSB")
    instr.set_field_by_name("NS_ID", mem_name)
    instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")

    instr.set_field_flex_param("BASE_ADDR", f"hag.util_fns.extract_bits(relocation_table.intermediate[cdlt.inputs[{inp_idx}].node_name].start, 16, 0)")

    instr = hag.get_primitive_template("LD_CONFIG_BASE_ADDR")
    instr.set_field_by_name("LSB_MSB", "MSB")
    instr.set_field_by_name("NS_ID", mem_name)
    instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    instr.set_field_flex_param("BASE_ADDR", f"hag.util_fns.extract_bits(relocation_table.intermediate[cdlt.inputs[{inp_idx}].node_name].start, 16, 16)")

    # instr = hag.get_primitive_template("LD_CONFIG_BASE_LOOP_ITER")
    # instr.set_field_by_name("NS_ID", mem_name)
    # instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    # instr.set_field_flex_param("NUM_ITERS", f"op.loop_id")
    # instr.set_field_by_name("LOOP_INDEX_ID", )
    return instructions

def simd_dram_template(mem_name, hag):
    instructions = []

    instr = hag.get_primitive_template("ST_CONFIG_BASE_ADDR")
    instr.set_field_by_name("LSB_MSB", "LSB")
    instr.set_field_by_name("NS_ID", mem_name)
    instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")

    instr.set_field_flex_param("BASE_ADDR",
                               f"hag.util_fns.extract_bits(relocation_table.intermediate[cdlt.outputs[0].node_name].start, 16, 0)")

    instr = hag.get_primitive_template("ST_CONFIG_BASE_ADDR")
    instr.set_field_by_name("LSB_MSB", "MSB")
    instr.set_field_by_name("NS_ID", mem_name)
    instr.set_field_flex_param("LOOP_INDEX_ID", f"op.loop_id")
    instr.set_field_flex_param("BASE_ADDR",
                               f"hag.util_fns.extract_bits(relocation_table.intermediate[cdlt.outputs[0].node_name].start, 16, 16)")

    instructions.append(instr)
    return instructions

def sa_mvmul_template(hag):
    instructions = []
    buffers = [('weight', 'WBUF'), ('data','IBUF'), ('bias', 'BBUF'), ('out', 'OBUF')]
    for b in buffers:
        instructions += buffer_sa_template_compute(*b, hag)
        if b[1] == 'OBUF':
            instructions += sa_buffer_template_compute(*b, hag)

    return instructions

GENESYS_TEMPLATES = {
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
        ("WBUF", "pe_array"): partial(buffer_sa_template, "WBUF"),
        ("IBUF", "pe_array"): partial(buffer_sa_template, "IBUF"),
        ("BBUF", "pe_array"): partial(buffer_sa_template, "BBUF"),
        ("OBUF", "pe_array"): partial(buffer_sa_template, "OBUF"),
        ("pe_array", "OBUF"): partial(sa_buffer_template, "OBUF"),
        ("pe_array", "IBUF"): partial(sa_buffer_template, "IBUF"),
        ("pe_array", "WBUF"): partial(sa_buffer_template, "WBUF"),
        ("pe_array", "BBUF"): partial(sa_buffer_template, "BBUF"),
        ("DRAM", "VMEM1"): partial(dram_simd_template, "VMEM1"),
        ("DRAM", "VMEM2"): partial(dram_simd_template, "VMEM2"),
        ("VMEM1", "DRAM"): partial(simd_dram_template, "VMEM1"),
        ("VMEM2", "DRAM"): partial(simd_dram_template, "VMEM2"),
    },
    "loop": loop_template,
    "compute": {
        ("pe_array", "MVMUL"): sa_mvmul_template,
        **{("SIMD", op_name): partial(simd_alu_template, op_name) for op_name in ALU_OP_NAMES}
    }
}