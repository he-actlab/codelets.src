from codelets.adl.graph import ComputeNode, StorageNode
from codelets.adl.flex_template import Instruction
from codelets.adl.backups.genesys_codelets import GENESYS_SA_CODELETS


from functools import partial

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
    instr.set_field_value("NUM_INSTR", 0)
    instructions.append(instr)

    instr = hag.get_primitive_template("INST_GROUP")
    instr.set_field_by_name("COMPUTE_TARGET", "SYSTOLIC_ARRAY")
    instr.set_field_by_name("START_END", "END")
    instr.set_field_flex_param("GROUP_NUM", "cdlt.instance_id")
    # Figure out what this is
    instr.set_field_flex_param("LOOP_ID", "max([o.loop_id for o in cdlt.ops])")
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
    instr.add_iterable('offset', f'op.transfers[("DRAM", "{buffer_name}")].src_offset')
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "offset.loop_id")
    instr.set_field_flex_param("STRIDE", "offset.stride")
    instructions.append(instr)

    instr = hag.get_primitive_template("LD_ST")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "op.loop_id")
    instr.set_field_flex_param("REQUEST_SIZE", "op.sizes[0]")
    instructions.append(instr)
    return instructions

def buffer_dram_template(buffer_name, hag):
    instructions = []
    instr = hag.get_primitive_template("SET_LOOP_STRIDE")


    instr.add_iterable('offset', f'op.transfers[("{buffer_name}", "DRAM")].src_offset')
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "offset.loop_id")
    instr.set_field_flex_param("STRIDE", "offset.stride")
    instructions.append(instr)

    # TODO: Fix loop id for this to be the minimum
    instr = hag.get_primitive_template("LD_ST")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "op.loop_id")
    instr.set_field_flex_param("REQUEST_SIZE", "op.sizes[0]")
    instructions.append(instr)
    return instructions

def buffer_sa_template(buffer_name, hag):
    instructions = []
    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.add_iterable('offset', f'op.transfers[("{buffer_name}", "pe_array")].src_offset')
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
    instr.add_iterable('offset', f'op.transfers[("pe_array", "{buffer_name}")].dst_offset')
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "WR")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_flex_param("LOOP_ID", "offset.loop_id")
    instr.set_field_flex_param("STRIDE", "offset.stride")
    instructions.append(instr)


    return instructions

def loop_template(hag):
    instructions = []
    instr = hag.get_primitive_template("SA_LOOP")
    instr.set_field_flex_param("LOOP_LEVEL", "op.loop_level")
    instr.set_field_flex_param("LOOP_ID", "op.loop_id")
    instr.set_field_flex_param("NUM_ITERATIONS", "op.iter_count")
    instructions.append(instr)
    return instr

def sa_mvmul_template(hag):
    return []

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
        }
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
    },
    "loop": loop_template,
    "compute": {
        ("pe_array", "MVMUL"): sa_mvmul_template
    }
}