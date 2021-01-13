from codelets.adl import ComputeNode, StorageNode, Instruction
from codelets.adl.backups.genesys_codelets import GENESYS_SA_CODELETS
from . import SIMD_NS, SIMD_OPCODE_BITWIDTH, OP_DTYPES, \
    OP_LOCATIONS, NS_BITWIDTH, NS_IDX_BITWIDTH

from functools import partial

def sa_end_template(hag):
    instructions = []
    return instructions

def sa_start_template(hag):

    instructions = []
    instr = hag.get_primitive_template("INST_GROUP")
    instr.set_field_by_name("COMPUTE_TARGET", "SYSTOLIC_ARRAY")
    instr.set_field_by_name("START_END", "START")
    instr.set_field_param_fn("GROUP_NUM", "cdlt.cdlt_id")
    # Figure out what this is
    instr.set_field_value("LOOP_ID", 0)
    instr.set_field_value("NUM_INSTR", 0)
    instructions.append(instr)

    instr = hag.get_primitive_template("INST_GROUP")
    instr.set_field_by_name("COMPUTE_TARGET", "SYSTOLIC_ARRAY")
    instr.set_field_by_name("START_END", "END")
    instr.set_field_param_fn("GROUP_NUM", "cdlt.cdlt_id")
    # Figure out what this is
    instr.set_field_param_fn("LOOP_ID", "max([o.loop_id for o in cdlt.ops])")
    instr.set_field_value("NUM_INSTR", 0)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "IMEM")
    instr.set_field_by_name("BUFFER", "IBUF")
    instr.set_field_param_fn("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.instr_mem[cdlt.offset_id].start, 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "IMEM")
    instr.set_field_by_name("BUFFER", "IBUF")
    instr.set_field_param_fn("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.instr_mem[cdlt.offset_id].start, 16, 16)")
    instructions.append(instr)

    instr = hag.get_primitive_template("LD_ST")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("MEM_TYPE", "IMEM")
    instr.set_field_by_name("BUFFER", "IBUF")
    instr.set_field_value("LOOP_ID", 0)
    instr.set_field_param_fn("REQUEST_SIZE", "cdlt.num_instr")
    instructions.append(instr)

    instr = hag.get_primitive_template("BLOCK_END")
    # TODO: Make sure this is evaluated after having gone through all codelets
    instr.set_field_param_fn("IS_END", "int(program.codelets[-1].cdlt_id == cdlt.cdlt_id)")
    instructions.append(instr)

    return instructions


def wbuf_start_template(hag):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "WBUF")
    instr.set_field_param_fn("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.state[cdlt.inputs[1].name].start, 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "WBUF")
    instr.set_field_param_fn("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.state[cdlt.inputs[1].name].start, 16, 16)")
    instructions.append(instr)
    return instructions

def ibuf_start_template(hag):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "IBUF")
    instr.set_field_param_fn("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.state[cdlt.inputs[0].name].start, 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "IBUF")
    instr.set_field_param_fn("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.state[cdlt.inputs[0].name].start, 16, 16)")
    instructions.append(instr)
    return instructions

def bbuf_start_template(hag):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "BBUF")
    instr.set_field_param_fn("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.state[cdlt.inputs[2].name].start, 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "BBUF")
    instr.set_field_param_fn("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.state[cdlt.inputs[2].name].start, 16, 16)")
    instructions.append(instr)
    return instructions

def obuf_start_template(hag):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "OBUF")
    instr.set_field_param_fn("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.state[cdlt.outputs[0].name].start, 16, 0)")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "OBUF")
    instr.set_field_param_fn("BASE_ADDR",
                             "hag.util_fns.extract_bits(relocation_table.state[cdlt.outputs[0].name].start, 16, 16)")
    instructions.append(instr)
    return instructions

def dram_buffer_template(buffer_name, hag):
    instructions = []
    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.src_offset[0].loop_id")
    instr.set_field_param_fn("STRIDE", "op.src_offset[0].stride")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.src_offset[1].loop_id")
    instr.set_field_param_fn("STRIDE", "op.src_offset[1].stride")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.src_offset[2].loop_id")
    instr.set_field_param_fn("STRIDE", "op.src_offset[2].stride")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.src_offset[3].loop_id")
    instr.set_field_param_fn("STRIDE", "op.src_offset[3].stride")
    instructions.append(instr)

    instr = hag.get_primitive_template("LD_ST")
    instr.set_field_by_name("ACCESS_TYPE", "LD")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.loop_id")
    instr.set_field_param_fn("REQUEST_SIZE", "op.size")
    instructions.append(instr)
    return instructions

def buffer_dram_template(buffer_name, hag):
    instructions = []
    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.src_offset[0].loop_id")
    instr.set_field_param_fn("STRIDE", "op.src_offset[0].stride")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.src_offset[1].loop_id")
    instr.set_field_param_fn("STRIDE", "op.src_offset[1].stride")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.src_offset[2].loop_id")
    instr.set_field_param_fn("STRIDE", "op.src_offset[2].stride")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.src_offset[3].loop_id")
    instr.set_field_param_fn("STRIDE", "op.src_offset[3].stride")
    instructions.append(instr)

    instr = hag.get_primitive_template("LD_ST")
    instr.set_field_by_name("ACCESS_TYPE", "ST")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.loop_id")
    instr.set_field_param_fn("REQUEST_SIZE", "op.size")
    instructions.append(instr)
    return instructions

def buffer_sa_template(buffer_name, hag):
    instructions = []
    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "RD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.src_offset[0].loop_id")
    instr.set_field_param_fn("STRIDE", "op.src_offset[0].stride")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "RD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.src_offset[1].loop_id")
    instr.set_field_param_fn("STRIDE", "op.src_offset[1].stride")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "RD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.src_offset[2].loop_id")
    instr.set_field_param_fn("STRIDE", "op.src_offset[2].stride")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "RD")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.src_offset[3].loop_id")
    instr.set_field_param_fn("STRIDE", "op.src_offset[3].stride")
    instructions.append(instr)

    return instructions

def sa_buffer_template(buffer_name, hag):
    instructions = []
    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "WR")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.src_offset[0].loop_id")
    instr.set_field_param_fn("STRIDE", "op.src_offset[0].stride")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "WR")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.src_offset[1].loop_id")
    instr.set_field_param_fn("STRIDE", "op.src_offset[1].stride")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "WR")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.src_offset[2].loop_id")
    instr.set_field_param_fn("STRIDE", "op.src_offset[2].stride")
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_LOOP_STRIDE")
    instr.set_field_by_name("LOW_HIGH_BITS", "LOW")
    instr.set_field_by_name("ACCESS_TYPE", "WR")
    instr.set_field_by_name("BUFFER", f"{buffer_name}")
    instr.set_field_param_fn("LOOP_ID", "op.src_offset[3].loop_id")
    instr.set_field_param_fn("STRIDE", "op.src_offset[3].stride")
    instructions.append(instr)

    return instructions

def loop_template(hag):
    instructions = []
    instr = hag.get_primitive_template("SA_LOOP")
    instr.set_field_param_fn("LOOP_LEVEL", "op.loop_level")
    instr.set_field_param_fn("LOOP_ID", "op.loop_id")
    instr.set_field_param_fn("NUM_ITERATIONS", "op.iter_count")
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
        ("WBUF", "systolic_array"): partial(buffer_sa_template, "WBUF"),
        ("IBUF", "systolic_array"): partial(buffer_sa_template, "IBUF"),
        ("BBUF", "systolic_array"): partial(buffer_sa_template, "BBUF"),
        ("OBUF", "systolic_array"): partial(buffer_sa_template, "OBUF"),
        ("systolic_array", "OBUF"): partial(sa_buffer_template, "OBUF"),
        ("systolic_array", "IBUF"): partial(sa_buffer_template, "IBUF"),
        ("systolic_array", "WBUF"): partial(sa_buffer_template, "WBUF"),
        ("systolic_array", "BBUF"): partial(sa_buffer_template, "BBUF"),
    },
    "loop": loop_template,
    "compute": {
        ("systolic_array", "MVMUL"): sa_mvmul_template
    }
}