from codelets.adl.graph import ComputeNode

BASE_ADDR_STR = "program.extract_bits(relocation_table.get_base_by_name({OPERAND_NAME}), {NUM_BITS}, {POS})"
SA_BASE_ADDR = "program.hag.meta_cfg['SA_BASE_ADDR']"


def ibuf_start_template(hag: ComputeNode):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "IBUF")
    # TODO: Fix relocation table imem value
    instr.set_field_flex_param("BASE_ADDR",
                             # f"program.extract_bits({BENCH_BASE_ADDR['IBUF']},"
                               f"program.extract_bits({SA_BASE_ADDR}['IBUF'],"
                               " 16, 0)",
                               lazy_eval=True)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "IBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               f"program.extract_bits({SA_BASE_ADDR}['IBUF'],"
                               " 16, 16)",
                               lazy_eval=True)
    instructions.append(instr)

    return instructions


def bbuf_start_template(hag: ComputeNode):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "BBUF")
    # TODO: Fix relocation table imem value
    instr.set_field_flex_param("BASE_ADDR",
                               f"program.extract_bits({SA_BASE_ADDR}['BBUF'],"
                               " 16, 0)",
                               lazy_eval=True)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "BBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               f"program.extract_bits({SA_BASE_ADDR}['BBUF'],"
                               " 16, 16)",
                               lazy_eval=True)
    instructions.append(instr)

    return instructions


def obuf_start_template(hag: ComputeNode):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "OBUF")
    # TODO: Fix relocation table imem value
    instr.set_field_flex_param("BASE_ADDR",
                               f"program.extract_bits({SA_BASE_ADDR}['OBUF'],"
                               " 16, 0)",
                               lazy_eval=True)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "OBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               f"program.extract_bits({SA_BASE_ADDR}['OBUF'],"
                               " 16, 16)",
                               lazy_eval=True)
    instructions.append(instr)

    return instructions


def wbuf_start_template(hag: ComputeNode):
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "WBUF")
    # TODO: Fix relocation table imem value
    instr.set_field_flex_param("BASE_ADDR",
                               f"program.extract_bits({SA_BASE_ADDR}['WBUF'],"
                               " 16, 0)",
                               lazy_eval=True)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "WBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               f"program.extract_bits({SA_BASE_ADDR}['WBUF'],"
                               " 16, 16)",
                               lazy_eval=True)
    instructions.append(instr)


    return instructions