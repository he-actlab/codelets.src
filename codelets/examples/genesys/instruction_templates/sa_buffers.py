from codelets.adl.graph import ComputeNode

BASE_ADDR_STR = "program.extract_bits(relocation_table.get_base_by_name({OPERAND_NAME}), {NUM_BITS}, {POS})"
SA_BASE_ADDR = "program.hag.meta_cfg['SA_BASE_ADDR']"


def ibuf_start_template(hag: ComputeNode):
    instructions = []
    if hag.meta_cfg['SINGLE_PROGRAM_COMPILATION']:
        operand = "cdlt.get_operand_by_location('IBUF')"
        base_addr = f"program.get_input_operand_offset({operand})"
    else:
        base_addr = f"{SA_BASE_ADDR}['IBUF']"
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "IBUF")
    # TODO: Fix relocation table imem value
    instr.set_field_flex_param("BASE_ADDR",
                             # f"program.extract_bits({BENCH_BASE_ADDR['IBUF']},"
                               f"program.extract_bits({base_addr}, 16, 0)",
                               lazy_eval=True)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "IBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               f"program.extract_bits({base_addr}, 16, 16)",
                               lazy_eval=True)
    instructions.append(instr)

    return instructions


def bbuf_start_template(hag: ComputeNode):
    if hag.meta_cfg['SINGLE_PROGRAM_COMPILATION']:
        operand = "cdlt.get_operand_by_location('BBUF')"
        base_addr = f"program.get_input_operand_offset({operand})"
    else:
        base_addr = f"{SA_BASE_ADDR}['BBUF']"
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "BBUF")
    # TODO: Fix relocation table imem value
    instr.set_field_flex_param("BASE_ADDR",
                               f"program.extract_bits({base_addr},"
                               " 16, 0)",
                               lazy_eval=True)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "BBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               f"program.extract_bits({base_addr},"
                               " 16, 16)",
                               lazy_eval=True)
    instructions.append(instr)

    return instructions


def obuf_start_template(hag: ComputeNode):
    if hag.meta_cfg['SINGLE_PROGRAM_COMPILATION']:
        operand = "cdlt.get_operand_by_location('OBUF')"
        base_addr_off = f"program.get_output_operand_offset({operand})"
        base_addr = f"({base_addr_off}) if cdlt.is_operand_location_used('OBUF') else 0"
    else:
        base_addr = f"{SA_BASE_ADDR}['OBUF']"
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "OBUF")
    # TODO: Fix relocation table imem value
    instr.set_field_flex_param("BASE_ADDR",
                               f"program.extract_bits({base_addr},"
                               " 16, 0)",
                               lazy_eval=True)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "OBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               f"program.extract_bits({base_addr},"
                               " 16, 16)",
                               lazy_eval=True)
    instructions.append(instr)

    return instructions


def wbuf_start_template(hag: ComputeNode):
    if hag.meta_cfg['SINGLE_PROGRAM_COMPILATION']:
        operand = "cdlt.get_operand_by_location('WBUF')"
        base_addr = f"program.get_input_operand_offset({operand})"
    else:
        base_addr = f"{SA_BASE_ADDR}['WBUF']"
    instructions = []
    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "LOW")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "WBUF")
    # TODO: Fix relocation table imem value
    instr.set_field_flex_param("BASE_ADDR",
                               f"program.extract_bits({base_addr},"
                               " 16, 0)",
                               lazy_eval=True)
    instructions.append(instr)

    instr = hag.get_primitive_template("SET_BASE_ADDR")
    instr.set_field_by_name("LOW_HIGH_ADDR", "HIGH")
    instr.set_field_by_name("MEM_TYPE", "BUFFER")
    instr.set_field_by_name("BUFFER", "WBUF")
    instr.set_field_flex_param("BASE_ADDR",
                               f"program.extract_bits({base_addr},"
                               " 16, 16)",
                               lazy_eval=True)
    instructions.append(instr)


    return instructions