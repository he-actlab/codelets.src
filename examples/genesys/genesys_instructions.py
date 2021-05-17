from codelets.instructions.instruction import Field, Instruction

OPCODE_WIDTH = 4
FUNCTION_CODE_WIDTH = 4

DTYPE_CFG_NAMES = ["32FXP", "16FXP", "8FXP", "32FP", "16FP"]

LOOP_OP_NAMES = ["SET_INDEX", "SET_ITER", "SET_INST"]

ITER_CFG_NAMES = ["BASE_SIGN_EXT", "BASE_LOW", "BASE_HIGH", "BASE_ZEROFILL", "STRIDE_SIGN_EXT", "STRIDE_LOW",
                  "STRIDE_HIGH", "STRIDE_ZEROFILL", "SET_IMM_LOW", "SET_IMM_HIGH", "IMM_SIGN_EXT"]
DTYPE_CAST_NAMES = ["32FXP_16FXP", "32FXP_8FXP", "32FXP_4FXP", "16FXP_32FXP", "8FXP_32FXP", "4FXP_32FXP",
                        "32FP_16FP", "16FP_32FP", "FLOOR", "CEIL"]
CMP_OP_NAMES = ["EQUAL", "NEQ", "GT", "GTE", "LT", "LTE"]
CALC_OP_NAMES = ["RELU", "LEAKY_RELU", "SIGMOID", "TANH", "EXP", "LN", "SQRT", "INV_SQRT", "LOG2"]
ALU_OP_NAMES = ["ADD", "SUB", "MUL", "MACC", "DIV", "MAX", "MIN", "RSHIFT", "LSHIFT", "MOVE", "COND_MOVE_TRUE",
                    "COND_MOVE_FALSE", "NOT", "AND", "OR", "NOP"]
LD_ST_NAMES = ["CONFIG_BASE_ADDR", "CONFIG_BASE_LOOP_ITER", "CONFIG_BASE_LOOP_STRIDE",
               "CONFIG_TILE_LOOP_ITER", "CONFIG_TILE_LOOP_STRIDE", "START"]
SIMD_LOOP_NAMES = ["SET_INDEX", "SET_ITER", "SET_INST"]
PERM_OP_NAMES = ["START_PERMUTE", "LOOP_INDEX"]
PLACEHOLDER_OP_NAMES = ["MULADD", "MEAN"]

ALU_OPS = (0, ALU_OP_NAMES, "ALU")
PLACEHOLDER_OPS = (0, PLACEHOLDER_OP_NAMES, "ALU")
CALC_OPS = (1, CALC_OP_NAMES, "CALCULUS")
CMP_OPS = (2, CMP_OP_NAMES, "COMPARISON")
DTYPE_CAST_OPS = (3, DTYPE_CAST_NAMES, "DTYPE_CAST")
DTYPE_CFG_OPS = (4, DTYPE_CFG_NAMES, "DTYPE_CONFIG")
LD_ST_OPS = (5, LD_ST_NAMES, "LD_ST")
ITER_CFG_OPS = (6, ITER_CFG_NAMES, "ITER_CONFIG")
SIMD_LOOP_OPS = (7, SIMD_LOOP_NAMES, "LOOP")
PERM_OPS = (8, PERM_OP_NAMES, "PERMUTATION")

# LOOP INSTR
def loop_cfg_instr():
    cfg = Field("LOOP_CFG", 6)
    cfg.set_value(0)
    loop_id = Field("LOOP_ID", 6)
    iterations = Field("NUM_ITERATIONS", 16)
    instr_temp = Instruction("SA_LOOP_CFG", 9, OPCODE_WIDTH, (cfg, loop_id, iterations))

    return instr_temp

def specific_loop_instr():
    ic_loop = Field("IC_LOOP", 6)
    ic_loop.set_value(1 << 5)
    loop_id = Field("LOOP_ID", 6)
    loop_type = Field("LOOP_TYPE", 16, value_names={"INNER": 1, "OUTER": 0})
    instr_temp = Instruction("SA_LOOP_IC", 9, OPCODE_WIDTH, (ic_loop, loop_id, loop_type))

    return instr_temp

def loop_stride_instr():
    low_high = Field("LOW_HIGH_BITS", 1, value_names={"LOW": 0, "HIGH": 1})
    access_type = Field("ACCESS_TYPE", 3, value_names={"LD": 0, "ST": 1, "RD": 2, "WR": 3})
    buffer = Field("BUFFER", 2, value_names={"WBUF": 0, "IBUF": 1, "OBUF": 2, "BBUF": 3})
    loop_id = Field("LOOP_ID", 6)
    stride = Field("STRIDE", 16)

    instr_temp = Instruction("SET_LOOP_STRIDE", 12, OPCODE_WIDTH,
                             (low_high, access_type, buffer, loop_id, stride))
    return instr_temp

# INIT/SETUP INSTR
def group_instr():
    target = Field("COMPUTE_TARGET", 1, value_names={"SYSTOLIC_ARRAY": 0, "SIMD": 1})
    start_end = Field("START_END", 1, value_names={"START": 0, "END": 1})
    group_num = Field("GROUP_NUM", 4)
    loop_id = Field("LOOP_ID", 6)
    num_instr = Field("NUM_INSTR", 16)
    instr_temp = Instruction("INST_GROUP", 10, OPCODE_WIDTH,
                             (target, start_end, group_num, loop_id, num_instr))
    return instr_temp

def block_instr():
    is_end = Field("IS_END", 16)
    instr_temp = Instruction("BLOCK_END", 11 << 12, OPCODE_WIDTH + 12, (is_end,))
    return instr_temp


# DATA TRANSFER INSTR

def base_addr_instr():
    low_high = Field("LOW_HIGH_ADDR", 1, value_names={"LOW": 0, "HIGH": 1})
    mem_type = Field("MEM_TYPE", 1, value_names={"BUFFER": 0, "IMEM": 1})
    buffer = Field("BUFFER", 4, value_names={"WBUF": 0, "IBUF": 1, "OBUF": 2, "BBUF": 3})
    null_field = Field("NULL", 6, value=0)
    base_addr = Field("BASE_ADDR", 16)
    instr_temp = Instruction("SET_BASE_ADDR", 13, OPCODE_WIDTH,
                             (low_high, mem_type, buffer, null_field, base_addr))

    return instr_temp

def load_store():
    access_type = Field("ACCESS_TYPE", 1, value_names={"LD": 0, "ST": 1})
    mem_type = Field("MEM_TYPE", 1, value_names={"BUFFER": 0, "IMEM": 1})
    buffer = Field("BUFFER", 4, value_names={"WBUF": 0, "IBUF": 1, "OBUF": 2, "BBUF": 3})
    loop_id = Field("LOOP_ID", 6)
    req_size = Field("REQUEST_SIZE", 16)
    instr_temp = Instruction("LD_ST", 14, OPCODE_WIDTH,
                             (access_type, mem_type, buffer, loop_id, req_size))
    return instr_temp


def create_simd_alu_ops(op_name, function_code):
    NS_NAMES = ["OBUF", "IBUF", "VMEM1", "IMM", "DRAM", "VMEM_RD", "VMEM_WR", "VMEM2"]
    NS_OP_CODES = {name: i for i, name in enumerate(NS_NAMES)}

    dest_ns = Field("DST_NS_ID", 3, value_names=NS_OP_CODES)
    dest_ns_idx = Field("DST_INDEX_ID", 5)

    src1_ns = Field("SRC1_NS_ID", 3, value_names=NS_OP_CODES)
    src1_ns_idx = Field("SRC1_INDEX_ID", 5)

    src2_ns = Field("SRC2_NS_ID", 3, value_names=NS_OP_CODES)
    src2_ns_idx = Field("SRC2_INDEX_ID", 5)

    instr_temp = Instruction(op_name, function_code, OPCODE_WIDTH + FUNCTION_CODE_WIDTH,
                             (dest_ns, dest_ns_idx, src1_ns, src1_ns_idx, src2_ns, src2_ns_idx))

    return instr_temp

def create_simd_calc_ops(op_name, function_code):
    NS_NAMES = ["OBUF", "IBUF", "VMEM1", "IMM", "DRAM", "VMEM_RD", "VMEM_WR", "VMEM2"]
    NS_OP_CODES = {name: i for i, name in enumerate(NS_NAMES)}

    dest_ns = Field("DST_NS_ID", 3, value_names=NS_OP_CODES)
    dest_ns_idx = Field("DST_INDEX_ID", 5)

    src1_ns = Field("SRC1_NS_ID", 3, value_names=NS_OP_CODES)
    src1_ns_idx = Field("SRC1_INDEX_ID", 5)

    src2_ns = Field("SRC2_NS_ID", 3, value_names=NS_OP_CODES)
    src2_ns_idx = Field("SRC2_INDEX_ID", 5)

    instr_temp = Instruction(op_name, function_code, OPCODE_WIDTH + FUNCTION_CODE_WIDTH,
                             (dest_ns, dest_ns_idx, src1_ns, src1_ns_idx, src2_ns, src2_ns_idx))

    return instr_temp

def create_simd_op(op_name, opcode, function_code):
    NS_NAMES = ["OBUF", "IBUF", "VMEM1", "IMM", "DRAM", "VMEM_RD", "VMEM_WR", "VMEM2"]
    NS_OP_CODES = {name: i for i, name in enumerate(NS_NAMES)}

    dest_ns = Field("DST_NS_ID", 3, value_names=NS_OP_CODES)
    dest_ns_idx = Field("DST_INDEX_ID", 5)

    src1_ns = Field("SRC1_NS_ID", 3, value_names=NS_OP_CODES)
    src1_ns_idx = Field("SRC1_INDEX_ID", 5)

    src2_ns = Field("SRC2_NS_ID", 3, value_names=NS_OP_CODES)
    src2_ns_idx = Field("SRC2_INDEX_ID", 5)

    instr_temp = Instruction(op_name, (opcode << FUNCTION_CODE_WIDTH) + function_code, OPCODE_WIDTH + FUNCTION_CODE_WIDTH,
                             (dest_ns, dest_ns_idx, src1_ns, src1_ns_idx, src2_ns, src2_ns_idx))
    return instr_temp


def create_dtype_cfg_ops():
    instructions = []
    for fn_code, fn_name in enumerate(DTYPE_CFG_NAMES):
        imm_ns_idx = Field("IMM_NS_INDEX_ID", 24)
        instr_temp = Instruction(fn_name, (4 << FUNCTION_CODE_WIDTH) + fn_code, OPCODE_WIDTH + FUNCTION_CODE_WIDTH,
                                 (imm_ns_idx,))
        instructions.append(instr_temp)
    return instructions

def create_lock_ns_op():
    NS_NAMES = ["OBUF", "IBUF", "VMEM1", "IMM", "DRAM", "VMEM_RD", "VMEM_WR", "VMEM2"]
    NS_OP_CODES = {name: i << 5 for i, name in enumerate(NS_NAMES)}
    instructions = []
    for op_code, op_name in enumerate(["LOCK_NS", "UNLOCK_NS"]):
        dest_ns = Field("DST_NS_ID", 8, value_names=NS_OP_CODES)

        src1_ns = Field("SRC1_NS_ID", 8, value_names=NS_OP_CODES)

        src2_ns = Field("SRC2_NS_ID", 8, value_names=NS_OP_CODES)

        instr = Instruction(op_name, (5 << FUNCTION_CODE_WIDTH) + op_code,
                                 OPCODE_WIDTH + FUNCTION_CODE_WIDTH,
                                 (dest_ns, src1_ns, src2_ns))
        instructions.append(instr)
    return instructions


def create_iterator_ops():

    instructions = []
    for fn_code, fn_name in enumerate(ITER_CFG_NAMES):
        ns_id = Field("NS_ID", 3)
        ns_index_id = Field("NS_INDEX_ID", 5)
        imm = Field("IMM", 16)
        instr_temp = Instruction(fn_name, (6 << FUNCTION_CODE_WIDTH) + fn_code, OPCODE_WIDTH + FUNCTION_CODE_WIDTH,
                                 (ns_id, ns_index_id, imm))
        instructions.append(instr_temp)
    return instructions

def create_simd_loop_ops():

    instructions = []

    # SET_INDEX
    loop_id = Field("LOOP_ID", 3)
    dst_idx = Field("DEST_INDEX", 5)
    src1_idx = Field("SRC1_INDEX", 8)
    src2_idx = Field("SRC2_INDEX", 8)
    instructions.append(Instruction(LOOP_OP_NAMES[0], (7 << FUNCTION_CODE_WIDTH) + 0, OPCODE_WIDTH + FUNCTION_CODE_WIDTH,
                             (loop_id, dst_idx, src1_idx, src2_idx)))

    # SET_ITER
    loop_id = Field("LOOP_ID", 3)
    num_iter = Field("NUM_ITER", 21)
    instructions.append(Instruction(LOOP_OP_NAMES[1], (7 << FUNCTION_CODE_WIDTH) + 1, OPCODE_WIDTH + FUNCTION_CODE_WIDTH,
                             (loop_id, num_iter)))

    # SET_INST
    num_instr = Field("NUM_INST", 24)
    instructions.append(Instruction(LOOP_OP_NAMES[2], (7 << FUNCTION_CODE_WIDTH) + 2, OPCODE_WIDTH + FUNCTION_CODE_WIDTH,
                             (num_instr,)))

    return instructions

def create_simd_perm_ops():
    instructions = []
    instructions.append(Instruction("START_PERMUTE", (8 << 28), 32,
                             tuple([])))
    loop_order = Field("LOOP_ORDER", 3)
    loop_idx_id = Field("LOOP_IDX_ID", 5)
    num_iters = Field("NUM_ITERS", 16)
    instructions.append(Instruction("LOOP_INDEX", (8 << FUNCTION_CODE_WIDTH) + 1, OPCODE_WIDTH + FUNCTION_CODE_WIDTH,
                                    (loop_order, loop_idx_id, num_iters)))
    return instructions

def create_placeholder_simd_instruction(op_fn):
    NS_NAMES = ["OBUF", "IBUF", "VMEM1", "IMM", "DRAM", "VMEM_RD", "VMEM_WR", "VMEM2"]
    NS_OP_CODES = {name: i for i, name in enumerate(NS_NAMES)}
    fn_code = len(ALU_OP_NAMES)

    op_code = ALU_OPS[0]
    op_type = ALU_OPS[2]
    dest_ns = Field("DST_NS_ID", 3, value_names=NS_OP_CODES)
    dest_ns_idx = Field("DST_INDEX_ID", 5)
    src1_ns = Field("SRC1_NS_ID", 3, value_names=NS_OP_CODES)
    src1_ns_idx = Field("SRC1_INDEX_ID", 5)
    src2_ns = Field("SRC2_NS_ID", 3, value_names=NS_OP_CODES)
    src2_ns_idx = Field("SRC2_INDEX_ID", 5)
    if op_type == "DTYPE_CAST":
        src2_ns.set_value_by_string("IMM")
    op_fn_code = (op_code << FUNCTION_CODE_WIDTH) + fn_code
    op_fn_code_width = OPCODE_WIDTH + FUNCTION_CODE_WIDTH
    instr_fields = (dest_ns, dest_ns_idx, src1_ns, src1_ns_idx, src2_ns, src2_ns_idx)
    instr = Instruction(op_fn, op_fn_code, op_fn_code_width, instr_fields)
    return instr

def create_simd_ops():
    instructions = []

    NS_NAMES = ["OBUF", "IBUF", "VMEM1", "IMM", "DRAM", "VMEM_RD", "VMEM_WR", "VMEM2"]
    NS_OP_CODES = {name: i for i, name in enumerate(NS_NAMES)}

    # TODO: Ask soroush about src2 in calculus ops
    for op_type_list in [ALU_OPS, CALC_OPS, CMP_OPS, DTYPE_CAST_OPS]:
        op_code = op_type_list[0]
        op_fnctions = op_type_list[1]
        op_type = op_type_list[2]
        for fn_code, op_fn in enumerate(op_fnctions):
            dest_ns = Field("DST_NS_ID", 3, value_names=NS_OP_CODES)
            dest_ns_idx = Field("DST_INDEX_ID", 5)

            src1_ns = Field("SRC1_NS_ID", 3, value_names=NS_OP_CODES)
            src1_ns_idx = Field("SRC1_INDEX_ID", 5)
            src2_ns = Field("SRC2_NS_ID", 3, value_names=NS_OP_CODES)
            src2_ns_idx = Field("SRC2_INDEX_ID", 5)
            if op_type == "DTYPE_CAST":
                src2_ns.set_value_by_string("IMM")
            op_fn_code = (op_code << FUNCTION_CODE_WIDTH) + fn_code
            op_fn_code_width = OPCODE_WIDTH + FUNCTION_CODE_WIDTH
            instr_fields = (dest_ns, dest_ns_idx, src1_ns, src1_ns_idx, src2_ns, src2_ns_idx)
            instr = Instruction(op_fn, op_fn_code, op_fn_code_width, instr_fields)
            instructions.append(instr)

    for op_type_list in [DTYPE_CFG_OPS]:
        op_code = op_type_list[0]
        op_fnctions = op_type_list[1]
        op_type = op_type_list[2]
        for fn_code, op_fn in enumerate(op_fnctions):
            null_ns = Field("NULL_NS_ID", 3)
            null_ns.set_value(0)
            null_ns_idx = Field("NULL_INDEX_ID", 5)
            null_ns_idx.set_value(0)
            imm_ns_idx = Field("IMM_NS_INDEX_ID", 16)
            op_fn_code = (op_code << FUNCTION_CODE_WIDTH) + fn_code
            op_fn_code_width = OPCODE_WIDTH + FUNCTION_CODE_WIDTH
            instr_fields = (null_ns, null_ns_idx, imm_ns_idx)
            instr_temp = Instruction(op_fn, op_fn_code, op_fn_code_width, instr_fields)
            instructions.append(instr_temp)

    LD_ST_NS = {"VMEM1": 1, "VMEM2": 2, "OBUF": 3}
    for op_type_list in [LD_ST_OPS]:
        op_code = op_type_list[0]
        op_fnctions = op_type_list[1]
        op_type = op_type_list[2]
        for msb_bit, ld_st_type in enumerate(["LD", "ST"]):
            for fn_code_base, op_fn_base in enumerate(op_fnctions):
                fn_code = (msb_bit << 3) + fn_code_base
                op_fn = f"{ld_st_type}_{op_fn_base}"
                fields = []
                lsb_msb = Field("LSB_MSB", 1, value_names={"LSB": 0, "MSB": 1})
                if op_fn_base != "CONFIG_BASE_ADDR":
                    lsb_msb.set_value(0)
                fields.append(lsb_msb)

                ns_id = Field("NS_ID", 2, value_names=LD_ST_NS)
                fields.append(ns_id)

                if op_fn_base == "START":
                     ld_data_width = Field(f"{ld_st_type}_DATA_WIDTH", 5)
                     fields.append(ld_data_width)
                     imm_field = Field("REQUEST_SIZE", 16)
                     fields.append(imm_field)
                else:
                    loop_index_id = Field(f"LOOP_INDEX_ID", 5)
                    fields.append(loop_index_id)

                    if op_fn_base == "CONFIG_BASE_ADDR":
                        imm_name = "BASE_ADDR"
                    elif op_fn_base[-4:] == "ITER":
                        imm_name = "NUM_ITERS"
                    else:
                        assert op_fn_base[-6:] == "STRIDE", f"Came across incorrect function: {op_fn_base}"
                        imm_name = "STRIDE"
                    imm_field = Field(imm_name, 16)
                    fields.append(imm_field)

                op_fn_code = (op_code << FUNCTION_CODE_WIDTH) + fn_code
                op_fn_code_width = OPCODE_WIDTH + FUNCTION_CODE_WIDTH
                instr_fields = tuple(fields)
                instr_temp = Instruction(op_fn, op_fn_code, op_fn_code_width, instr_fields)
                instructions.append(instr_temp)

    for op_type_list in [ITER_CFG_OPS]:
        op_code = op_type_list[0]
        op_fnctions = op_type_list[1]
        op_type = op_type_list[2]
        for fn_code, op_fn in enumerate(op_fnctions):
            ns_id = Field("NS_ID", 3, value_names=NS_OP_CODES)
            ns_index_id = Field("NS_INDEX_ID", 5)
            imm = Field("IMM", 16)

            op_fn_code = (op_code << FUNCTION_CODE_WIDTH) + fn_code
            op_fn_code_width = OPCODE_WIDTH + FUNCTION_CODE_WIDTH
            instr_fields = (ns_id, ns_index_id, imm)
            instr_temp = Instruction(op_fn, op_fn_code, op_fn_code_width, instr_fields)
            instructions.append(instr_temp)


    for op_type_list in [SIMD_LOOP_OPS]:
        op_code = op_type_list[0]
        op_fnctions = op_type_list[1]
        op_type = op_type_list[2]
        for fn_code, op_fn in enumerate(op_fnctions):
            if op_fn == "SET_INDEX":
                dest_ns = Field("DST_NS_ID", 3, value_names=NS_OP_CODES)
                dest_ns_idx = Field("DST_INDEX_ID", 5)

                src1_ns = Field("SRC1_NS_ID", 3, value_names=NS_OP_CODES)
                src1_ns_idx = Field("SRC1_INDEX_ID", 5)
                src2_ns = Field("SRC2_NS_ID", 3, value_names=NS_OP_CODES)
                src2_ns_idx = Field("SRC2_INDEX_ID", 5)
                instr_fields = (dest_ns, dest_ns_idx, src1_ns, src1_ns_idx, src2_ns, src2_ns_idx)

            elif op_fn == "SET_ITER":
                loop_id = Field("LOOP_ID", 3)
                null_ns = Field("NULL_NS", 5)
                null_ns.set_value(0)
                num_iter = Field("NUM_ITER", 16)
                instr_fields = (loop_id, null_ns, num_iter)

            else:
                assert op_fn == "SET_INST"
                single_nested = Field("SINGLE_NESTED", 3, value_names={"SINGLE": 0, "NESTED": 1})
                single_nested.set_value(0)
                null_ns = Field("NULL_NS", 5)
                null_ns.set_value(0)
                num_instr = Field("NUM_INSTR", 16)
                instr_fields = (single_nested, null_ns, num_instr)

            op_fn_code = (op_code << FUNCTION_CODE_WIDTH) + fn_code
            op_fn_code_width = OPCODE_WIDTH + FUNCTION_CODE_WIDTH
            instr = Instruction(op_fn, op_fn_code, op_fn_code_width, instr_fields)
            instructions.append(instr)

    for op_type_list in [PERM_OPS]:
        op_code = op_type_list[0]
        op_fnctions = op_type_list[1]
        op_type = op_type_list[2]
        for fn_code, op_fn in enumerate(op_fnctions):
            if op_fn == "START_PERMUTE":
                null_field = Field("NULL", 24)
                null_field.set_value(0)
                instr_fields = (null_field,)
            else:
                assert op_fn == "LOOP_INDEX"
                loop_order = Field("LOOP_ORDER", 3)
                loop_idx_id = Field("LOOP_IDX_ID", 5)
                num_iters = Field("NUM_ITERS", 16)
                instr_fields = (loop_order, loop_idx_id, num_iters)
            op_fn_code = (op_code << FUNCTION_CODE_WIDTH) + fn_code
            op_fn_code_width = OPCODE_WIDTH + FUNCTION_CODE_WIDTH
            instr = Instruction(op_fn, op_fn_code, op_fn_code_width, instr_fields)
            instructions.append(instr)
    for op_type_list in [PERM_OPS]:
        op_code = op_type_list[0]
        op_fnctions = op_type_list[1]
        op_type = op_type_list[2]
        for fn_code, op_fn in enumerate(op_fnctions):
            dest_ns = Field("DST_NS_ID", 3, value_names=NS_OP_CODES)
            dest_ns_idx = Field("DST_INDEX_ID", 5)

            src1_ns = Field("SRC1_NS_ID", 3, value_names=NS_OP_CODES)
            src1_ns_idx = Field("SRC1_INDEX_ID", 5)
            src2_ns = Field("SRC2_NS_ID", 3, value_names=NS_OP_CODES)
            src2_ns_idx = Field("SRC2_INDEX_ID", 5)
            if op_type == "DTYPE_CAST":
                src2_ns.set_value_by_string("IMM")
            op_fn_code = (op_code << FUNCTION_CODE_WIDTH) + fn_code + len(ALU_OP_NAMES)
            op_fn_code_width = OPCODE_WIDTH + FUNCTION_CODE_WIDTH
            instr_fields = (dest_ns, dest_ns_idx, src1_ns, src1_ns_idx, src2_ns, src2_ns_idx)
            instr = Instruction(op_fn, op_fn_code, op_fn_code_width, instr_fields)
            instructions.append(instr)
    return instructions


GENESYS_INSTRUCTIONS = {
    "systolic_array": [specific_loop_instr(), loop_cfg_instr(), loop_stride_instr(), group_instr(), base_addr_instr(), block_instr(), load_store()],
    "SIMD": create_simd_ops()
}