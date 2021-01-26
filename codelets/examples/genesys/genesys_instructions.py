from codelets.adl.flex_template.instruction import Field, Instruction

OPCODE_WIDTH = 4
FUNCTION_CODE_WIDTH = 4



# LOOP INSTR
def loop_instr():
    llevel = Field("LOOP_LEVEL", 6)
    loop_id = Field("LOOP_ID", 6)
    iterations = Field("NUM_ITERATIONS", 16)
    instr_temp = Instruction("SA_LOOP", 9, OPCODE_WIDTH, (llevel, loop_id, iterations))

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
    NS_NAMES = ["OBUF", "IBUF", "VMEM", "IMM", "DRAM", "VMEM_RD", "VMEM_WR"]
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
    NS_NAMES = ["OBUF", "IBUF", "VMEM", "IMM", "DRAM", "VMEM_RD", "VMEM_WR"]
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
    NS_NAMES = ["OBUF", "IBUF", "VMEM", "IMM", "DRAM", "VMEM_RD", "VMEM_WR"]
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

def creat_bin_ops():


    DTYPE_CAST_NAMES = (3, ["32FXP_16FXP", "32FXP_8FXP", "32FXP_4FXP", "16FXP_32FXP", "8FXP_32FXP", "4FXP_32FXP",
                        "32FP_16FP", "16FP_32FP", "FLOOR", "CEIL"])
    CMP_OP_NAMES = (2, ["EQUAL", "NEQ", "GT", "GTE", "LT", "LTE"])
    CALC_OP_NAMES = (1, ["RELU", "LEAKY_RELU", "SIGMOID", "TANH", "EXP", "LN", "SQRT", "INV_SQRT", "LOG2"])
    ALU_OP_NAMES = (0, ["ADD", "SUB", "MUL", "MACC", "DIV", "MAX", "MIN", "RSHIFT", "LSHIFT", "MOVE", "COND_MOVE_TRUE",
                    "COND_MOVE_FALSE", "NOT", "AND", "OR"])
    instructions = []
    for op_type_list in [DTYPE_CAST_NAMES, CALC_OP_NAMES, ALU_OP_NAMES, CMP_OP_NAMES]:
        op_code = op_type_list[0]
        op_fnctions = op_type_list[1]
        for fn_code, op_fn in enumerate(op_fnctions):
            instructions.append(create_simd_op(op_fn, op_code, fn_code))
    return instructions

def create_dtype_cfg_ops():
    DTYPE_CFG_NAMES = ["32FXP", "16FXP", "8FXP", "32FP", "16FP"]
    instructions = []
    for fn_code, fn_name in enumerate(DTYPE_CFG_NAMES):
        imm_ns_idx = Field("IMM_NS_INDEX_ID", 24)
        instr_temp = Instruction(fn_name, (4 << FUNCTION_CODE_WIDTH) + fn_code, OPCODE_WIDTH + FUNCTION_CODE_WIDTH,
                                 (imm_ns_idx,))
        instructions.append(instr_temp)
    return instructions

def create_lock_ns_op():
    NS_NAMES = ["OBUF", "IBUF", "VMEM", "IMM", "DRAM", "VMEM_RD", "VMEM_WR"]
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
    ITER_CFG_NAMES = ["BASE_SIGN_EXT", "BASE_LOW", "BASE_HIGH", "BASE_ZEROFILL", "STRIDE_SIGN_EXT", "STRIDE_LOW",
                      "STRIDE_HIGH", "STRIDE_ZEROFILL", "SET_IMM_LOW", "SET_IMM_HIGH", "IMM_SIGN_EXT"]
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

    LOOP_OP_NAMES = ["SHORT_LOOP", "LONG_LOOP_INSTR", "LONG_LOOP_ITER"]
    instructions = []

    # short
    num_instr = Field("NUM_INSTR", 12)
    num_iters = Field("NUM_ITERS", 12)
    instructions.append(Instruction(LOOP_OP_NAMES[0], (7 << FUNCTION_CODE_WIDTH) + 0, OPCODE_WIDTH + FUNCTION_CODE_WIDTH,
                             (num_instr, num_iters)))

    # long loop inst
    num_instr_long = Field("NUM_INSTR", 24)
    instructions.append(Instruction(LOOP_OP_NAMES[1], (7 << FUNCTION_CODE_WIDTH) + 1, OPCODE_WIDTH + FUNCTION_CODE_WIDTH,
                             (num_instr_long,)))

    # long loop iter
    num_iters_long = Field("NUM_ITERS", 24)
    instructions.append(Instruction(LOOP_OP_NAMES[2], (7 << FUNCTION_CODE_WIDTH) + 2, OPCODE_WIDTH + FUNCTION_CODE_WIDTH,
                             (num_iters_long,)))

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


GENESYS_INSTRUCTIONS = {
    "systolic_array": [loop_instr(), loop_stride_instr(), group_instr(), base_addr_instr(), block_instr(), load_store()],
    "SIMD": creat_bin_ops() + create_dtype_cfg_ops() + create_lock_ns_op() + create_iterator_ops() +
            create_simd_loop_ops() + create_simd_loop_ops()
}