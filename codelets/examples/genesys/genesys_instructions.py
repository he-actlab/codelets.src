from codelets.adl.flex_template.instruction import Field, Instruction

OPCODE_WIDTH = 4



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


def create_simd_alu_ops():
    pass
    # alu_op_code = 0 << SIMD_OPCODE_BITWIDTH
    # ALU_OPS = ['ADD', 'SUB', 'MUL', 'MACC', 'DIV', 'MAX', 'MIN', 'MIN', 'RSHIFT', 'LSHIFT', 'MOVE', 'COND_MOVE_TRUE',
    #            'COND_MOVE_FALSE', 'NOT', 'AND', 'OR']
    # SIMD_INPUT_NS = ["OBUF", "VMEM", "IMM"]
    # SIMD_OUTPUT_NS = ["IBUF", "VMEM", "IMM"]
    #
    # ALU_INPUT_NS = {i: OP_LOCATIONS[i] for i in SIMD_INPUT_NS}
    # ALU_OUTPUT_NS = {i: OP_LOCATIONS[i] for i in SIMD_OUTPUT_NS}
    #
    # MOVE_INPUT_NS = ALU_INPUT_NS
    # MOVE_INPUT_NS['EXTMEM'] = OP_LOCATIONS['EXTMEM']
    #
    # MOVE_OUTPUT_NS = ALU_OUTPUT_NS
    # MOVE_OUTPUT_NS['EXTMEM'] = OP_LOCATIONS['EXTMEM']
    #
    # alu_caps = []
    #
    # for i, a in enumerate(ALU_OPS):
    #     if "MOVE" in a:
    #         dest = Operand("dest", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=MOVE_OUTPUT_NS,
    #                        index_size=NS_IDX_BITWIDTH)
    #         src1 = Operand("src1", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=MOVE_INPUT_NS,
    #                        index_size=NS_IDX_BITWIDTH)
    #         src2 = Operand("src2", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=MOVE_INPUT_NS,
    #                        index_size=NS_IDX_BITWIDTH)
    #     else:
    #         dest = Operand("dest", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=ALU_OUTPUT_NS,
    #                        index_size=NS_IDX_BITWIDTH)
    #         src1 = Operand("src1", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=ALU_INPUT_NS,
    #                        index_size=NS_IDX_BITWIDTH)
    #         src2 = Operand("src2", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=ALU_INPUT_NS,
    #                        index_size=NS_IDX_BITWIDTH)
    #     cap = Instruction(a, alu_op_code + i, SIMD_OPCODE_BITWIDTH * 2, operands=[dest, src1, src2])
    #     alu_caps.append(cap)
    # return alu_caps

def create_simd_calc_ops():
    pass
    # calc_op_code = 1 << SIMD_OPCODE_BITWIDTH
    # CALC_OPS = ['RELU', 'LEAKY_RELU', 'SIGMOID', 'TANH', 'EXP', 'LN', 'SQRT', 'INV_SQRT', 'LOG2']
    # SIMD_INPUT_NS = ["OBUF", "VMEM", "IMM"]
    # SIMD_OUTPUT_NS = ["IBUF", "VMEM", "IMM"]
    #
    # CLC_INPUT_NS = {i: OP_LOCATIONS[i] for i in SIMD_INPUT_NS}
    # CLC_OUTPUT_NS = {i: OP_LOCATIONS[i] for i in SIMD_OUTPUT_NS}
    #
    # calc_caps = []
    #
    # for i, a in enumerate(CALC_OPS):
    #     dest = Operand("dest", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=CLC_OUTPUT_NS, index_size=NS_IDX_BITWIDTH)
    #     src1 = Operand("src1", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=CLC_INPUT_NS, index_size=NS_IDX_BITWIDTH)
    #     fill_val = NullOperand(NS_IDX_BITWIDTH + NS_BITWIDTH, 0)
    #     cap = Instruction(a, calc_op_code + i, SIMD_OPCODE_BITWIDTH * 2, operands=[dest, src1, fill_val])
    #     calc_caps.append(cap)
    # return calc_caps


def create_simd_comparison_ops():
    pass
    # cmp_op_code = 2 << SIMD_OPCODE_BITWIDTH
    # COMPARISON_OPS = ['EQUAL', 'NEQ', 'GT', 'GTE', 'LT', 'LTE']
    #
    # SIMD_INPUT_NS = ["OBUF", "VMEM", "IMM"]
    # SIMD_OUTPUT_NS = ["IBUF", "VMEM", "IMM"]
    #
    # CMP_INPUT_NS = {i: OP_LOCATIONS[i] for i in SIMD_INPUT_NS}
    # CMP_OUTPUT_NS = {i: OP_LOCATIONS[i] for i in SIMD_OUTPUT_NS}
    #
    # cmp_caps = []
    #
    # for i, a in enumerate(COMPARISON_OPS):
    #     dest = Operand("dest", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=CMP_OUTPUT_NS, index_size=NS_IDX_BITWIDTH)
    #     src1 = Operand("src1", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=CMP_INPUT_NS, index_size=NS_IDX_BITWIDTH)
    #     src2 = Operand("src2", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=CMP_INPUT_NS, index_size=NS_IDX_BITWIDTH)
    #     cap = Instruction(a, cmp_op_code + i, SIMD_OPCODE_BITWIDTH * 2, operands=[dest, src1, src2])
    #     cmp_caps.append(cap)
    # return cmp_caps

def create_simd_cast_ops():
    pass
    # cast_op_code = 3 << SIMD_OPCODE_BITWIDTH
    # DTYPE_MAP = {d.type.upper() + str(d.bitwidth): d for d in OP_DTYPES}
    # CAST_OPS = ['FXP32_FXP16', 'FXP32_FXP8', 'FXP16_FXP32', 'FXP8_FXP32', 'FP32_FP16', 'FP16_FP32']
    #
    # SIMD_INPUT_NS = ["OBUF", "VMEM", "IMM"]
    # SIMD_OUTPUT_NS = ["IBUF", "VMEM", "IMM"]
    #
    # CAST_INPUT_NS = {i: OP_LOCATIONS[i] for i in SIMD_INPUT_NS}
    # CAST_OUTPUT_NS = {i: OP_LOCATIONS[i] for i in SIMD_OUTPUT_NS}
    #
    # cast_caps = []
    #
    # for i, a in enumerate(CAST_OPS):
    #     supported_dtypes = a.split("_")
    #     dest_dtype = DTYPE_MAP[supported_dtypes[1]]
    #     src_dtype = DTYPE_MAP[supported_dtypes[0]]
    #     dest = Operand("dest", "storage", [dest_dtype], NS_BITWIDTH, value_names=SIMD_NS, components=CAST_OUTPUT_NS, index_size=NS_IDX_BITWIDTH)
    #     src1 = Operand("src1", "storage", [src_dtype], NS_BITWIDTH, value_names=SIMD_NS, components=CAST_INPUT_NS, index_size=NS_IDX_BITWIDTH)
    #     src2 = Operand("fraction_pt", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components={"IMM": OP_LOCATIONS["IMM"]},
    #                    index_size=NS_IDX_BITWIDTH)
    #     cap = Instruction(a, cast_op_code + i, SIMD_OPCODE_BITWIDTH * 2, operands=[dest, src1, src2])
    #     cast_caps.append(cap)
    #
    # return cast_caps


def create_simd_dtype_cfg_ops():
    pass
    # cfg_op_code = 4 << SIMD_OPCODE_BITWIDTH
    # DTYPE_MAP = {d.type.upper() + str(d.bitwidth): d for d in OP_DTYPES}
    #
    # DTYPE_CFG_OPS = ['FXP32', 'FXP16', 'FXP8', 'FP32', 'FP16']
    #
    # CFG_INPUT_NS = {"IMM": OP_LOCATIONS["IMM"]}
    #
    # cfg_caps = []
    #
    # for i, a in enumerate(DTYPE_CFG_OPS):
    #     fill_val = NullOperand((NS_IDX_BITWIDTH + NS_BITWIDTH) * 2, 0)
    #     src_dtype = DTYPE_MAP[a]
    #     src1 = Operand("src1", "storage", [src_dtype], NS_BITWIDTH, value_names=SIMD_NS, components=CFG_INPUT_NS, index_size=NS_IDX_BITWIDTH)
    #     cap = Instruction(a, cfg_op_code + i, SIMD_OPCODE_BITWIDTH * 2, operands=[fill_val, src1])
    #     cfg_caps.append(cap)
    # return cfg_caps

def create_simd_lock_ops():
    pass
    # LOCK_NS_OPS = ['LOCK', 'UNLOCK']
    # lock_op_code = 5 << SIMD_OPCODE_BITWIDTH
    #
    # SIMD_INPUT_NS = ["OBUF", "VMEM", "IMM"]
    #
    # LOCK_NS_LOCS = {i: OP_LOCATIONS[i] for i in SIMD_INPUT_NS}
    # LOCK_NS_LOCS['IBUF'] = OP_LOCATIONS['IBUF']
    # lock_caps = []
    #
    # # TODO: Should the operation datatypes be filled?
    # for i, a in enumerate(LOCK_NS_OPS):
    #     dest = Operand("ns0", "storage", [], NS_BITWIDTH, value_names=SIMD_NS, components=LOCK_NS_LOCS, index_size=0)
    #     dest_fill_idx = NullOperand(NS_IDX_BITWIDTH, 0)
    #     src1 = Operand("ns1", "storage", [], NS_BITWIDTH, value_names=SIMD_NS, components=LOCK_NS_LOCS, index_size=0)
    #     src1_fill_idx = NullOperand(NS_IDX_BITWIDTH, 0)
    #     src2 = Operand("ns2", "storage", [], NS_BITWIDTH, value_names=SIMD_NS, components=LOCK_NS_LOCS, index_size=0)
    #     src2_fill_idx = NullOperand(NS_IDX_BITWIDTH, 0)
    #     cap = Instruction(a, lock_op_code + i, SIMD_OPCODE_BITWIDTH * 2, operands=[dest, dest_fill_idx, src1, src1_fill_idx, src2, src2_fill_idx])
    #     lock_caps.append(cap)
    # return lock_caps

def create_simd_iter_ops():
    pass
    # iter_op_code = 6 << SIMD_OPCODE_BITWIDTH
    # ITER_OPS = ['BASE_SIGNEXT', 'BASE_LOW', 'BASE_HIGH', 'BASE_ZERO_FILL', 'STRIDE_SIGNEXT', 'STRIDE_LOW',
    #             'STRIDE_HIGH', 'STRIDE_ZEROFILL', 'SET_IMMMEDIATE_LOW', 'SET_IMMMEDIATE_HIGH', 'IMM_SIGN_EXTEND']
    #
    # SIMD_INPUT_NS = ["OBUF", "VMEM", "IMM"]
    # ITER_NS_LOCS = {i: OP_LOCATIONS[i] for i in SIMD_INPUT_NS}
    # ITER_NS_LOCS['IBUF'] = OP_LOCATIONS['IBUF']
    #
    # iter_caps = []
    #
    # for i, a in enumerate(ITER_OPS):
    #     src = Operand("src", "storage", [], NS_BITWIDTH, value_names=SIMD_NS, components=ITER_NS_LOCS, index_size=NS_IDX_BITWIDTH)
    #     imm = Operand("iter_value", "constant", [], (NS_BITWIDTH + NS_IDX_BITWIDTH)*2)
    #     cap = Instruction(a, iter_op_code + i, SIMD_OPCODE_BITWIDTH * 2, operands=[src, imm])
    #     iter_caps.append(cap)
    # return iter_caps


def create_simd_loop_ops():
    pass
    # loop_op_code = 7 << SIMD_OPCODE_BITWIDTH
    # loop_caps = []
    # LOOP_OPS = ['SHORT_LOOP', 'LONG_LOOP_INST', 'LONG_LOOP_ITER']
    #
    # # Short loop
    # num_instr1 = Operand("num_instr1", "constant", [], 12)
    # num_instr2 = Operand("num_instr2", "constant", [], 12)
    # sloop = Instruction(LOOP_OPS[0], loop_op_code, SIMD_OPCODE_BITWIDTH * 2, operands=[num_instr1, num_instr2])
    # loop_caps.append(sloop)
    #
    # # Long loop inst
    # num_instr = Operand("num_instr", "constant", [], 24)
    # lloop_instr = Instruction(LOOP_OPS[1], loop_op_code + 1, SIMD_OPCODE_BITWIDTH * 2, operands=[num_instr])
    # loop_caps.append(lloop_instr)
    #
    # # Long loop iter
    # num_iter = Operand("num_iters", "constant", [], 24)
    # lloop_iter = Instruction(LOOP_OPS[2], loop_op_code + 2, SIMD_OPCODE_BITWIDTH * 2, operands=[num_iter])
    # loop_caps.append(lloop_iter)
    # return loop_caps

def create_simd_permutation_ops():
    pass
    # perm_op_code = 8 << SIMD_OPCODE_BITWIDTH
    # perm_caps = []
    #
    # # Start instr
    # null_op = NullOperand((NS_IDX_BITWIDTH + NS_BITWIDTH)*3, 0)
    # start = Instruction("START_PERMUTE", perm_op_code, SIMD_OPCODE_BITWIDTH * 2, operands=[null_op])
    # perm_caps.append(start)
    #
    # # Loop index instr
    # loop_order = Operand("loop_order", "constant", [], NS_BITWIDTH)
    # loop_index_id = Operand("loop_index", "constant", [], NS_IDX_BITWIDTH)
    # num_iters = Operand("num_iters", "constant", [], (NS_BITWIDTH + NS_IDX_BITWIDTH)*2)
    # loop_index = Instruction("LOOP_INDEX", perm_op_code + 1, SIMD_OPCODE_BITWIDTH * 2, operands=[loop_order, loop_index_id, num_iters])
    # perm_caps.append(loop_index)
    # return perm_caps


GENESYS_INSTRUCTIONS = {
    "systolic_array": [loop_instr, loop_stride_instr, group_instr, base_addr_instr, block_instr, load_store],
    "SIMD": []
}