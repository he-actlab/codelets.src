from codelets.adl import ArchitectureGraph, ComputeNode, CommunicationNode,\
    StorageNode, Codelet, Capability, Operand, NullOperand
from collections import namedtuple
from .codelets import GENESYS_SA_CAPS
from codelets.codelet import Codelet
import numpy as np
from . import SIMD_NS, SIMD_OPCODE_BITWIDTH, OP_DTYPES, \
    OP_LOCATIONS, NS_BITWIDTH, NS_IDX_BITWIDTH

def generate_genesys(genesys_cfg):
    genesys = ComputeNode("Genesys")
    compute = genesys_cfg['compute']
    storage = genesys_cfg['storage']
    sys_array_nodes = ["IBUF", "WBUF", "BBUF", "OBUF", "pe_array"]
    simd_caps = generate_simd_capabilities()
    sa_caps = generate_systolic_array_capabilities()
    systolic_array = ComputeNode("systolic_array")

    for name, attr in storage.items():
        if "capabilities" in attr:
            attr.pop("capabilities")
        mem = StorageNode(name, **attr)
        if name in sys_array_nodes:
            systolic_array.add_subgraph_node(mem)
        else:
            genesys.add_subgraph_node(mem)

    for name, attr in compute.items():
        if "capabilities" in attr:
            attr.pop("capabilities")
        comp = ComputeNode(name, **attr)
        if name in sys_array_nodes:
            systolic_array.add_subgraph_node(comp)

        else:
            genesys.add_subgraph_node(comp)

    systolic_array.add_subgraph_edge('IBUF', 'pe_array')
    systolic_array.add_subgraph_edge('WBUF', 'pe_array')
    systolic_array.add_subgraph_edge('BBUF', 'pe_array')
    systolic_array.add_subgraph_edge('OBUF', 'pe_array')
    systolic_array.add_subgraph_edge('pe_array', 'OBUF')
    genesys.add_subgraph_node(systolic_array)

    for c in sa_caps:
        systolic_array.add_capability(c)

    for _, cdlt_getter in GENESYS_SA_CAPS.items():
        cdlt = cdlt_getter(genesys)
        systolic_array.add_codelet(cdlt)

    simd = genesys.get_subgraph_node("SIMD")

    for c in simd_caps:
        simd.add_capability(c)

    genesys.add_subgraph_edge('VMEM', 'SIMD')
    genesys.add_subgraph_edge('SIMD', 'VMEM')
    genesys.add_subgraph_edge('IMM', 'SIMD')
    genesys.add_subgraph_edge('SIMD', 'IMM')

    genesys.add_subgraph_edge('SIMD', 'DRAM')
    genesys.add_subgraph_edge('DRAM', 'SIMD')
    genesys.add_subgraph_edge('OBUF', 'SIMD')
    return genesys

def create_simd_alu_ops():
    alu_op_code = 0 << SIMD_OPCODE_BITWIDTH
    ALU_OPS = ['ADD', 'SUB', 'MUL', 'MACC', 'DIV', 'MAX', 'MIN', 'MIN', 'RSHIFT', 'LSHIFT', 'MOVE', 'COND_MOVE_TRUE',
               'COND_MOVE_FALSE', 'NOT', 'AND', 'OR']
    SIMD_INPUT_NS = ["OBUF", "VMEM", "IMM"]
    SIMD_OUTPUT_NS = ["IBUF", "VMEM", "IMM"]

    ALU_INPUT_NS = {i: OP_LOCATIONS[i] for i in SIMD_INPUT_NS}
    ALU_OUTPUT_NS = {i: OP_LOCATIONS[i] for i in SIMD_OUTPUT_NS}

    MOVE_INPUT_NS = ALU_INPUT_NS
    MOVE_INPUT_NS['EXTMEM'] = OP_LOCATIONS['EXTMEM']

    MOVE_OUTPUT_NS = ALU_OUTPUT_NS
    MOVE_OUTPUT_NS['EXTMEM'] = OP_LOCATIONS['EXTMEM']

    alu_caps = []

    for i, a in enumerate(ALU_OPS):
        if "MOVE" in a:
            dest = Operand("dest", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=MOVE_OUTPUT_NS,
                           index_size=NS_IDX_BITWIDTH)
            src1 = Operand("src1", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=MOVE_INPUT_NS,
                           index_size=NS_IDX_BITWIDTH)
            src2 = Operand("src2", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=MOVE_INPUT_NS,
                           index_size=NS_IDX_BITWIDTH)
        else:
            dest = Operand("dest", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=ALU_OUTPUT_NS,
                           index_size=NS_IDX_BITWIDTH)
            src1 = Operand("src1", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=ALU_INPUT_NS,
                           index_size=NS_IDX_BITWIDTH)
            src2 = Operand("src2", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=ALU_INPUT_NS,
                           index_size=NS_IDX_BITWIDTH)
        cap = Capability(a, alu_op_code + i, SIMD_OPCODE_BITWIDTH*2, operands=[dest, src1, src2])
        alu_caps.append(cap)
    return alu_caps

def create_simd_calc_ops():
    calc_op_code = 1 << SIMD_OPCODE_BITWIDTH
    CALC_OPS = ['RELU', 'LEAKY_RELU', 'SIGMOID', 'TANH', 'EXP', 'LN', 'SQRT', 'INV_SQRT', 'LOG2']
    SIMD_INPUT_NS = ["OBUF", "VMEM", "IMM"]
    SIMD_OUTPUT_NS = ["IBUF", "VMEM", "IMM"]

    CLC_INPUT_NS = {i: OP_LOCATIONS[i] for i in SIMD_INPUT_NS}
    CLC_OUTPUT_NS = {i: OP_LOCATIONS[i] for i in SIMD_OUTPUT_NS}

    calc_caps = []

    for i, a in enumerate(CALC_OPS):
        dest = Operand("dest", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=CLC_OUTPUT_NS, index_size=NS_IDX_BITWIDTH)
        src1 = Operand("src1", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=CLC_INPUT_NS, index_size=NS_IDX_BITWIDTH)
        fill_val = NullOperand(NS_IDX_BITWIDTH + NS_BITWIDTH, 0)
        cap = Capability(a, calc_op_code + i, SIMD_OPCODE_BITWIDTH*2, operands=[dest, src1, fill_val])
        calc_caps.append(cap)
    return calc_caps


def create_simd_comparison_ops():
    cmp_op_code = 2 << SIMD_OPCODE_BITWIDTH
    COMPARISON_OPS = ['EQUAL', 'NEQ', 'GT', 'GTE', 'LT', 'LTE']

    SIMD_INPUT_NS = ["OBUF", "VMEM", "IMM"]
    SIMD_OUTPUT_NS = ["IBUF", "VMEM", "IMM"]

    CMP_INPUT_NS = {i: OP_LOCATIONS[i] for i in SIMD_INPUT_NS}
    CMP_OUTPUT_NS = {i: OP_LOCATIONS[i] for i in SIMD_OUTPUT_NS}

    cmp_caps = []

    for i, a in enumerate(COMPARISON_OPS):
        dest = Operand("dest", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=CMP_OUTPUT_NS, index_size=NS_IDX_BITWIDTH)
        src1 = Operand("src1", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=CMP_INPUT_NS, index_size=NS_IDX_BITWIDTH)
        src2 = Operand("src2", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components=CMP_INPUT_NS, index_size=NS_IDX_BITWIDTH)
        cap = Capability(a, cmp_op_code + i, SIMD_OPCODE_BITWIDTH*2, operands=[dest, src1, src2])
        cmp_caps.append(cap)
    return cmp_caps

def create_simd_cast_ops():
    cast_op_code = 3 << SIMD_OPCODE_BITWIDTH
    DTYPE_MAP = {d.type.upper() + str(d.bitwidth): d for d in OP_DTYPES}
    CAST_OPS = ['FXP32_FXP16', 'FXP32_FXP8', 'FXP16_FXP32', 'FXP8_FXP32', 'FP32_FP16', 'FP16_FP32']

    SIMD_INPUT_NS = ["OBUF", "VMEM", "IMM"]
    SIMD_OUTPUT_NS = ["IBUF", "VMEM", "IMM"]

    CAST_INPUT_NS = {i: OP_LOCATIONS[i] for i in SIMD_INPUT_NS}
    CAST_OUTPUT_NS = {i: OP_LOCATIONS[i] for i in SIMD_OUTPUT_NS}

    cast_caps = []

    for i, a in enumerate(CAST_OPS):
        dtypes = a.split("_")
        dest_dtype = DTYPE_MAP[dtypes[1]]
        src_dtype = DTYPE_MAP[dtypes[0]]
        dest = Operand("dest", "storage", [dest_dtype], NS_BITWIDTH, value_names=SIMD_NS, components=CAST_OUTPUT_NS, index_size=NS_IDX_BITWIDTH)
        src1 = Operand("src1", "storage", [src_dtype], NS_BITWIDTH, value_names=SIMD_NS, components=CAST_INPUT_NS, index_size=NS_IDX_BITWIDTH)
        src2 = Operand("fraction_pt", "storage", OP_DTYPES, NS_BITWIDTH, value_names=SIMD_NS, components={"IMM": OP_LOCATIONS["IMM"]},
                       index_size=NS_IDX_BITWIDTH)
        cap = Capability(a, cast_op_code + i, SIMD_OPCODE_BITWIDTH*2, operands=[dest, src1, src2])
        cast_caps.append(cap)

    return cast_caps


def create_simd_dtype_cfg_ops():
    cfg_op_code = 4 << SIMD_OPCODE_BITWIDTH
    DTYPE_MAP = {d.type.upper() + str(d.bitwidth): d for d in OP_DTYPES}

    DTYPE_CFG_OPS = ['FXP32', 'FXP16', 'FXP8', 'FP32', 'FP16']

    CFG_INPUT_NS = {"IMM": OP_LOCATIONS["IMM"]}

    cfg_caps = []

    for i, a in enumerate(DTYPE_CFG_OPS):
        fill_val = NullOperand((NS_IDX_BITWIDTH + NS_BITWIDTH) * 2, 0)
        src_dtype = DTYPE_MAP[a]
        src1 = Operand("src1", "storage", [src_dtype], NS_BITWIDTH, value_names=SIMD_NS, components=CFG_INPUT_NS, index_size=NS_IDX_BITWIDTH)
        cap = Capability(a, cfg_op_code + i, SIMD_OPCODE_BITWIDTH*2, operands=[fill_val, src1])
        cfg_caps.append(cap)
    return cfg_caps

def create_simd_lock_ops():
    LOCK_NS_OPS = ['LOCK', 'UNLOCK']
    lock_op_code = 5 << SIMD_OPCODE_BITWIDTH

    SIMD_INPUT_NS = ["OBUF", "VMEM", "IMM"]

    LOCK_NS_LOCS = {i: OP_LOCATIONS[i] for i in SIMD_INPUT_NS}
    LOCK_NS_LOCS['IBUF'] = OP_LOCATIONS['IBUF']
    lock_caps = []

    # TODO: Should the operation datatypes be filled?
    for i, a in enumerate(LOCK_NS_OPS):
        dest = Operand("ns0", "storage", [], NS_BITWIDTH, value_names=SIMD_NS, components=LOCK_NS_LOCS, index_size=0)
        dest_fill_idx = NullOperand(NS_IDX_BITWIDTH, 0)
        src1 = Operand("ns1", "storage", [], NS_BITWIDTH, value_names=SIMD_NS, components=LOCK_NS_LOCS, index_size=0)
        src1_fill_idx = NullOperand(NS_IDX_BITWIDTH, 0)
        src2 = Operand("ns2", "storage", [], NS_BITWIDTH, value_names=SIMD_NS, components=LOCK_NS_LOCS, index_size=0)
        src2_fill_idx = NullOperand(NS_IDX_BITWIDTH, 0)
        cap = Capability(a, lock_op_code + i, SIMD_OPCODE_BITWIDTH*2, operands=[dest, dest_fill_idx, src1, src1_fill_idx, src2, src2_fill_idx])
        lock_caps.append(cap)
    return lock_caps

def create_simd_iter_ops():
    iter_op_code = 6 << SIMD_OPCODE_BITWIDTH
    ITER_OPS = ['BASE_SIGNEXT', 'BASE_LOW', 'BASE_HIGH', 'BASE_ZERO_FILL', 'STRIDE_SIGNEXT', 'STRIDE_LOW',
                'STRIDE_HIGH', 'STRIDE_ZEROFILL', 'SET_IMMMEDIATE_LOW', 'SET_IMMMEDIATE_HIGH', 'IMM_SIGN_EXTEND']

    SIMD_INPUT_NS = ["OBUF", "VMEM", "IMM"]
    ITER_NS_LOCS = {i: OP_LOCATIONS[i] for i in SIMD_INPUT_NS}
    ITER_NS_LOCS['IBUF'] = OP_LOCATIONS['IBUF']

    iter_caps = []

    for i, a in enumerate(ITER_OPS):
        src = Operand("src", "storage", [], NS_BITWIDTH, value_names=SIMD_NS, components=ITER_NS_LOCS, index_size=NS_IDX_BITWIDTH)
        imm = Operand("iter_value", "constant", [], (NS_BITWIDTH + NS_IDX_BITWIDTH)*2)
        cap = Capability(a, iter_op_code + i, SIMD_OPCODE_BITWIDTH*2, operands=[src, imm])
        iter_caps.append(cap)
    return iter_caps


def create_simd_loop_ops():
    loop_op_code = 7 << SIMD_OPCODE_BITWIDTH
    loop_caps = []
    LOOP_OPS = ['SHORT_LOOP', 'LONG_LOOP_INST', 'LONG_LOOP_ITER']

    # Short loop
    num_instr1 = Operand("num_instr1", "constant", [], 12)
    num_instr2 = Operand("num_instr2", "constant", [], 12)
    sloop = Capability(LOOP_OPS[0], loop_op_code, SIMD_OPCODE_BITWIDTH*2, operands=[num_instr1, num_instr2])
    loop_caps.append(sloop)

    # Long loop inst
    num_instr = Operand("num_instr", "constant", [], 24)
    lloop_instr = Capability(LOOP_OPS[1], loop_op_code + 1, SIMD_OPCODE_BITWIDTH*2, operands=[num_instr])
    loop_caps.append(lloop_instr)

    # Long loop iter
    num_iter = Operand("num_iters", "constant", [], 24)
    lloop_iter = Capability(LOOP_OPS[2], loop_op_code + 2, SIMD_OPCODE_BITWIDTH*2, operands=[num_iter])
    loop_caps.append(lloop_iter)
    return loop_caps

def create_simd_permutation_ops():
    perm_op_code = 8 << SIMD_OPCODE_BITWIDTH
    perm_caps = []

    # Start instr
    null_op = NullOperand((NS_IDX_BITWIDTH + NS_BITWIDTH)*3, 0)
    start = Capability("START_PERMUTE", perm_op_code, SIMD_OPCODE_BITWIDTH*2, operands=[null_op])
    perm_caps.append(start)

    # Loop index instr
    loop_order = Operand("loop_order", "constant", [], NS_BITWIDTH)
    loop_index_id = Operand("loop_index", "constant", [], NS_IDX_BITWIDTH)
    num_iters = Operand("num_iters", "constant", [], (NS_BITWIDTH + NS_IDX_BITWIDTH)*2)
    loop_index = Capability("LOOP_INDEX", perm_op_code + 1, SIMD_OPCODE_BITWIDTH*2, operands=[loop_order, loop_index_id, num_iters])
    perm_caps.append(loop_index)
    return perm_caps

def create_sa_loop_ops():
    sa_loop_op_code = 9
    sa_loop_caps = []

    loop_level = Operand("loop_level", "constant", [], 6)
    loop_id = Operand("loop_id", "constant", [], 6)
    num_iters = Operand("num_iters", "constant", [], 16)
    loop_cap = Capability("SA_LOOP", sa_loop_op_code, SIMD_OPCODE_BITWIDTH, operands=[loop_level, loop_id, num_iters])
    sa_loop_caps.append(loop_cap)

    return sa_loop_caps

def create_sa_group_op():
    group_op_code = 10

    sys_array_simd = Operand("sa_simd", "constant", [], 1, value_names=["SYSTOLIC_ARRAY", "SIMD"])
    start_end = Operand("start_end", "constant", [], 1, value_names=["START", "END"])
    group_num = Operand("group_num", "constant", [], 4)
    loop_id = Operand("loop_id", "constant", [], 6)
    num_instr = Operand("num_instr", "constant", [], 16)

    group_cap = Capability("INSTR_ARRAY_GROUP", group_op_code, SIMD_OPCODE_BITWIDTH,
                                    operands=[sys_array_simd, start_end, group_num, loop_id, num_instr])
    return [group_cap]

def create_sa_block_end():
    block_end_code = 11
    fill_val = NullOperand(12, 0)
    last = Operand("last", "constant", [], 16)
    block_cap = Capability("BLOCK_END", block_end_code, SIMD_OPCODE_BITWIDTH, operands=[fill_val, last])
    return [block_cap]


def create_sa_gen_addr():
    gen_addr_op_code = 12

    low_high = Operand("low_high", "constant", [], 1, value_names=["LOW", "HIGH"])
    fill_val = NullOperand(1, 0)
    ld_st = Operand("ld_st", "constant", [], 2, value_names=["LD", "ST", "RD",  "WR"])
    ns = Operand("namespace", "constant", [], 2, value_names=["WBUF", "IBUF", "OBUF", "BBUF"])
    loop_id = Operand("loop_id", "constant", [], 6)
    stride = Operand("stride", "constant", [], 16)

    gen_addr_cap = Capability("GENADDR", gen_addr_op_code, SIMD_OPCODE_BITWIDTH,
                           operands=[low_high, fill_val, ld_st, ns, loop_id, stride])
    return [gen_addr_cap]

def create_sa_base_addr():
    gen_addr_op_code = 13

    low_high = Operand("low_high", "constant", [], 1, value_names=["LOW", "HIGH"])
    ns_or_imm = Operand("namespace_or_imm", "constant", [], 1, value_names=["NS", "INST_MEM"])
    fill_val0 = NullOperand(1, 0)
    ns = Operand("namespace", "constant", [], 2, value_names=["WBUF", "IBUF", "OBUF", "BBUF"])
    fill_val1 = NullOperand(6, 0)
    base_addr = Operand("base_addr", "constant", [], 16)

    gen_addr_cap = Capability("BASEADDR", gen_addr_op_code, SIMD_OPCODE_BITWIDTH,
                              operands=[low_high, ns_or_imm, fill_val0, ns, fill_val1, base_addr])
    return [gen_addr_cap]

def create_sa_ld_st():
    ld_st_op_code = 14 << 1

    ns_or_imm = Operand("namespace_or_imem", "constant", [], 1, value_names=["NS", "INST_MEM"])
    fill_val = NullOperand(2, 0)
    ns = Operand("namespace", "constant", [], 2, value_names=["WBUF", "IBUF", "OBUF", "BBUF"])
    loop_id = Operand("loop_id", "constant", [], 6)
    req_size = Operand("request_size", "constant", [], 16)

    # LOAD op
    ld_cap = Capability("LD", ld_st_op_code + 0, SIMD_OPCODE_BITWIDTH + 1,
                              operands=[ns_or_imm, fill_val, ns, loop_id, req_size])

    st_cap = Capability("ST", ld_st_op_code + 1, SIMD_OPCODE_BITWIDTH + 1,
                              operands=[ns_or_imm, fill_val, ns, loop_id, req_size])

    return [ld_cap, st_cap]

def generate_simd_capabilities():
    alu_caps = create_simd_alu_ops()
    cast_caps = create_simd_cast_ops()
    calc_caps = create_simd_calc_ops()
    cmp_caps = create_simd_comparison_ops()
    cfg_caps = create_simd_dtype_cfg_ops()
    lock_caps = create_simd_lock_ops()
    loop_caps = create_simd_loop_ops()
    iter_caps = create_simd_iter_ops()
    perm_caps = create_simd_permutation_ops()
    return (alu_caps + cast_caps + calc_caps + cmp_caps + cfg_caps + lock_caps + loop_caps + iter_caps + perm_caps)

def generate_systolic_array_capabilities():

    sa_loop = create_sa_loop_ops()
    sa_group = create_sa_group_op()
    block_end = create_sa_block_end()
    gen_addr = create_sa_gen_addr()
    base_addr = create_sa_base_addr()
    ld_st_addr = create_sa_ld_st()
    return (sa_loop + sa_group + block_end + gen_addr + base_addr + ld_st_addr)
