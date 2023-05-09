from codelets.adl.graph import ArchitectureNode

OP_COMPUTE_CYCLES = {
    "SIGMOID": 4,
    "RELU": 0,
    "ADD": 0,
    "SUB": 0,
    "DIV": 0,
    "LEAKY_RELU": 0,
    "MAX": 0,
    "MIN": 0,
    "MUL": 0,
    "MACC": 2,
    "RSHIFT": 0,
    "LSHIFT": 0,
    "MOVE": 0,
    "COND_MOVE_TRUE": 0,
    "COND_MOVE_FALSE": 0,
    "NOT": 0,
    "AND": 0,
    "OR": 0,
    "NOP": 0,
    "ABS": 0,
    "SIGN": 0,
    "TANH": 4,
    "EXP": 0,
    "POW": 0,
    "LN": 0,
    "SQRT": 8,
    "DIV": 51,
    "INV_SQRT": 0,
    "LOG2": 0,
    "EQUAL": 0,
    "NEQ": 0,
    "GT": 0,
    "GTE": 0,
    "LTE": 0,
    "LT": 0,
    "32FXP_16FXP": 0,
    "32FXP_8FXP": 0,
    "32FXP_4FXP": 0,
    "16FXP_32FXP": 0,
    "8FXP_32FXP": 0,
    "4FXP_32FXP": 0,
    "32FP_16FP": 0,
    "16FP_32FP": 0,
    "32FP_162BFP": 0,
    "16BFP_32BFP": 0,
    "FLOOR": 0,
    "CEIL": 0,
    "TRANSPOSE": 0
}
DTYPE_CAST_NAMES = ["32FXP_16FXP", "32FXP_8FXP", "32FXP_4FXP", "16FXP_32FXP", "8FXP_32FXP", "4FXP_32FXP",
                        "32FP_16FP", "16FP_32FP"]
# Loops
# ALL_LOOP_ID = f"(len(cdlt.get_ops_by_type('loop'))//2)"
ALL_LOOP_ID = f"(len(cdlt.compute_node_loops('SIMD')))"
OPERAND_ITER = ("operand", "op.operands_by_unique_location")
LOOP_ITER = ('loop_op', f'cdlt.get_ops_by_type("loop")')
LOOP_ITER_ENUM = ('loop_op', f'enumerate(cdlt.get_ops_by_type("loop"))')

# Stride calculation
MVMT_TYPE = f"'up' if operand.data_path[0] == 'DRAM' else 'down'"
RD_WRITE_OPERAND_TEMPLATE = "(True if {OPERAND} in op.dests else False)"
RD_WRITE_OPERAND = f"(True if operand in op.dests else False)"

LOOP_STRIDE_TEMPLATE = "{OPERAND}.get_offset(cdlt," \
              "loop_op.loop_id," \
              "hag, op.op_str, 'SIMD', write={RD_WRITE}, " \
              "outer_loop=False)"

LOOP_STRIDE = f"operand.get_offset(cdlt," \
              f"loop_op.loop_id," \
              f"hag, op.op_str, 'SIMD', write={RD_WRITE_OPERAND}, " \
              f"outer_loop=False)"

TRANSPOSE_LOOP_STRIDE_RD = f"op.sources[0].get_offset(cdlt," \
              f"loop_op[1].loop_id," \
              f"hag, op.op_str, 'SIMD', write=False, " \
              f"outer_loop=False)"


# Instruction generation conditions
IS_DEP_COND = f'loop_op.op_str in cdlt.all_dependencies(op.dependencies)'
COMPUTE_DEP = f'loop_op.op_str in op.operand_indices'
OUTER_LOOP_COND = f'loop_op.loop_level < op.loop_level'
IS_DIRECT_DEP_COND = f"cdlt.is_direct_loop_dep(loop_op, 'SIMD')"
# LOOP_CONDS = [IS_DEP_COND, COMPUTE_DEP, OUTER_LOOP_COND, IS_DIRECT_DEP_COND]
LOOP_CONDS = [COMPUTE_DEP, OUTER_LOOP_COND, IS_DIRECT_DEP_COND]

# Operand location string
OPERAND_LOC_TMPLT = "op.get_operand_location({OPERAND}.name)"
OPERAND_LOC = OPERAND_LOC_TMPLT.format(OPERAND="operand")


BASE_SIGN_EXT_TEMPLATE = "({OPERAND}.get_mem_offset({OP_LOC})//(hag.get_subgraph_node({OP_LOC}).data_size)) if op.get_operand_location({OPERAND}.name) " \
                "!= 'IMM' else cdlt.temps.index({OPERAND})"

BASE_SIGN_EXT = BASE_SIGN_EXT_TEMPLATE.format(OPERAND="operand", OP_LOC=OPERAND_LOC)
BASE_SIGN_EXT_TR_RD = BASE_SIGN_EXT_TEMPLATE.format(OPERAND="op.sources[0]", OP_LOC=f"op.get_operand_location(op.sources[0].name)")
BASE_SIGN_EXT_TR_WR = BASE_SIGN_EXT_TEMPLATE.format(OPERAND="op.dests[0]", OP_LOC=f"op.get_operand_location(op.dests[0].name)")
# BASE_SIGN_EXT = f"(operand.get_mem_offset({OPERAND_LOC})//(hag.get_subgraph_node({OPERAND_LOC}).data_size) + 1) if op.get_operand_location(operand.name) " \
#                 f"!= 'IMM' else cdlt.temps.index(operand)"



ENUM_IS_DEP_COND = f'loop_op[1].op_str in cdlt.all_dependencies(op.dependencies)'
ENUM_OUTER_LOOP_COND = f'loop_op[1].loop_level < op.loop_level'
ENUM_IS_DIRECT_DEP_COND = f"cdlt.is_direct_loop_dep(loop_op[1], 'SIMD')"
ENUM_LOOP_CONDS = [ENUM_IS_DEP_COND, ENUM_OUTER_LOOP_COND, ENUM_IS_DIRECT_DEP_COND]
LOOP_ITER_ENUM_TRANSPOSE_WR = ('loop_op', f'cdlt.get_transpose_loops(op)')


def simd_transpose_rd_(hag):
    instructions = []
    loop_tabs = f"{ALL_LOOP_ID} + loop_op[0] - 1"
    src_loop_iter = ('loop_op', f'enumerate(cdlt.get_ops_by_type("loop"))')

    ## Source Read
    operand = "op.sources[0]"
    # src_loop_iters = f"cdlt.inner_iter({operand}, loop_op[1], loop_op[0]) - 1"
    src_loop_iters = f"loop_op[1].iter_count // cdlt.param_tiling[2][cdlt.loop_param_map[loop_op[1].op_str]] - 1"

    src_op_loc = f"op.get_operand_location({operand}.name)"

    src_ns_idx = f"(loop_op[1].loop_id % {ALL_LOOP_ID}) + ({operand}.get_mem_index({src_op_loc}) * {ALL_LOOP_ID}) if op.get_operand_location({operand}.name) != 'IMM' " \
                 f"else cdlt.temps.index({operand})"
    # src_base_sign_ext = f"({operand}.get_mem_offset({src_op_loc})//({operand}.dtype.bits()) + 1) if {src_op_loc} " \
    #                     f"!= 'IMM' else cdlt.temps.index({operand})"
    src_base_sign_ext = BASE_SIGN_EXT_TEMPLATE.format(OPERAND=operand, OP_LOC=src_op_loc)
    # src_loop_stride = f"cdlt.inner_stride({operand}, loop_op[1], loop_op[0])"
    src_loop_stride = TRANSPOSE_LOOP_STRIDE_RD

    src_base_instr = hag.get_primitive_template("PERM_SET_BASE_ADDR")
    src_base_instr.set_print_tabs(ALL_LOOP_ID)
    src_base_instr.set_field_by_name('RD_WR', "RD")
    src_base_instr.set_field_flex_param('BASE_ADDR', src_base_sign_ext)
    instructions.append(src_base_instr)

    src_stride_instr = hag.get_primitive_template("PERM_SET_LOOP_STRIDE")
    src_stride_instr.set_print_tabs(loop_tabs)
    src_stride_instr.add_iterable(*src_loop_iter)
    src_stride_instr.add_condition(" and ".join(ENUM_LOOP_CONDS))
    src_stride_instr.set_field_by_name('RD_WR', "RD")
    src_stride_instr.set_field_flex_param('LOOP_INDEX_ID', src_ns_idx)
    src_stride_instr.set_field_flex_param('STRIDE', src_loop_stride)

    src_iter_instr = hag.get_primitive_template("PERM_SET_LOOP_ITER")
    src_iter_instr.set_print_tabs(loop_tabs)
    src_iter_instr.add_iterable(*src_loop_iter)
    src_iter_instr.add_condition(" and ".join(ENUM_LOOP_CONDS))
    src_iter_instr.set_field_by_name('RD_WR', "RD")
    src_iter_instr.set_field_flex_param('LOOP_INDEX_ID', src_ns_idx)
    src_iter_instr.set_field_flex_param('NUM_ITERS', src_loop_iters)
    src_stride_instr.add_base_instruction(src_iter_instr)
    instructions.append(src_stride_instr)

    return instructions

def simd_transpose_wr_(hag):
    instructions = []
    simd_size = hag.get_subgraph_node("SIMD").dimensions[0]
    loop_tabs = f"{ALL_LOOP_ID} + loop_op[0]"
    dst_loop_iter = ('loop_op', f'enumerate(cdlt.get_ops_by_type("loop")[:-2] + [list(cdlt.get_ops_by_type("loop"))[-1]] + [list(cdlt.get_ops_by_type("loop"))[-2]])')

    dst_outer_iters = f'max([l.iter_count for i, l in enumerate(cdlt.get_ops_by_type("loop")) if i >= len(cdlt.get_ops_by_type("loop"))/2]) // {simd_size}'

    dst_outer_stride = f'cdlt.get_ops_by_type("loop")[-1].iter_count'

    operand = "op.dests[0]"
    src_operand = "op.sources[0]"

    dst_loop_iter_base = f"cdlt.inner_iter({operand}, loop_op[1], loop_op[0])"

    ## Changes for benchmarking
    # Original implementation
    dst_loop_iters = f"({dst_loop_iter_base}-1) if loop_op[0] != list({dst_loop_iter[1]})[-1][0] else (({dst_loop_iter_base})//({dst_outer_iters})) - 1)"


    ## End changes for benchmarking

    dst_op_loc = f"op.get_operand_location({operand}.name)"

    dst_ns_idx = f"(loop_op[1].loop_id % {ALL_LOOP_ID}) + ({operand}.get_mem_index({dst_op_loc}) * {ALL_LOOP_ID}) + 1 if op.get_operand_location({operand}.name) != 'IMM' " \
             f"else cdlt.temps.index({operand}) + 1"


    dst_base_sign_ext = BASE_SIGN_EXT_TEMPLATE.format(OPERAND=operand, OP_LOC=dst_op_loc)

    dst_loop_stride = f"cdlt.inner_stride({operand}, loop_op[1], loop_op[0])"

    dst_base_instr = hag.get_primitive_template("PERM_SET_BASE_ADDR")
    dst_base_instr.set_print_tabs(ALL_LOOP_ID)
    dst_base_instr.set_field_by_name('RD_WR', "WR")
    dst_base_instr.set_field_flex_param('BASE_ADDR', dst_base_sign_ext)
    instructions.append(dst_base_instr)
    ## Outer loop for dest

    outer_stride_instr = hag.get_primitive_template("PERM_SET_LOOP_STRIDE")
    outer_stride_instr.set_print_tabs(f"{ALL_LOOP_ID} + 1")
    outer_stride_instr.set_field_by_name('RD_WR', "WR")
    outer_stride_instr.set_field_value('LOOP_INDEX_ID', 0)
    outer_stride_instr.set_field_flex_param('STRIDE', f"{dst_outer_stride}")
    instructions.append(outer_stride_instr)

    outer_iter_instr = hag.get_primitive_template("PERM_SET_LOOP_ITER")
    outer_iter_instr.set_print_tabs(f"{ALL_LOOP_ID} + 1")
    outer_iter_instr.set_field_by_name('RD_WR', "WR")
    outer_iter_instr.set_field_value('LOOP_INDEX_ID', 0)
    outer_iter_instr.set_field_flex_param('NUM_ITERS', f"({dst_outer_iters}) - 1")
    instructions.append(outer_iter_instr)

    ## Inner loops for dest

    dst_stride_instr = hag.get_primitive_template("PERM_SET_LOOP_STRIDE")
    dst_stride_instr.set_print_tabs(loop_tabs)
    dst_stride_instr.add_iterable(*dst_loop_iter)
    dst_stride_instr.add_condition(" and ".join(ENUM_LOOP_CONDS))
    dst_stride_instr.set_field_by_name('RD_WR', "WR")
    dst_stride_instr.set_field_flex_param('LOOP_INDEX_ID', dst_ns_idx)
    dst_stride_instr.set_field_flex_param('STRIDE', dst_loop_stride)


    dst_iter_instr = hag.get_primitive_template("PERM_SET_LOOP_ITER")
    dst_iter_instr.set_print_tabs(loop_tabs)
    dst_iter_instr.add_iterable(*dst_loop_iter)
    dst_iter_instr.add_condition(" and ".join(ENUM_LOOP_CONDS))
    dst_iter_instr.set_field_by_name('RD_WR', "WR")
    dst_iter_instr.set_field_flex_param('LOOP_INDEX_ID', dst_ns_idx)
    dst_iter_instr.set_field_flex_param('NUM_ITERS', dst_loop_iters)
    dst_stride_instr.add_base_instruction(dst_iter_instr)
    instructions.append(dst_stride_instr)
    return instructions

def simd_transpose_wr(hag):
    instructions = []
    # In contrast to transpose rd, write must take into account the dimensions to be transposed.
    operand = "op.dests[0]"
    ns_idx = f"(loop_op[1].loop_level % {ALL_LOOP_ID}) + ({operand}.get_mem_index({OPERAND_LOC_TMPLT.format(OPERAND=operand)}) * {ALL_LOOP_ID}) if op.get_operand_location({operand}.name) != 'IMM' else cdlt.temps.index({operand})"
    # We also have an additional loop for the write instruction, which means we need to increment the ns_idx by 1
    ns_idx = f"({ns_idx}) + 1"

    # First, we set the base addr for the input operand being written
    # Currently, permutation operations only support 16 bit addresses
    base_sign_ext = f"program.extract_bits({BASE_SIGN_EXT_TR_WR}, 16, 0)"
    base_instr = hag.get_primitive_template("PERM_SET_BASE_ADDR")
    base_instr.set_print_tabs(ALL_LOOP_ID)
    base_instr.set_field_by_name('RD_WR', "WR")
    base_instr.set_field_flex_param('BASE_ADDR', base_sign_ext)
    instructions.append(base_instr)

    # Second, we generate an outer loop which iterates over the maximum iteration number

    #TODO

    # Third, we need to generate stride and iteration instructions for each dimension
    # To factor in the dimensions being transposed, we need to iterate over loops in a different order than the read instruction
    # Specifically, if one of the dimensions being transposed is one of the innermost dimensions,

    # Case #1: Outer dimensions are being transposed, e.g. for an input with shape (N, H, W, C)=(4,2,2,8) which transposes N, H, we get (H, N, W, C)
    # Lanes = 4
    # A. (N, H, W, C)=(4,2,2,8) --> (H, N, W, C)=(2,4,2,8)
        # PERM DIMS -> (N, H)
        ## RD Loop order -> (N, H, W, C) -> (4,2,2,8):
            ## N: iter=4, stride=(2*2*8)/4 -> 8
            ## H: iter=2, stride=(2*8)/4 -> 4
            ## W: iter=2, stride=8/4 -> 2
            ## C: iter=8/4 -> 2, stride=1
        ## WR Loop order -> (H, N, W, C) -> (2,4,2,8):
            ## N: iter=4, stride= 2*8/4 -> 4
            ## H: iter=2, stride=(4*2*8/4) -> 16
            ## W: iter=2, stride=2
            ## C: iter=8/4 -> 2, stride=1
    # dims = [N: 4, H: 2, W: 2, C: 8]
    # B. (N, H, W, C)=(4,2,2,8) --> (N, W, H, C)=(2,2,4,8)
        # PERM DIMS -> (H, W)
        ## RD Loop order -> (N, H, W, C) -> (4,2,2,8):
            ## N: iter=4, stride=(2*2*8)/4 -> 8
            ## H: iter=2, stride=(2*8)/4 -> 4
            ## W: iter=2, stride=8/4 -> 2
            ## C: iter=8/4 -> 2, stride=1
        ## WR Loop order -> (N, W, H, C) -> (2,2,4,8):
            ## N: iter=4, stride=(2*2*8)/4 -> 8
            ## W: iter=2, stride=8/4 -> 2
            ## H: iter=2, stride=(2*8)/4 -> 4
            ## C: iter=8/4 -> 2, stride=1


    # Case #2: An inner dimensions is being transposed, e.g. for an input with shape (N, H, W, C)=(4,2,2,8) which transposes W, C we get (N, H, C, W)
    # Lanes = 4
    ## RD Loop order -> (N, H, W, C):
        ## C: iter=2, stride=2
        ## N: iter=2, stride=1
        ## W: iter=2, stride=16
        ## H: iter=4, stride=4
    ## WR Loop order -> (N, H, C, W)
        ## C: iter=2, stride=2
        ## N: iter=2, stride=1
        ## H: iter=1, stride=0
        ## W: iter=8, stride=4

    macro_instr = hag.get_primitive_template("PERM_SET_LOOP_STRIDE")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.add_iterable(*LOOP_ITER_ENUM_TRANSPOSE_WR)
    macro_instr.set_field_by_name("RD_WR", "WR")
    macro_instr.set_field_flex_param("LOOP_INDEX_ID", ns_idx)
    macro_instr.set_field_flex_param("STRIDE", "loop_op[2]")

    iter_instr = hag.get_primitive_template("PERM_SET_LOOP_ITER")
    iter_instr.set_print_tabs(ALL_LOOP_ID)
    iter_instr.add_iterable(*LOOP_ITER_ENUM_TRANSPOSE_WR)
    iter_instr.set_field_by_name("RD_WR", "WR")
    iter_instr.set_field_flex_param("LOOP_INDEX_ID", ns_idx)
    iter_instr.set_field_flex_param("NUM_ITERS", "loop_op[3]")
    macro_instr.add_base_instruction(iter_instr)
    instructions.append(macro_instr)

    return instructions

def simd_transpose_rd(hag):
    instructions = []
    operand = "op.sources[0]"
    ns_idx = f"(loop_op[1].loop_level % {ALL_LOOP_ID}) + ({operand}.get_mem_index({OPERAND_LOC_TMPLT.format(OPERAND=operand)}) * {ALL_LOOP_ID}) if op.get_operand_location({operand}.name) != 'IMM' else cdlt.temps.index({operand})"
    niters = f"loop_op[1].iter_count // cdlt.param_tiling[2][cdlt.loop_param_map[loop_op[1].op_str]] - 1"
    # First, we set the base addr for the input operand being read
    # Currently, permutation operations only support 16 bit addresses
    base_sign_ext = f"program.extract_bits({BASE_SIGN_EXT_TR_RD}, 16, 0)"

    base_instr = hag.get_primitive_template("PERM_SET_BASE_ADDR")
    base_instr.set_print_tabs(ALL_LOOP_ID)
    base_instr.set_field_by_name('RD_WR', "RD")
    base_instr.set_field_flex_param('BASE_ADDR', base_sign_ext)
    instructions.append(base_instr)

    # Second, we need to generate stride and iteration instructions for each dimension
    macro_instr = hag.get_primitive_template("PERM_SET_LOOP_STRIDE")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.add_iterable(*LOOP_ITER_ENUM)
    macro_instr.add_condition(" and ".join(ENUM_LOOP_CONDS))
    macro_instr.set_field_by_name("RD_WR", "RD")
    macro_instr.set_field_flex_param("LOOP_INDEX_ID", ns_idx)
    macro_instr.set_field_flex_param("STRIDE", TRANSPOSE_LOOP_STRIDE_RD)

    iter_instr = hag.get_primitive_template("PERM_SET_LOOP_ITER")
    iter_instr.set_print_tabs(ALL_LOOP_ID)
    iter_instr.add_iterable(*LOOP_ITER_ENUM)
    iter_instr.add_condition(" and ".join(ENUM_LOOP_CONDS))
    iter_instr.set_field_by_name("RD_WR", "RD")
    iter_instr.set_field_flex_param("LOOP_INDEX_ID", ns_idx)
    iter_instr.set_field_flex_param("NUM_ITERS", niters)
    macro_instr.add_base_instruction(iter_instr)
    instructions.append(macro_instr)


    return instructions


def simd_transpose(hag):
    instructions = []
    src_op_loc = f"op.get_operand_location(op.sources[0].name)"
    dst_op_loc = f"op.get_operand_location(op.dests[0].name)"

    # instructions += simd_transpose_rd(hag)
    instructions += simd_transpose_wr(hag)


    start_instr = hag.get_primitive_template("PERM_START")
    start_instr.set_print_tabs(f"{ALL_LOOP_ID}*2 + {ALL_LOOP_ID}//2 + 1")
    start_instr.set_field_flex_param('DST_NS_ID', dst_op_loc)
    start_instr.set_field_flex_param('SRC_NS_ID', src_op_loc)
    start_instr.set_field_by_name('SHUFFLE_BANKS', 'DO_SHUFFLE')
    instructions.append(start_instr)


    return instructions

def add_pow_loop(hag):
    # Params
    simd_size = hag.get_subgraph_node("SIMD").dimensions[0]



    # Index generation
    ns_idx = f"(loop_op.loop_id % {ALL_LOOP_ID}) + (operand.get_mem_index({OPERAND_LOC}) * {ALL_LOOP_ID}) if op.get_operand_location(operand.name) != 'IMM' else cdlt.temps.index(operand)"
    # base_sign_ext = f"operand.get_mem_offset({OPERAND_LOC})//(operand.dtype.bits()) if op.get_operand_location(operand.name) " \
    #                 f"!= 'IMM' else cdlt.temps.index(operand)"


    instructions = []
    pow_ns_idx = f"({ALL_LOOP_ID}) + (operand.get_mem_index({OPERAND_LOC}) * {ALL_LOOP_ID}) if op.get_operand_location(operand.name) != 'IMM' else cdlt.temps.index(operand)"

    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.add_iterable(*OPERAND_ITER)
    macro_instr.set_field_flex_param('NS_ID', OPERAND_LOC)
    macro_instr.set_field_flex_param('NS_INDEX_ID', pow_ns_idx)
    macro_instr.set_field_flex_param('IMM', BASE_SIGN_EXT)
    #
    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.add_iterable(*OPERAND_ITER)
    sub_instr.set_field_flex_param('NS_ID', OPERAND_LOC)
    sub_instr.set_field_flex_param('NS_INDEX_ID', pow_ns_idx)
    sub_instr.set_field_flex_param('IMM', '0')
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    exp_tabs = f"len(cdlt.get_ops_by_type('loop'))"
    macro_instr = hag.get_primitive_template("SET_ITER")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.set_field_flex_param("LOOP_ID", f"0")
    macro_instr.set_field_flex_param("NUM_ITER",
                                     f"cdlt.required_params['exp'].value - 1")

    # Extra set index
    sub_instr = hag.get_primitive_template("SET_INDEX")
    set_index_fmt = "({all_loop_id}) + ({operand}.get_mem_index({op_loc}) * {all_loop_id}) if op.get_operand_location({operand}.name) != 'IMM' else cdlt.temps.index({operand})"

    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
    sub_instr.set_field_flex_param("DST_INDEX_ID",
                                   set_index_fmt.format(all_loop_id=ALL_LOOP_ID, operand="op.dests[0]",
                                                        op_loc="op.get_operand_location(op.dests[0].name)"))
    sub_instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID",
                                   set_index_fmt.format(all_loop_id=ALL_LOOP_ID, operand="op.sources[0]",
                                                        op_loc="op.get_operand_location(op.sources[0].name)"))
    src2_ns = "'IMM' if op.get_operand_location(op.sources[0].name) != 'IMM' else 'VMEM1'"
    sub_instr.set_field_flex_param("SRC2_NS_ID",
                                   f"({src2_ns}) if len(op.sources) == 1 else op.get_operand_location(op.sources[1].name)")
    src2_idx = f"0 if len(op.sources) == 1 else " + set_index_fmt.format(all_loop_id=ALL_LOOP_ID,
                                                                         operand="op.sources[1]",
                                                                         op_loc="op.get_operand_location(op.sources[1].name)")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", src2_idx)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)
    if hag.meta_cfg['ADDR_GEN_TEST']:
        instructions += sw_addr_nops_nested("POW", hag)
    elif hag.meta_cfg['OBUF_TO_VMEM_TEST']:
        instructions += obuf_mv_vmem_test("POW", hag)
    return instructions

def add_pow_compute(hag, rd_op_str):
    instructions = []
    instr = hag.get_primitive_template("MUL")
    instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
    instr.set_field_flex_param("DST_INDEX_ID",
                               f"op.dests[0].get_mem_index(op.get_operand_location(op.dests[0].name))")
    instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
    instr.set_field_flex_param("SRC1_INDEX_ID", rd_op_str.format(IDX=0))
    instr.set_field_flex_param("SRC2_NS_ID", "op.get_operand_location(op.sources[1].name)")
    instr.set_field_flex_param("SRC2_INDEX_ID", rd_op_str.format(IDX=1))
    instructions.append(instr)
    return instructions

def insert_dtype_cfg_from_cast(op_name, hag):
    instructions = []
    instr = hag.get_primitive_template("DTYPE_CFG")
    instr.set_field_flex_param("DTYPE", "str(op.dests[0].dtype.bits()) + op.dests[0].dtype.type")
    instr.set_field_flex_param("DST_BITS", "op.dests[0].dtype.exp")
    instr.set_field_flex_param("SRC1_BITS", "op.sources[0].dtype.exp")
    instr.set_field_flex_param("SRC2_BITS", "op.sources[0].dtype.exp")
    instructions.append(instr)
    return instructions

def single_loop_overhead(hag, iter_split_cond, total_iters):
    instructions = []
    ns_idx = "0"

    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.add_condition(f"{iter_split_cond}")
    macro_instr.set_field_by_name('NS_ID', "VMEM1")
    macro_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    macro_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    macro_instr.set_field_value('IMM', 0)
    #

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.add_condition(f"{iter_split_cond}")
    sub_instr.set_field_by_name('NS_ID', "VMEM1")
    sub_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    sub_instr.set_field_flex_param('IMM', 0)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    ## Now generate loops
    macro_instr = hag.get_primitive_template("SET_ITER")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.add_condition(f"{iter_split_cond}")
    macro_instr.set_field_flex_param("LOOP_ID", f"0")
    macro_instr.set_field_flex_param("NUM_ITER",
                                     f"{total_iters}")

    sub_instr = hag.get_primitive_template("SET_INDEX")
    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.add_condition(f"{iter_split_cond}")
    sub_instr.set_field_by_name("DST_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("DST_INDEX_ID", ns_idx)
    sub_instr.set_field_by_name("SRC1_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID", ns_idx)
    sub_instr.set_field_by_name("SRC2_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", ns_idx)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)
    return instructions

def multi_loop_overhead(hag, iter_split_cond):

    total_iters_outer = f"cdlt.loop_overhead_iters(op, 'SIMD', True, 16)[1]"
    total_iters_inner = f"cdlt.loop_overhead_iters(op, 'SIMD', True, 16)[0]"
    instructions = []
    ns_idx_outer = "0"
    ns_idx_inner = "1"
    iter_split_cond = f"not ({iter_split_cond})"
    ## First, do the outer loops
    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.add_condition(iter_split_cond)
    macro_instr.set_field_by_name('NS_ID', "VMEM1")
    macro_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    macro_instr.set_field_flex_param('NS_INDEX_ID', ns_idx_outer)
    macro_instr.set_field_value('IMM', 0)
    #

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.add_condition(f"{iter_split_cond}")
    sub_instr.set_field_by_name('NS_ID', "VMEM1")
    sub_instr.set_field_flex_param('NS_INDEX_ID', ns_idx_outer)
    sub_instr.set_field_flex_param('IMM', 0)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    ## Now generate loops
    macro_instr = hag.get_primitive_template("SET_ITER")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.add_condition(f"{iter_split_cond}")
    macro_instr.set_field_flex_param("LOOP_ID", f"0")
    macro_instr.set_field_flex_param("NUM_ITER",
                                     f"{total_iters_outer}")

    sub_instr = hag.get_primitive_template("SET_INDEX")
    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.add_condition(f"{iter_split_cond}")
    sub_instr.set_field_by_name("DST_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("DST_INDEX_ID", ns_idx_outer)
    sub_instr.set_field_by_name("SRC1_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID", ns_idx_outer)
    sub_instr.set_field_by_name("SRC2_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", ns_idx_outer)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)


    # Next, do the inner loops

    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.add_condition(iter_split_cond)
    macro_instr.set_field_by_name('NS_ID', "VMEM1")
    macro_instr.set_print_tabs("op.loop_level - op.loop_level//2 + 1")
    macro_instr.set_field_flex_param('NS_INDEX_ID', ns_idx_inner)
    macro_instr.set_field_value('IMM', 0)
    #

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_print_tabs(f"{ALL_LOOP_ID} + 1")
    sub_instr.add_condition(f"{iter_split_cond}")
    sub_instr.set_field_by_name('NS_ID', "VMEM1")
    sub_instr.set_field_flex_param('NS_INDEX_ID', ns_idx_inner)
    sub_instr.set_field_flex_param('IMM', 0)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    ## Now generate loops
    macro_instr = hag.get_primitive_template("SET_ITER")
    macro_instr.set_print_tabs(f"{ALL_LOOP_ID} + 1")
    macro_instr.add_condition(f"{iter_split_cond}")
    macro_instr.set_field_flex_param("LOOP_ID", f"0")
    macro_instr.set_field_flex_param("NUM_ITER",
                                     f"{total_iters_inner}")

    sub_instr = hag.get_primitive_template("SET_INDEX")
    sub_instr.set_print_tabs(f"{ALL_LOOP_ID} + 1")
    sub_instr.add_condition(f"{iter_split_cond}")
    sub_instr.set_field_by_name("DST_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("DST_INDEX_ID", ns_idx_inner)
    sub_instr.set_field_by_name("SRC1_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID", ns_idx_inner)
    sub_instr.set_field_by_name("SRC2_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", ns_idx_inner)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    return instructions

def loop_overhead(op_name, hag):
    instructions = []

    ## First, need to do BASE/SIGN_EXT

    loop_idx_offset = 0 if op_name != "POW" else 1

    ## Now, create the loop
    ## First loop
    all_loop_list = LOOP_ITER[1]
    other_constr = " and ".join(LOOP_CONDS + ["loop_op.op_str in op.dependencies"])
    tgt_loop_list = f"[loop_op.iter_count // cdlt.param_tiling[2][cdlt.loop_param_map[loop_op.op_str]] for loop_op in {all_loop_list} if ({other_constr})]"

    # total_iters = f"np.sum([np.prod(({tgt_loop_list})[:i+1])*3 for i in range(len({tgt_loop_list}))])"

    total_iters = f"cdlt.loop_overhead_iters(op, 'SIMD')"

    iter_split_cond = f"np.log2({total_iters}) <= 16"

    ## Single loop body execution
    instructions += single_loop_overhead(hag, iter_split_cond, total_iters)
    ## End Single loop body execution

    ## Multi-loop body execution
    instructions += multi_loop_overhead(hag, iter_split_cond)
    ## End Multi-loop body execution


    # Compute instruction
    instr = hag.get_primitive_template("SET_INST")
    instr.add_condition("cdlt.op_id_counters['compute'] - 1 > op.op_id")
    instr.set_field_flex_param("SINGLE_NESTED", "1")
    instr.set_field_flex_param("NUM_INSTR", "1")
    instructions.append(instr)


    noop_instr = hag.get_primitive_template("NOP")
    instructions.append(noop_instr)

    return instructions

def sw_addr_nops(op_name, hag):
    instructions = []

    ## First, need to do BASE/SIGN_EXT



    ## Now, create the loop
    ## First loop
    all_loop_list = LOOP_ITER[1]
    other_constr = " and ".join(LOOP_CONDS + ["loop_op.op_str in op.dependencies"])

    tgt_loop_list = f"[loop_op.iter_count // cdlt.param_tiling[2][cdlt.loop_param_map[loop_op.op_str]] for loop_op in {all_loop_list} if {other_constr}]"
    loop_count = f"np.prod({tgt_loop_list})"
    filterd_operand_locs = "([op.get_operand_location(o.name) for o in op.operands if op.get_operand_location(o.name) != 'IMM'])"
    bin_un_count = f"{loop_count}*3 if len({filterd_operand_locs}) > 2 else {loop_count}*2"


    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.set_field_by_name('NS_ID', "VMEM1")
    macro_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    macro_instr.set_field_flex_param('NS_INDEX_ID', "0")
    macro_instr.set_field_value('IMM', 1)
    #

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_field_by_name('NS_ID', "VMEM1")
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_flex_param('NS_INDEX_ID', "0")
    sub_instr.set_field_flex_param('IMM', f"0")
    macro_instr.add_base_instruction(sub_instr)

    sub_instr = hag.get_primitive_template("SET_ITER")
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_flex_param("LOOP_ID", f"0")
    sub_instr.set_field_flex_param("NUM_ITER", loop_count)
    macro_instr.add_base_instruction(sub_instr)

    sub_instr = hag.get_primitive_template("SET_INDEX")
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_by_name("DST_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("DST_INDEX_ID", "0")
    sub_instr.set_field_by_name("SRC1_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID", "0")
    sub_instr.set_field_by_name("SRC2_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", '0')
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    noop_instr = hag.get_primitive_template("NOP")
    noop_instr.set_print_tabs("op.loop_level - op.loop_level//2 + 1")
    instructions.append(noop_instr)

    ### Second Loop
    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.set_field_by_name('NS_ID', "VMEM1")
    macro_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    macro_instr.set_field_flex_param('NS_INDEX_ID', "0")
    macro_instr.set_field_value('IMM', 1)
    #

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_field_by_name('NS_ID', "VMEM1")
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_flex_param('NS_INDEX_ID', "0")
    sub_instr.set_field_flex_param('IMM', f"0")
    macro_instr.add_base_instruction(sub_instr)
    sub_instr = hag.get_primitive_template("SET_ITER")
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_flex_param("LOOP_ID", f"0")
    sub_instr.set_field_flex_param("NUM_ITER", loop_count)
    macro_instr.add_base_instruction(sub_instr)

    sub_instr = hag.get_primitive_template("SET_INDEX")
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_by_name("DST_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("DST_INDEX_ID", "0")
    sub_instr.set_field_by_name("SRC1_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID", "0")
    sub_instr.set_field_by_name("SRC2_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", '0')
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    noop_instr = hag.get_primitive_template("NOP")
    noop_instr.set_print_tabs("op.loop_level - op.loop_level//2 + 1")
    instructions.append(noop_instr)


    ### Optional third
    third_cond = f"len({filterd_operand_locs}) > 2"
    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.add_condition(third_cond)
    macro_instr.set_field_by_name('NS_ID', "VMEM1")
    macro_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    macro_instr.set_field_flex_param('NS_INDEX_ID', "0")
    macro_instr.set_field_value('IMM', 1)
    #

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.add_condition(third_cond)
    sub_instr.set_field_by_name('NS_ID', "VMEM1")
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_flex_param('NS_INDEX_ID', "0")
    sub_instr.set_field_flex_param('IMM', f"0")
    macro_instr.add_base_instruction(sub_instr)
    sub_instr = hag.get_primitive_template("SET_ITER")
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_flex_param("LOOP_ID", f"0")
    sub_instr.set_field_flex_param("NUM_ITER", loop_count)
    macro_instr.add_base_instruction(sub_instr)

    sub_instr = hag.get_primitive_template("SET_INDEX")
    sub_instr.add_condition(third_cond)
    sub_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    sub_instr.set_field_by_name("DST_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("DST_INDEX_ID", "0")
    sub_instr.set_field_by_name("SRC1_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID", "0")
    sub_instr.set_field_by_name("SRC2_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", '0')
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    noop_instr = hag.get_primitive_template("NOP")
    noop_instr.add_condition(third_cond)

    noop_instr.set_print_tabs("op.loop_level - op.loop_level//2 + 1")
    instructions.append(noop_instr)

    return instructions



def sw_addr_nops_nested(op_name, hag):
    instructions = []

    ## First, need to do BASE/SIGN_EXT

    loop_idx_offset = 0 if op_name != "POW" else 1


    ## Now, create the loop
    ## First loop
    all_loop_list = LOOP_ITER[1]
    other_constr = " and ".join(LOOP_CONDS + ["loop_op.op_str in op.dependencies"])
    ns_idx = f"(loop_op.loop_level % {ALL_LOOP_ID})"

    tgt_loop_list = f"[loop_op.iter_count // cdlt.param_tiling[2][cdlt.loop_param_map[loop_op.op_str]] for loop_op in {all_loop_list} if {other_constr}]"
    filtered_loops = f"[loop_op for loop_op in {all_loop_list} if {other_constr}]"
    loop_count = f"np.prod({tgt_loop_list})"
    filterd_operand_locs = "([op.get_operand_location(o.name) for o in op.operands if op.get_operand_location(o.name) != 'IMM'])"

    multiplier = f"(3 if len({filterd_operand_locs}) > 2 else 2)"
    multiplier_val = f"({multiplier} if loop_op.loop_id == {filtered_loops}[0].loop_id else 1)"

    lconds = " and ".join(LOOP_CONDS)

    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.add_iterable(*LOOP_ITER)
    macro_instr.add_condition(lconds)
    macro_instr.set_field_by_name('NS_ID', "VMEM1")
    macro_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    macro_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    macro_instr.set_field_value('IMM', 0)
    #

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.add_iterable(*LOOP_ITER)
    sub_instr.add_condition(lconds)
    sub_instr.set_field_by_name('NS_ID', "VMEM1")
    sub_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    sub_instr.set_field_flex_param('IMM', 0)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    ## Now generate loops

    macro_instr = hag.get_primitive_template("SET_ITER")
    macro_instr.set_print_tabs("loop_op.loop_level")
    macro_instr.add_iterable(*LOOP_ITER)
    macro_instr.add_condition(other_constr)
    macro_instr.set_field_flex_param("LOOP_ID", f"(loop_op.loop_level % {ALL_LOOP_ID}) + {loop_idx_offset}")
    macro_instr.set_field_flex_param("NUM_ITER", f"(loop_op.iter_count // cdlt.param_tiling[2][cdlt.loop_param_map[loop_op.op_str]])*{multiplier_val}")


    sub_instr = hag.get_primitive_template("SET_INDEX")
    sub_instr.set_print_tabs("loop_op.loop_level")
    sub_instr.add_iterable(*LOOP_ITER)
    sub_instr.add_condition(other_constr)
    sub_instr.set_field_by_name("DST_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("DST_INDEX_ID", ns_idx)
    sub_instr.set_field_by_name("SRC1_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID", ns_idx)
    sub_instr.set_field_by_name("SRC2_NS_ID", "VMEM1")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", ns_idx)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    # Compute instruction
    instr = hag.get_primitive_template("SET_INST")
    instr.add_condition("cdlt.op_id_counters['compute'] - 1 > op.op_id")
    instr.set_field_flex_param("SINGLE_NESTED", "1")
    instr.set_field_flex_param("NUM_INSTR", "1")
    instructions.append(instr)

    noop_instr = hag.get_primitive_template("NOP")
    instructions.append(noop_instr)


    return instructions


def obuf_mv_vmem_test(op_name, hag):

    ## First, need to do BASE/SIGN_EX


    ## Now, create the loop
    ## First loop
    instructions = []

    buffer_name = "VMEM1"
    all_obuf_ops = f"([o for o in cdlt.get_ops_by_type('compute') if 'OBUF' in o.source_locations])"
    obuf_cond = f"len({all_obuf_ops}) > 0 and op == {all_obuf_ops}[-1]"

    other_buff = "VMEM2" if buffer_name == "VMEM1" else "VMEM1"
    n_banks = f"hag.get_subgraph_node('{buffer_name}').banks"
    loop_id_str = f"0"
    ld_st_tabs = f"op.loop_level + 1"
    buff_name_str = f"'OBUF'"
    operand = f"([o for o in op.operands if op.get_operand_location(o.name) == 'OBUF'][0])"

    # ns_idx = f"{all_loop_id} + op.operand.get_mem_index({buff_name_str}) * {all_loop_id}"
    ns_idx = f"0"
    imm_base_sign_ext = f"{operand}.get_mem_offset({buff_name_str})//{operand}.dtype.bits()"
    base_sign_ext_low = f"program.extract_bits({imm_base_sign_ext}, 16, 0)"
    base_sign_ext_high = f"program.extract_bits({imm_base_sign_ext}, 16, 16)"
    bitwidth = f"len(np.binary_repr({imm_base_sign_ext})) + int(np.signbit({imm_base_sign_ext}))"
    bitwidth_cond = f"{bitwidth} <= 16"


    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.add_condition(obuf_cond)
    macro_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    macro_instr.set_field_by_name('NS_ID', "OBUF")
    macro_instr.set_field_flex_param('NS_INDEX_ID', f"{loop_id_str}")
    macro_instr.set_field_value('IMM', 0)

    micro_instr1 = hag.get_primitive_template("STRIDE_SIGN_EXT")
    micro_instr1.set_print_tabs("op.loop_level - op.loop_level//2")
    micro_instr1.add_condition(obuf_cond)
    micro_instr1.set_field_by_name('NS_ID', "OBUF")
    micro_instr1.set_field_flex_param('NS_INDEX_ID', f"{loop_id_str}")
    micro_instr1.set_field_value('IMM', 0)
    macro_instr.add_base_instruction(micro_instr1)

    instructions.append(macro_instr)

    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    macro_instr.add_condition(f"{obuf_cond} and {bitwidth_cond}")
    macro_instr.set_field_by_name('NS_ID', buffer_name)
    macro_instr.set_field_flex_param('NS_INDEX_ID', f"{ns_idx}")
    macro_instr.set_field_flex_param('IMM', imm_base_sign_ext)
    instructions.append(macro_instr)

    low_instr = hag.get_primitive_template("BASE_LOW")
    low_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    low_instr.add_condition(f"{obuf_cond} and not {bitwidth_cond}")
    low_instr.set_field_by_name('NS_ID', buffer_name)
    low_instr.set_field_flex_param('NS_INDEX_ID', f"{ns_idx}")
    low_instr.set_field_flex_param('IMM', base_sign_ext_low)
    instructions.append(low_instr)

    high_instr = hag.get_primitive_template("BASE_HIGH")
    high_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    high_instr.add_condition(f"{obuf_cond} and not {bitwidth_cond}")
    high_instr.set_field_by_name('NS_ID', buffer_name)
    high_instr.set_field_flex_param('NS_INDEX_ID', f"{ns_idx}")
    high_instr.set_field_flex_param('IMM', base_sign_ext_high)
    instructions.append(high_instr)



    micro_instr1 = hag.get_primitive_template("STRIDE_SIGN_EXT")
    micro_instr1.set_print_tabs("op.loop_level - op.loop_level//2")
    micro_instr1.add_condition(obuf_cond)
    micro_instr1.set_field_by_name('NS_ID', buffer_name)
    micro_instr1.set_field_flex_param('NS_INDEX_ID', f"{ns_idx}")
    micro_instr1.set_field_value('IMM', 1)
    instructions.append(micro_instr1)

    iter_instr = hag.get_primitive_template("SET_ITER")
    iter_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    iter_instr.add_condition(obuf_cond)
    iter_instr.set_field_flex_param("LOOP_ID", f"{loop_id_str}")
    iter_instr.set_field_flex_param("NUM_ITER",
                                   f"{operand}.get_tile_size('OBUF', 'pe_array')//{n_banks}")
    instructions.append(iter_instr)

    idx_instr = hag.get_primitive_template("SET_INDEX")
    idx_instr.set_print_tabs("op.loop_level - op.loop_level//2")
    idx_instr.add_condition(obuf_cond)
    idx_instr.set_field_by_name("DST_NS_ID", f"{buffer_name}")
    idx_instr.set_field_flex_param("DST_INDEX_ID", f"{ns_idx}")
    idx_instr.set_field_by_name("SRC1_NS_ID", "OBUF")
    idx_instr.set_field_flex_param("SRC1_INDEX_ID", f"0")
    idx_instr.set_field_by_name("SRC2_NS_ID", other_buff)
    idx_instr.set_field_flex_param("SRC2_INDEX_ID", f"0")
    instructions.append(idx_instr)

    set_inst_instr = hag.get_primitive_template("SET_INST")
    set_inst_instr.set_print_tabs("op.loop_level - op.loop_level//2")

    set_inst_instr.add_condition(obuf_cond)
    set_inst_instr.set_field_flex_param("SINGLE_NESTED", "1")
    set_inst_instr.set_field_flex_param("NUM_INSTR", "1")
    instructions.append(set_inst_instr)


    move_instr = hag.get_primitive_template(f"MOVE")
    move_instr.add_condition(obuf_cond)
    move_instr.set_field_by_name("DST_NS_ID", f"{buffer_name}")
    move_instr.set_field_flex_param("DST_INDEX_ID", f"{ns_idx}")
    move_instr.set_field_by_name("SRC1_NS_ID", f"OBUF")
    move_instr.set_field_flex_param("SRC1_INDEX_ID", f"0")
    move_instr.set_field_by_name("SRC2_NS_ID", other_buff)
    move_instr.set_field_flex_param("SRC2_INDEX_ID", f"0")
    move_instr.set_print_tabs(ld_st_tabs)

    instructions.append(move_instr)


    return instructions

def single_base_sign_ext(operand, hag: ArchitectureNode, cond=None):
    instructions = []
    operand_loc = f"op.get_operand_location({operand}.name)"
    ns_idx = f"(loop_op.loop_level % {ALL_LOOP_ID}) + ({operand}.get_mem_index({operand_loc}) * {ALL_LOOP_ID}) if op.get_operand_location({operand}.name) != 'IMM' else cdlt.temps.index({operand})"
    base_sign_ext = BASE_SIGN_EXT_TEMPLATE.format(OPERAND=operand, OP_LOC=operand_loc)
    base_sign_ext_low = f"program.extract_bits({base_sign_ext}, 16, 0)"
    base_sign_ext_high = f"program.extract_bits({base_sign_ext}, 16, 16)"
    rd_write_operand = RD_WRITE_OPERAND_TEMPLATE.format(OPERAND=operand)
    loop_stride = LOOP_STRIDE_TEMPLATE.format(OPERAND=operand, RD_WRITE=rd_write_operand)
    bitwidth = f"len(np.binary_repr({base_sign_ext})) + int(np.signbit({base_sign_ext}))"
    bitwidth_cond = f"{bitwidth} <= 16"
    single_base_ext_conds = LOOP_CONDS + [bitwidth_cond]
    multi_base_ext_conds = LOOP_CONDS + [f"not {bitwidth_cond}"]
    cond = cond or "True"
    assert isinstance(cond, str)
    ## First, instructions for sign ext lower than 16 bits
    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.add_iterable(*LOOP_ITER)
    macro_instr.add_condition(" and ".join([cond] + single_base_ext_conds))
    macro_instr.set_field_flex_param('NS_ID', operand_loc)
    macro_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    macro_instr.set_field_flex_param('IMM', base_sign_ext)

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.add_iterable(*LOOP_ITER)
    sub_instr.add_condition(" and ".join([cond] + single_base_ext_conds))
    sub_instr.set_field_flex_param('NS_ID', operand_loc)
    sub_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    sub_instr.set_field_flex_param('IMM', loop_stride)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    ## NExt, instructions for sign ext greater than 16 bits
    macro_instr = hag.get_primitive_template("BASE_LOW")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.add_iterable(*LOOP_ITER)
    macro_instr.add_condition(" and ".join([cond] + multi_base_ext_conds))
    macro_instr.set_field_flex_param('NS_ID', operand_loc)
    macro_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    macro_instr.set_field_flex_param('IMM', base_sign_ext_low)

    sub_instr = hag.get_primitive_template("BASE_HIGH")
    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.add_iterable(*LOOP_ITER)
    sub_instr.add_condition(" and ".join([cond] + multi_base_ext_conds))
    sub_instr.set_field_flex_param('NS_ID', operand_loc)
    sub_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    sub_instr.set_field_flex_param('IMM', base_sign_ext_high)
    macro_instr.add_base_instruction(sub_instr)

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.add_iterable(*LOOP_ITER)
    sub_instr.add_condition(" and ".join([cond] + multi_base_ext_conds))
    sub_instr.set_field_flex_param('NS_ID', operand_loc)
    sub_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    sub_instr.set_field_flex_param('IMM', loop_stride)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)
    return instructions

def single_operand_noop(op_name, operand, hag: ArchitectureNode, cond=None):
    instructions = []
    loop_idx_offset = 0 if op_name != "POW" else 1
    cond = cond or "True"
    other_constr = LOOP_CONDS + ["loop_op.op_str in op.dependencies"]
    macro_instr = hag.get_primitive_template("SET_ITER")
    macro_instr.set_print_tabs("loop_op.loop_level")
    macro_instr.add_iterable(*LOOP_ITER)
    macro_instr.add_condition(" and ".join(other_constr + [cond]))
    macro_instr.set_field_flex_param("LOOP_ID", f"(loop_op.loop_level % {ALL_LOOP_ID}) + {loop_idx_offset}")
    macro_instr.set_field_flex_param("NUM_ITER", f"loop_op.iter_count // cdlt.param_tiling[2][cdlt.loop_param_map[loop_op.op_str]]")


    sub_instr = hag.get_primitive_template("SET_INDEX")
    set_index_fmt = "(loop_op.loop_level % {all_loop_id}) + ({operand}.get_mem_index({op_loc}) * {all_loop_id}) if op.get_operand_location({operand}.name) != 'IMM' else cdlt.temps.index({operand})"

    sub_instr.set_print_tabs("loop_op.loop_level")
    sub_instr.add_iterable(*LOOP_ITER)
    sub_instr.add_condition(" and ".join(other_constr + [cond]))
    sub_instr.set_field_flex_param("DST_NS_ID", f"op.get_operand_location({operand}.name)")
    sub_instr.set_field_flex_param("DST_INDEX_ID",
                                   set_index_fmt.format(all_loop_id=ALL_LOOP_ID, operand=operand,
                                                        op_loc=f"op.get_operand_location({operand}.name)"))
    sub_instr.set_field_flex_param("SRC1_NS_ID", f"op.get_operand_location({operand}.name)")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID", "0")
    sub_instr.set_field_flex_param("SRC2_NS_ID", f"op.get_operand_location({operand}.name)")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", "0")
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    instr = hag.get_primitive_template("SET_INST")
    instr.add_condition(cond)
    instr.set_field_flex_param("SINGLE_NESTED", "0")
    instr.set_field_flex_param("NUM_INSTR", "1")
    instructions.append(instr)
    noop_instr = hag.get_primitive_template("NOP")
    noop_instr.add_condition(cond)
    noop_instr.set_print_tabs("op.loop_level")
    instructions.append(noop_instr)
    return instructions

def ld_st_overhead(op_name, hag: ArchitectureNode):
    first_read_cond = "{OPERAND}.get_first_read('SIMD') == op.op_str and ({OPERAND} in cdlt.inputs or op.get_operand_location({OPERAND}.name) == 'OBUF')"
    first_write_cond = "{OPERAND}.get_first_write('SIMD') == op.op_str and {OPERAND} in cdlt.outputs"
    instructions = []
    op0 = "op.sources[0]"
    instructions += single_base_sign_ext(op0, hag, cond=first_read_cond.format(OPERAND=op0))
    instructions += single_operand_noop(op_name, op0, hag, first_read_cond.format(OPERAND=op0))

    op1 = f"op.sources[1]"
    op1_cond = first_read_cond.format(OPERAND=op1)
    op1_cond = "len(op.sources) > 1 and op.get_operand_location(op.sources[1].name) != 'IMM' and " + op1_cond
    instructions += single_base_sign_ext(op1, hag, cond=op1_cond)
    instructions += single_operand_noop(op_name, op1, hag, cond=op1_cond)

    dst = f"op.dests[0]"
    dst_cond = first_write_cond.format(OPERAND=dst)
    instructions += single_base_sign_ext(dst, hag, cond=dst_cond)
    instructions += single_operand_noop(op_name, dst, hag, cond=dst_cond)
    return instructions

def obuf_read_overhead(hag):
    instructions = []
    obuf_read_cond = "op.get_operand_location({OPERAND}.name) == 'OBUF'"
    op0 = "op.sources[0]"
    op0_cond = obuf_read_cond.format(OPERAND=op0)
    instructions += single_base_sign_ext(op0, hag, cond=op0_cond)
    instructions += single_operand_noop("ADD", op0, hag, op0_cond)
    instructions += single_base_sign_ext(op0, hag, cond=op0_cond)
    instructions += single_operand_noop("ADD", op0, hag, op0_cond)

    op1 = "op.sources[1]"
    op1_cond = "len(op.sources) > 1 and " + obuf_read_cond.format(OPERAND=op1)
    instructions += single_base_sign_ext(op1, hag, cond=op1_cond)
    instructions += single_operand_noop("ADD", op1, hag, op1_cond)
    instructions += single_base_sign_ext(op1, hag, cond=op1_cond)
    instructions += single_operand_noop("ADD", op1, hag, op1_cond)
    return instructions


def simd_alu_template(op_name, hag: ArchitectureNode):
    if op_name == "TRANSPOSE":
        return simd_transpose(hag)
    instructions = []

    loop_idx_offset = 0 if op_name != "POW" else 1

    if op_name in DTYPE_CAST_NAMES:
        instructions += insert_dtype_cfg_from_cast(op_name, hag)
    ### Base and stride first
    instructions += base_sign_ext_gen(op_name, hag)


    if op_name == "POW":
        # Additional loop requires additional base/offset
      instructions += add_pow_loop(hag)

    ### iters and index
    other_constr = LOOP_CONDS + ["loop_op.op_str in op.dependencies"]
    macro_instr = hag.get_primitive_template("SET_ITER")
    macro_instr.set_print_tabs("loop_op.loop_level")
    macro_instr.add_iterable(*LOOP_ITER)
    macro_instr.add_condition(" and ".join(other_constr))
    macro_instr.set_field_flex_param("LOOP_ID", f"(loop_op.loop_level % {ALL_LOOP_ID}) + {loop_idx_offset}")
    macro_instr.set_field_flex_param("NUM_ITER", f"loop_op.iter_count // cdlt.param_tiling[2][cdlt.loop_param_map[loop_op.op_str]]")


    sub_instr = hag.get_primitive_template("SET_INDEX")
    set_index_fmt = "(loop_op.loop_level % {all_loop_id}) + ({operand}.get_mem_index({op_loc}) * {all_loop_id}) if op.get_operand_location({operand}.name) != 'IMM' else cdlt.temps.index({operand})"

    sub_instr.set_print_tabs("loop_op.loop_level")
    sub_instr.add_iterable(*LOOP_ITER)
    sub_instr.add_condition(" and ".join(other_constr))
    sub_instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
    sub_instr.set_field_flex_param("DST_INDEX_ID",
                                   set_index_fmt.format(all_loop_id=ALL_LOOP_ID, operand="op.dests[0]",
                                                        op_loc="op.get_operand_location(op.dests[0].name)"))
    sub_instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
    sub_instr.set_field_flex_param("SRC1_INDEX_ID",
                                   set_index_fmt.format(all_loop_id=ALL_LOOP_ID, operand="op.sources[0]",
                                                        op_loc="op.get_operand_location(op.sources[0].name)"))
    src2_ns = "'IMM' if op.get_operand_location(op.sources[0].name) != 'IMM' else 'VMEM1'"
    sub_instr.set_field_flex_param("SRC2_NS_ID",
                                   f"({src2_ns}) if len(op.sources) == 1 else op.get_operand_location(op.sources[1].name)")
    src2_idx = f"0 if len(op.sources) == 1 else " + set_index_fmt.format(all_loop_id=ALL_LOOP_ID,
                                                                         operand="op.sources[1]",
                                                                         op_loc="op.get_operand_location(op.sources[1].name)")
    sub_instr.set_field_flex_param("SRC2_INDEX_ID", src2_idx)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    # Compute instruction
    instr = hag.get_primitive_template("SET_INST")
    instr.add_condition("cdlt.op_id_counters['compute'] - 1 > op.op_id")
    instr.set_field_flex_param("SINGLE_NESTED", "0 if op.num_loop_dependencies == 1 else 1")
    instr.set_field_flex_param("NUM_INSTR", "1")
    instructions.append(instr)
    rd_op_str = "op.sources[{IDX}].get_mem_index(op.get_operand_location(op.sources[{IDX}].name)) if op.get_operand_location(op.sources[{IDX}].name) != 'IMM' else cdlt.temps.index(op.sources[{IDX}])"

    if op_name != "POW":
        instr = hag.get_primitive_template(op_name)
        instr.add_condition("len(op.sources) > 1")
        instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
        instr.set_field_flex_param("DST_INDEX_ID", f"op.dests[0].get_mem_index(op.get_operand_location(op.dests[0].name))")
        instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
        instr.set_field_flex_param("SRC1_INDEX_ID", rd_op_str.format(IDX=0))
        instr.set_field_flex_param("SRC2_NS_ID", "op.get_operand_location(op.sources[1].name)")
        instr.set_field_flex_param("SRC2_INDEX_ID", rd_op_str.format(IDX=1))
        instructions.append(instr)

        instr = hag.get_primitive_template(op_name)
        instr.add_condition("len(op.sources) == 1")
        instr.set_field_flex_param("DST_NS_ID", "op.get_operand_location(op.dests[0].name)")
        instr.set_field_flex_param("DST_INDEX_ID", f"op.dests[0].get_mem_index(op.get_operand_location(op.dests[0].name))")
        instr.set_field_flex_param("SRC1_NS_ID", "op.get_operand_location(op.sources[0].name)")
        instr.set_field_flex_param("SRC1_INDEX_ID", rd_op_str.format(IDX=0))
        instr.set_field_flex_param("SRC2_NS_ID", f"{src2_ns}")
        instr.set_field_value("SRC2_INDEX_ID", 0)
        instructions.append(instr)
    else:
        instructions += add_pow_compute(hag, rd_op_str)


    instructions += alu_noop(op_name, hag)
    instr = hag.get_primitive_template("SYNC_INST")
    all_obuf_ops = f"([o for o in cdlt.get_ops_by_type('compute') if 'OBUF' in o.source_locations])"
    # instr.add_condition("any([op.get_operand_location(o.name) == 'OBUF' for o in op.operands])")
    instr.add_condition(f"len({all_obuf_ops}) > 0 and op == {all_obuf_ops}[-1]")
    instr.set_field_by_name("COMPUTE_TARGET", "SIMD")
    instr.set_field_by_name("START_END", "END")
    instr.set_field_by_name("EXEC_BUF", "BUF")
    instr.set_field_flex_param("GROUP_NUM", "(cdlt.instance_id - 1) % 64")
    instr.set_field_flex_param("NUM_INSTR", "0")
    instructions.append(instr)

    if hag.meta_cfg['ADDR_GEN_TEST']:
        instructions += sw_addr_nops_nested(op_name, hag)
    elif hag.meta_cfg['LOOP_OVERHEAD']:
        instructions += loop_overhead(op_name, hag)
    elif hag.meta_cfg['OBUF_TO_VMEM_TEST']:
        instructions += obuf_mv_vmem_test(op_name, hag)
    elif hag.meta_cfg['LD_ST_OVERHEAD']:
        instructions += ld_st_overhead(op_name, hag)
    elif hag.meta_cfg['TPU_TEST']:
        instructions += loop_overhead(op_name, hag)
        instructions += ld_st_overhead(op_name, hag)
        instructions += obuf_read_overhead(hag)

    return instructions



def base_sign_ext_gen(op_name, hag: ArchitectureNode):
    instructions = []

    # Index generation
    # ns_idx = f"(loop_op.loop_id % {ALL_LOOP_ID}) + (operand.get_mem_index({OPERAND_LOC}) * {ALL_LOOP_ID}) if op.get_operand_location(operand.name) != 'IMM' else cdlt.temps.index(operand)"

    ns_idx = f"(loop_op.loop_level % {ALL_LOOP_ID}) + (operand.get_mem_index({OPERAND_LOC}) * {ALL_LOOP_ID}) if op.get_operand_location(operand.name) != 'IMM' else cdlt.temps.index(operand)"
    # base_sign_ext = f"(operand.get_mem_offset({OPERAND_LOC})//(operand.dtype.bits()) + 1) if op.get_operand_location(operand.name) " \
    #                 f"!= 'IMM' else cdlt.temps.index(operand)"

    # base_sign_ext = f"(operand.get_mem_offset({OPERAND_LOC})//(hag.get_subgraph_node({OPERAND_LOC}).data_size) + 1) if op.get_operand_location(operand.name) " \
    #                 f"!= 'IMM' else cdlt.temps.index(operand)"
    # base_sign_ext = f"(operand.get_mem_offset({OPERAND_LOC})) if op.get_operand_location(operand.name) " \
    #                 f"!= 'IMM' else cdlt.temps.index(operand)"

    base_sign_ext_low = f"program.extract_bits({BASE_SIGN_EXT}, 16, 0)"
    base_sign_ext_high = f"program.extract_bits({BASE_SIGN_EXT}, 16, 16)"


    bitwidth = f"len(np.binary_repr({BASE_SIGN_EXT})) + int(np.signbit({BASE_SIGN_EXT}))"
    bitwidth_cond = f"{bitwidth} <= 16"
    single_base_ext_conds = LOOP_CONDS + [bitwidth_cond]
    multi_base_ext_conds = LOOP_CONDS + [f"not {bitwidth_cond}"]

    ## First, instructions for sign ext lower than 16 bits
    macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.add_iterable(*OPERAND_ITER)
    macro_instr.add_iterable(*LOOP_ITER)
    macro_instr.add_condition(" and ".join(single_base_ext_conds))
    macro_instr.set_field_flex_param('NS_ID', OPERAND_LOC)
    macro_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    macro_instr.set_field_flex_param('IMM', BASE_SIGN_EXT)

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.add_iterable(*OPERAND_ITER)
    sub_instr.add_iterable(*LOOP_ITER)
    sub_instr.add_condition(" and ".join(single_base_ext_conds))
    sub_instr.set_field_flex_param('NS_ID', OPERAND_LOC)
    sub_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    sub_instr.set_field_flex_param('IMM', LOOP_STRIDE)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)


    ## NExt, instructions for sign ext greater than 16 bits
    macro_instr = hag.get_primitive_template("BASE_LOW")
    macro_instr.set_print_tabs(ALL_LOOP_ID)
    macro_instr.add_iterable(*OPERAND_ITER)
    macro_instr.add_iterable(*LOOP_ITER)
    macro_instr.add_condition(" and ".join(multi_base_ext_conds))
    macro_instr.set_field_flex_param('NS_ID', OPERAND_LOC)
    macro_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    macro_instr.set_field_flex_param('IMM', base_sign_ext_low)

    sub_instr = hag.get_primitive_template("BASE_HIGH")
    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.add_iterable(*OPERAND_ITER)
    sub_instr.add_iterable(*LOOP_ITER)
    sub_instr.add_condition(" and ".join(multi_base_ext_conds))
    sub_instr.set_field_flex_param('NS_ID', OPERAND_LOC)
    sub_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    sub_instr.set_field_flex_param('IMM', base_sign_ext_high)
    macro_instr.add_base_instruction(sub_instr)

    sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
    sub_instr.set_print_tabs(ALL_LOOP_ID)
    sub_instr.add_iterable(*OPERAND_ITER)
    sub_instr.add_iterable(*LOOP_ITER)
    sub_instr.add_condition(" and ".join(multi_base_ext_conds))
    sub_instr.set_field_flex_param('NS_ID', OPERAND_LOC)
    sub_instr.set_field_flex_param('NS_INDEX_ID', ns_idx)
    sub_instr.set_field_flex_param('IMM', LOOP_STRIDE)
    macro_instr.add_base_instruction(sub_instr)
    instructions.append(macro_instr)

    return instructions


def alu_noop(op_name, hag):
    all_loop_id = ALL_LOOP_ID

    instructions = []
    if OP_COMPUTE_CYCLES[op_name] > 0:
        macro_instr = hag.get_primitive_template("BASE_SIGN_EXT")
        macro_instr.set_field_by_name('NS_ID', "VMEM1")
        macro_instr.set_print_tabs("op.loop_level - 1")
        macro_instr.set_field_flex_param('NS_INDEX_ID', "0")
        macro_instr.set_field_value('IMM', 1)
        #

        sub_instr = hag.get_primitive_template("STRIDE_SIGN_EXT")
        sub_instr.set_field_by_name('NS_ID', "VMEM1")
        sub_instr.set_print_tabs("op.loop_level - 1")
        sub_instr.set_field_flex_param('NS_INDEX_ID', "0")
        sub_instr.set_field_flex_param('IMM', f"0")
        macro_instr.add_base_instruction(sub_instr)
        if op_name == "DIV":
            noop_iters = f"{OP_COMPUTE_CYCLES[op_name] + hag.get_subgraph_node('SIMD').dimensions[0]}"
        else:
            noop_iters = f"{OP_COMPUTE_CYCLES[op_name]}"

        sub_instr = hag.get_primitive_template("SET_ITER")
        sub_instr.set_print_tabs("op.loop_level - 1")
        sub_instr.set_field_flex_param("LOOP_ID", f"op.loop_id % {all_loop_id}")
        sub_instr.set_field_flex_param("NUM_ITER", f"{noop_iters}")
        macro_instr.add_base_instruction(sub_instr)

        sub_instr = hag.get_primitive_template("SET_INDEX")
        sub_instr.set_print_tabs("op.loop_level + 1")
        sub_instr.set_field_by_name("DST_NS_ID", "VMEM1")
        sub_instr.set_field_flex_param("DST_INDEX_ID", "0")
        sub_instr.set_field_by_name("SRC1_NS_ID", "VMEM1")
        sub_instr.set_field_flex_param("SRC1_INDEX_ID", "0")
        sub_instr.set_field_by_name("SRC2_NS_ID", "VMEM1")
        sub_instr.set_field_flex_param("SRC2_INDEX_ID", '0')
        macro_instr.add_base_instruction(sub_instr)
        instructions.append(macro_instr)

        instr = hag.get_primitive_template("SET_INST")
        instr.set_field_flex_param("SINGLE_NESTED", "1")
        instr.set_field_flex_param("NUM_INSTR", "1")
        instructions.append(instr)

        noop_instr = hag.get_primitive_template("NOP")
        noop_instr.set_print_tabs("op.loop_level")
        instructions.append(noop_instr)
    return instructions