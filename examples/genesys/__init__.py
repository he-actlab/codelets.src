from codelets import Datatype

OP_LOCATIONS = {"OBUF": 0, "IBUF": 1, "VMEM": 2, "IMM": 3, "EXTMEM": 4}
SIMD_OP_READ_NS = ["OBUF", "VMEM", "IMM"]
SIMD_OP_WRITE_NS = ["IBUF", "VMEM", "IMM"]
SIMD_NS = ["OBUF", "IBUF", "VMEM", "IMM"]

OP_DTYPES = [Datatype(type='FXP', bitwidth=8), Datatype(type='FXP', bitwidth=16), Datatype(type='FXP', bitwidth=32),
             Datatype(type='FP', bitwidth=16), Datatype(type='FP', bitwidth=32), Datatype(type='FXP', bitwidth=4)]

DTYPE_MAP = {}
DTYPE_MAP['FXP32'] = Datatype(type='FXP', bitwidth=32)
DTYPE_MAP['FXP16'] = Datatype(type='FXP', bitwidth=16)
DTYPE_MAP['FXP8'] = Datatype(type='FXP', bitwidth=8)
DTYPE_MAP['FXP4'] = Datatype(type='FXP', bitwidth=4)
DTYPE_MAP['FP32'] = Datatype(type='FXP', bitwidth=32)
DTYPE_MAP['FP16'] = Datatype(type='FXP', bitwidth=16)

GENESYS_DTYPES = {}
GENESYS_DTYPES['SIMD'] = 'FXP32'
GENESYS_DTYPES['SYSTOLIC_ARRAY'] = {}
GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight'] = 'FXP8'
GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out'] = 'FXP32'

BIT = 1
BYTE = 8
GENESYS_CFG = {}
GENESYS_CFG['ARRAY_N'] = 64
GENESYS_CFG['ARRAY_M'] = 64
GENESYS_CFG['DATA_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight']].bits()
GENESYS_CFG['WGT_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight']].bits()
GENESYS_CFG['BIAS_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out']].bits()
GENESYS_CFG['ACC_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out']].bits()
GENESYS_CFG['INSTR_WIDTH'] = 32
GENESYS_CFG['SIMD_WIDTH'] = GENESYS_CFG['ARRAY_M']
GENESYS_CFG['PARAM_BUF_CHANNEL_BW'] = 512 // BIT
GENESYS_CFG['IBUF_CHANNEL_BW'] = 512 // BIT
GENESYS_CFG['OBUF_CHANNEL_BW'] = 512 // BIT
GENESYS_CFG['INSTR_CHANNEL_BW'] = 512 // BIT
GENESYS_CFG['SIMD_CHANNEL_BW'] = 512 // BIT

GENESYS_CFG['IBUF_DEPTH'] = 2048
GENESYS_CFG['WBUF_DEPTH'] = 512
GENESYS_CFG['OBUF_DEPTH'] = 2048
GENESYS_CFG['BBUF_DEPTH'] = 2048
GENESYS_CFG['INSTR_DEPTH'] = 1024
GENESYS_CFG['IMM_DEPTH'] = 1
GENESYS_CFG['DRAM_DEPTH'] = 100000
GENESYS_CFG['VMEM_DEPTH'] = GENESYS_CFG['OBUF_DEPTH']

GENESYS_CFG['VMEM_BANKS'] = GENESYS_CFG['SIMD_WIDTH']
GENESYS_CFG['INSTR_BANKS'] = 1
GENESYS_CFG['DRAM_BANKS'] = 1
GENESYS_CFG['DRAM_WIDTH'] = 32

SIMD_OPCODE_BITWIDTH = 4
SIMD_FNCODE_BITWIDTH = 4
NS_BITWIDTH = 5
NS_IDX_BITWIDTH = 3

from .genesys import define_genesys, compile_genesys, compile_genesys_layer, get_transformed_srdfg, \
    compile_extracted_genesys_layer, get_arch
