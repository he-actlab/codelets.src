from codelets import Datatype

OP_LOCATIONS = {"OBUF": 0, "IBUF": 1, "VMEM": 2, "IMM": 3, "EXTMEM": 4}
SIMD_OP_READ_NS = ["OBUF", "VMEM", "IMM"]
SIMD_OP_WRITE_NS = ["IBUF", "VMEM", "IMM"]
SIMD_NS = ["OBUF", "IBUF", "VMEM", "IMM"]

OP_DTYPES = [Datatype(type='FXP', bitwidth=8), Datatype(type='FXP', bitwidth=16), Datatype(type='FXP', bitwidth=32),
             Datatype(type='FP', bitwidth=16), Datatype(type='FP', bitwidth=32)]

SIMD_OPCODE_BITWIDTH = 4
SIMD_FNCODE_BITWIDTH = 4
NS_BITWIDTH = 5
NS_IDX_BITWIDTH = 3

from .genesys import generate_genesys, define_genesys
