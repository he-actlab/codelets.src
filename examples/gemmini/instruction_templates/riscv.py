# type: ignore

from __future__ import annotations
from dataclasses import dataclass, field
from codelets.adl.flex_template.instruction import Field, Instruction

from .instr_base import Operation, Data

from typing import Union, TypeVar, Type


RD1RS1IMM1_OPS = []
RS2IMM1_OPS = []
RS2OFF1_OPS = []
RD1RS2_OPS = []
RD1OFF1_OPS = []
RD1RS1OFF1_OPS = []
NOPARAMS_OPS = []
RD1IMM1_OPS = []
RD1RS1_OPS = []
RS1OFF1_OPS = []
RS1RT1OFF1_OPS = []
AUX_OPS = []

# All load instr
RD1RS1IMM1_OPS += ["lb", "lbu", "lh", "lhu", "lw"]

# All store instr
RS2IMM1_OPS += ["sb", "sh", "sw"]

# Branch instr
RS2OFF1_OPS += ["beq", "bne", "blt", "bge", "bltu", "bgeu"]

# Shift instr
RD1RS2_OPS += ["sll", "srl", "sra"]

RD1RS1IMM1_OPS += ["slli", "srli", "srai"]

# Arithmetic
RD1RS2_OPS += ["add", "sub"]
RD1RS1IMM1_OPS += ["addi", "lui", "auipc"]

# Logical
RD1RS2_OPS += ["xor", "or", "and"]
RD1RS1IMM1_OPS += ["xori", "ori", "andi"]

# Compare
RD1RS2_OPS += ["slt", "sltu"]
RD1RS1IMM1_OPS += ["slti", "sltui"]

# Jump & Link
RD1OFF1_OPS += ["jal"]
RD1RS1OFF1_OPS += ["jalr"]


# System
NOPARAMS_OPS += ["scall", "sbreak"]

#  Optional Multiply-Devide Instruction Extension (RVM)
RD1RS2_OPS += ["mul", "mulh", "mulhsu", "divu", "rem", "remu"]

# Pseudo ops
NOPARAMS_OPS += ["nop"]
RD1IMM1_OPS += ["li"]
RD1RS1_OPS += ["mv", "not", "neg", "negw", "seqz", "snez", "sltz", "sgtz"]
RS1OFF1_OPS += ["beqz", "bnez", "blez", "bgez", "bltz", "bgtz"]
RS1RT1OFF1_OPS += ["bgt", "ble", "bgtu", "bleu", "ret"]

# Aux Ops
AUX_OPS += ["label", "directive"]


ABI_NAMES = {
        "zero": 0,
        "ra": 1,
        "sp": 2,
        "gp": 3,
        "tp": 4,
        "t0": 5,
        "t1": 6,
        "t2": 7,
        "fp": 8,
        "s0": 8,
        "s1": 9,
        "a0": 10,
        "a1": 11,
        "a2": 12,
        "a3": 13,
        "a4": 14,
        "a5": 15,
        "a6": 16,
        "a7": 17,
        "s2": 18,
        "s3": 19,
        "s4": 20,
        "s5": 21,
        "s6": 22,
        "s7": 23,
        "s8": 24,
        "s9": 25,
        "s10": 26,
        "s11": 27,
        "t3": 28,
        "t4": 29,
        "t5": 30,
        "t6": 31
    }

def create_riscv():
    pass

def create_rd1rs1imm1_ops():
    rd = Field("rd", 5)
    rs = Field("rs", 5)
    imm = Field("imm", 12)

def create_rs2imm1_ops():
    pass

def create_rs2off1_ops():
    pass

def create_rd1rs2_ops():
    pass

def create_rd1off1_ops():
    pass

def create_rd1rs1off1_ops():
    pass

def create_noparams_ops():
    pass

def create_rd1imm1_ops():
    pass

def create_rd1rs1_ops():
    pass

def create_rs1off1_ops():
    pass

def create_rs1rt1off1_ops():
    pass

def create_aux_ops():
    pass




