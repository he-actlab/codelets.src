from codelets.adl.flex_template.instruction import Field, Instruction

ARITHMETIC_NAMES = ["ADD", "SUB", "MUL", "DIV", "LT", "LTE",
                    "GT", "GTE", "EQ", "NEQ"]

UNARY_NAMES = ["SIGMOID", "GAUSSIAN", "SQRT", "SIGMOID_SYMM",
               "LOG", "SQUARE"]

COMM_NAMES = ["PASS"]

ARITH_OPS = {i: op for i, op in enumerate(ARITHMETIC_NAMES)}
UNARY_OPS = {i + 16: op for i, op in enumerate(UNARY_NAMES)}
COMM_OPS = {i + 24: op for i, op in enumerate(COMM_NAMES)}

def arith_instr():
    src1 = Field("SRC1", 6)
    src2 = Field("SRC2", 6)
