from typing import List, Dict, Union

from functools import singledispatch
from .dummy_op import DummyOp, DummyParam
from .operand_template import OperandTemplate, IndexOperandTemplate, Offset
from .micro_template import MicroTemplate

class ControlTemplate(MicroTemplate):
    PARAM_KEYS = ['control_type']
    USE_DUMMY_STRING = [False]
    def __init__(self, control_type: str, param_map,
                 add_codelet=True):
        super(ControlTemplate, self).__init__(control_type, param_map, add_codelet=add_codelet)

    def __enter__(self):
        if self.op_type == 'loop':
            MicroTemplate.loop_ctxt_level += 1
            MicroTemplate.loop_stack.append(self.loop_id)
            MicroTemplate.loop_ctx_dependencies.append(self.op_str)
        MicroTemplate.block.append(MicroTemplate.current_block)
        MicroTemplate.current_block = []
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.op_type == "loop":
            MicroTemplate.loop_ctxt_level -= 1
            MicroTemplate.loop_stack.pop()
            MicroTemplate.loop_ctx_dependencies.pop()
        MicroTemplate.finished_blocks.append(MicroTemplate.current_block)
        MicroTemplate.current_block = MicroTemplate.block.pop()

class LoopTemplate(ControlTemplate):
    PARAM_KEYS = ['start', 'end', 'stride', 'offset']
    USE_DUMMY_STRING = [True, True, True, True]
    loop_ids = 0

    def __init__(self, start: Union[int, DummyOp, DummyParam],
                 end=None,
                 stride=1,
                 offset=0,
                 add_codelet=True,
                 **kwargs
                 ):
        param_map = {}
        if end is not None:
            param_map['start'] = start
            param_map['end'] = end
        else:
            param_map['start'] = 0
            param_map['end'] = start

        param_map['stride'] = stride
        param_map['offset'] = offset

        super(LoopTemplate, self).__init__("loop", {**param_map, **kwargs}, add_codelet=add_codelet)

    def __str__(self):
        return f"{self.op_str}: START: {self.start}; END: {self.end}; STRIDE: {self.stride}"

    @property
    def positional_args(self):
        return LoopTemplate.PARAM_KEYS

    @property
    def arg_dummy_strings(self):
        return LoopTemplate.USE_DUMMY_STRING

    def __mul__(self, other):
        return loop_mul(other, self)

    def __rmul__(self, other):
        return loop_mul(other, self)

    def __add__(self, other):
        return loop_add(other, self)

    def __radd__(self, other):
        return loop_add(other, self)

    @property
    def start(self):
        return self.param_map['start']

    @property
    def end(self):
        return self.param_map['end']

    @property
    def stride(self):
        return self.param_map['stride']

@singledispatch
def loop_mul(op1, op2):
    raise NotImplementedError(f"No implementation for multiplying loop with operand type:"
                              f"Operand1: {op1}, type: {type(op1)}\n"
                              f"Operand2: {op2}, type: {type(op2)}.")

@singledispatch
def loop_add(op1, op2):
    raise NotImplementedError(f"No implementation for adding loop with operand type:"
                              f"Operand1: {op1}, type: {type(op1)}\n"
                              f"Operand2: {op2}, type: {type(op2)}.")

@loop_mul.register
def _(op1: int, op2: LoopTemplate):
    sign = '+' if np.sign(op1) == 1 else '-'
    base = op2.start
    scale = op2.stride*op1
    return Offset(base, [scale], [op2.op_str], [sign])

@loop_mul.register
def _(op1: DummyOp, op2: LoopTemplate):
    sign = '+'
    base = op2.start
    scale = op2.stride*op1
    return Offset(base, [scale], [op2.op_str], [sign])

@loop_add.register
def _(op1: int, op2: LoopTemplate):
    sign = '+'
    base = op2.start + op1
    scale = op2.stride
    return Offset(base, [scale], [op2.op_str], [sign])

@loop_add.register
def _(op1: LoopTemplate, op2: LoopTemplate):
    sign = '+'
    return Offset(op1.start + op2.start,
                  [op1.stride, op2.stride],
                  [op1.op_str, op2.op_str],
                  [sign, sign])

@loop_add.register
def _(op1: Offset, op2: LoopTemplate):
    op1.base = op1.base + op2.start
    op1.scales.append(op2.stride)
    op1.loops.append(op2.op_str)
    op1.signs.append('+')
    return op1