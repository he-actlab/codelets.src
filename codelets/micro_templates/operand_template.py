from numbers import Number
from typing import List, Dict, Union, Tuple
from dataclasses import dataclass, field
from functools import singledispatch
import numpy as np

from codelets import Datatype
from codelets.adl.operation import Operand
from .dummy_op import DummyOp, DummyParam
OPERATION_TEMPLATE_CLASSES = ['MicroTemplate', 'LoopTemplate', 'ComputeTemplate', 'ConfigureTemplate',
                              'TransferTemplate']


@dataclass
class OperandTemplate:
    name: str
    location: str
    operand_type: str
    writes: List[str] = field(default_factory=list)
    reads: List[str] = field(default_factory=list)
    init_value: Number = field(default=None)
    init_dummy_op: DummyOp = field(default=None)
    shape_list: List[Union[int, DummyOp, DummyParam]] = field(default=None)
    default_dtype: Union[None, Datatype] = field(default=None)
    extra_kwargs: Dict[str, DummyOp] = field(default_factory=dict)

    def __getitem__(self, item: List[DummyParam]):
        if isinstance(item, (list, tuple)) and len(item) == 0:
            return self
        if isinstance(item, list):
            assert not isinstance(item[0], (tuple, list))
            item = tuple(item)
        elif not isinstance(item, tuple):
            item = tuple([item])
        offsets = []
        for i in item:
            if isinstance(i, (DummyOp, int)):
                offsets.append(Offset(i, [], [], []))
            elif isinstance(i, Offset):
                offsets.append(i)
            else:
                assert i.__class__.__name__ == 'LoopTemplate'
                offsets.append(Offset(i.start, [i.stride], [i.op_str], ['+']))
        return IndexOperandTemplate(self, tuple(offsets))

    def add_read(self, op_name: str):
        if op_name not in self.reads:
            self.reads.append(op_name)

    def add_write(self, op_name: str):
        if op_name not in self.writes:
            self.writes.append(op_name)

    def reorder_shapes(self, permutation: List[int]):
        assert len(permutation) == len(self.shape_list)
        self.shape_list = [self.shape_list[i] for i in permutation]

    def is_location_set(self):
        return self.location is not None

    def set_location(self, location: str):
        assert not self.is_location_set()
        self.location = location

    @property
    def shape_list_names(self):
        slist = []
        for s in self.shape_list:
            if isinstance(s, (DummyParam, DummyOp)):
                slist.append(s.name)
            else:
                slist.append(s)
        return slist

@dataclass
class Offset:
    base: Union[int, DummyOp]
    scales: List[Union[int, None]]
    loops: List[Union[str, None]]
    signs: List[Union[str]]
    # signs = + or -
    # offset = base (+) loops[0]*scales[0] (+) loops[1]*scales[1] ...
    @property
    def name(self):
        if len(self.loops) == 0:
            return str(self.base)
        else:
            return '+'.join(self.loops)

    def __mul__(self, other):
        return offset_mul(other, self)

    def __rmul__(self, other):
        return offset_mul(other, self)

    def __add__(self, other):
        return offset_add(other, self)

    def __radd__(self, other):
        return offset_add(other, self)

    def __str__(self):
        return self.affine_expr_str

    @property
    def affine_expr_str(self):
        if len(self.scales) == 0:
            return str(self.base)
        base_str = ""
        if self.base != 0:
            base_str += f"{self.base}+"
        offset_strs = []
        for i in range(len(self.scales)):
            if self.scales[i] != 1:
                offset_strs.append(f"{self.scales[i]}*{self.loops[i]}")
            else:
                offset_strs.append(f"{self.loops[i]}")
        return base_str + "+".join(offset_strs)

    def evaluate(self, instance_args):
        if isinstance(self.base, DummyOp):
            base_eval = self.base.evaluate(instance_args)
        loop_ranges = []
        for i in range(len(self.scales)):
            scale_eval = evaluate_args(self.scales[i], instance_args, tuple([]))

@dataclass
class IndexOperandTemplate:
    operand: OperandTemplate
    offsets: Tuple[Offset]

    def evaluate(self, instance_args):
        operand = instance_args['CodeletTemplate'].get_operand(self.operand.name)
        # TODO: Need to fix initialization so that evaluated dummy args are placed here
        evaluated_offset = evaluate_args(self.offsets, instance_args, tuple([]))
        assert isinstance(evaluated_offset, tuple)
        operand_idx = operand[evaluated_offset]
        return operand_idx

    def reorder_offsets(self, permutation: List[int]):
        assert len(permutation) == len(self.offsets)
        self.offsets = tuple([self.offsets[i] for i in permutation])

    @property
    def offset_names(self):
        return [o.name for o in self.offsets]

    @property
    def name(self):
        return self.operand.name

    def __str__(self):
        offset_str = ",".join([str(o) for o in self.offsets])
        return f"{self.operand}[{offset_str}]"


def evaluate_args(args, instance_args, preserve_types):


    if isinstance(args, list):
        eval_arg = []
        for a in args:
            eval_arg.append(evaluate_args(a, instance_args, preserve_types))
    elif isinstance(args, tuple):
        eval_arg = []
        for a in args:
            eval_arg.append(evaluate_args(a, instance_args, preserve_types))
        eval_arg = tuple(eval_arg)
    elif isinstance(args, (DummyParam, DummyOp)):
        eval_arg = args.evaluate(instance_args)
    elif args.__class__.__name__ in OPERATION_TEMPLATE_CLASSES:
        cdlt = instance_args['CodeletTemplate']
        eval_arg = cdlt.op_map[args.op_str]
    elif isinstance(args, OperandTemplate):
        cdlt = instance_args['CodeletTemplate']
        eval_arg = cdlt.get_operand(args.name)
    else:
        eval_arg = args

    if isinstance(args, preserve_types):
        return args

    return eval_arg


@singledispatch
def offset_mul(op1, op2):
    raise NotImplementedError(f"No implementation for multiplying loop with operand type:"
                              f"Operand1: {op1}, type: {type(op1)}\n"
                              f"Operand2: {op2}, type: {type(op2)}.")

@singledispatch
def offset_add(op1, op2):
    return op1.__add__(op2)

@offset_mul.register
def _(op1: int, op2: Offset):
    for i in range(len(op2.scales)):
        op2.scales[i] = op2.scales[i]*op1
    op2.base = op2.base * op1
    return op2

@offset_mul.register
def _(op1: DummyOp, op2: Offset):
    for i in range(len(op2.scales)):
        op2.scales[i] = op2.scales[i]*op1
    op2.base = op2.base * op1
    return op2

@offset_add.register
def _(op1: int, op2: Offset):
    op2.base = op2.base + op1
    return op2

@offset_add.register
def _(op1: Offset, op2: Offset):
    return Offset(op1.base + op2.base,
                  op1.scales + op2.scales,
                  op1.loops + op2.loops,
                  op1.signs + op2.signs)