from collections import defaultdict, deque
from codelets.adl.operation import Loop, Compute, Configure, Transfer
from codelets.common.flex_param import FlexParam
from functools import singledispatch
from typing import List, Dict, Union, Any
from copy import copy
import numpy as np
from .dummy_op import DummyOp, DummyParam
from .operand_template import OperandTemplate, IndexOperandTemplate, Offset

INITIALIZER_FN_MAP = {'loop': Loop,
                      'config': Configure,
                      'transfer': Transfer,
                      'compute': Compute}

class MicroTemplate(object):

    BASE_ATTRS = ['loop_id', 'loop_level', 'op_id', 'global_op_id', 'target', 'operation_type',
                  'dependencies', 'param_symbols', 'target', 'operation_type', 'instructions',
                  'required_params', 'resolved_params', 'dependencies']
    BASE_KWARG_NAMES = ['codelet', 'target', 'instructions', 'resolved_params', 'param_symbols',
                        'add_codelet', 'dependencies']
    id_counter = 0
    op_id_counters = defaultdict(int)
    loop_ctx_dependencies = deque()
    loop_ctxt_level = 0
    loop_stack = deque()
    current_codelet = None
    current_block = []
    block = deque()
    finished_blocks = []

    def __init__(self, operation_type: str,
                 param_map: Dict[str, Union[FlexParam, int, None]],
                 codelet=None,
                 add_codelet=True,
                 dependencies=None):

        self._loop_id = MicroTemplate.current_loop_id()
        self._loop_level = copy(MicroTemplate.loop_ctxt_level)
        self._global_op_id = MicroTemplate.id_counter
        self._op_id = copy(MicroTemplate.op_id_counters[operation_type])
        if operation_type == "loop":
            self._loop_id = self._op_id

        MicroTemplate.op_id_counters[operation_type] += 1
        MicroTemplate.id_counter += 1
        self._dependencies = dependencies or []
        self._operation_type = operation_type
        self._output_operand = None
        self._param_map = param_map
        codelet = codelet or MicroTemplate.current_codelet
        if add_codelet:
            codelet.add_op(self)

        MicroTemplate.current_block.append(self.op_str)

    @property
    def loop_id(self) -> int:
        return self._loop_id

    @loop_id.setter
    def loop_id(self, loop_id: int):
        self._loop_id = loop_id

    @property
    def param_map(self) -> Dict[str, Any]:
        return self._param_map

    @property
    def op_id(self) -> int:
        return self._op_id

    @op_id.setter
    def op_id(self, op_id: int):
        self._op_id = op_id

    @property
    def global_op_id(self) -> int:
        return self._global_op_id

    @global_op_id.setter
    def global_op_id(self, global_op_id: int):
        self._global_op_id = global_op_id

    @property
    def loop_level(self):
        return self._loop_level

    @loop_level.setter
    def loop_level(self, loop_level: int):
        self._loop_level = loop_level

    @property
    def is_template(self) -> bool:
        return self.loop_id is not None

    @property
    def op_type(self) -> str:
        return self._operation_type

    @property
    def output_operand(self) -> OperandTemplate:
        return self._output_operand

    @staticmethod
    def current_loop_id():
        return MicroTemplate.loop_stack[-1]

    @property
    def op_str(self):
        return f"{self.op_type}{self.op_id}"

    @property
    def dependencies(self):
        return self._dependencies

    @dependencies.setter
    def dependencies(self, dependencies):
        self._dependencies = dependencies

    @staticmethod
    def reset():
        MicroTemplate.id_counter = 0
        MicroTemplate.op_id_counters = defaultdict(int)
        MicroTemplate.loop_ctx_dependencies = deque()
        MicroTemplate.loop_ctxt_level = 0
        MicroTemplate.loop_stack = deque()
        MicroTemplate.current_codelet = None

    def set_output_operand(self, operand: OperandTemplate):
        self._output_operand = operand

    def evaluate(self, instance_args):
        return instance_args['CodeletTemplate'].op_map[self.op_str]

    @property
    def positional_args(self):
        raise NotImplementedError

    def instantiate(self, instance_args):
        args = []
        for i, param_name in enumerate(self.positional_args):
            param = self.param_map[param_name]
            use_dummy_str = self.arg_dummy_strings[i]
            args.append(self.evaluate_args(param, instance_args, use_dummy_str))
        args = tuple(args)
        kwargs = {}
        for key, value in self.param_map.items():
            if key in self.positional_args:
                continue
            else:
                kwargs[key] = self.evaluate_args(value, instance_args, False)

        kwargs['add_codelet'] = False
        instance = INITIALIZER_FN_MAP[self.op_type](*args, **kwargs)
        return instance

    def evaluate_args(self, args, instance_args, use_dummy_string):
        if use_dummy_string and isinstance(args, (DummyParam, DummyOp)):
            return args.name

        if isinstance(args, list):
            eval_arg = []
            for a in args:
                eval_arg.append(self.evaluate_args(a, instance_args, use_dummy_string))
        elif isinstance(args, tuple):
            eval_arg = []
            for a in args:
                eval_arg.append(self.evaluate_args(a, instance_args, use_dummy_string))
            eval_arg = tuple(eval_arg)
        elif isinstance(args, (DummyParam, DummyOp, IndexOperandTemplate)):
            eval_arg = args.evaluate(instance_args)
        elif isinstance(args, MicroTemplate):
            cdlt = instance_args['CodeletTemplate']
            eval_arg = cdlt.op_map[args.op_str]
        elif isinstance(args, OperandTemplate):
            cdlt = instance_args['CodeletTemplate']
            eval_arg = cdlt.get_operand(args.name)
        else:
            eval_arg = args

        return eval_arg

    def __str__(self):
        return f"{self.op_str}"

    def __repr__(self):
        return f"<op={self.op_type}, id={self.op_id}>"

    @property
    def arg_dummy_strings(self):
        raise NotImplementedError

class ConfigureTemplate(MicroTemplate):
    PARAM_KEYS = ['start_or_finish', 'target']
    USE_DUMMY_STRING = [False, False]

    def __init__(self, start_or_finish,
                 target,
                 add_codelet=True,
                 **kwargs
                 ):
        param_map = {}
        param_map['start_or_finish'] = start_or_finish
        param_map['target'] = target
        super(ConfigureTemplate, self).__init__("config", {**param_map, **kwargs}, add_codelet=add_codelet)

    @property
    def positional_args(self):
        return ConfigureTemplate.PARAM_KEYS

    @property
    def arg_dummy_strings(self):
        return ConfigureTemplate.USE_DUMMY_STRING

class ComputeTemplate(MicroTemplate):
    PARAM_KEYS = ['op_name', 'sources', 'target']
    USE_DUMMY_STRING = [True, False, False, False]
    def __init__(self, op_name: str,
                 sources: List[OperandTemplate],
                 compute_target: str,
                 add_codelet=True,
                 **kwargs
                 ):
        param_map = {}
        param_map['op_name'] = op_name
        param_map['target'] = compute_target
        param_map['sources'] = sources
        super(ComputeTemplate, self).__init__("compute", {**param_map, **kwargs}, add_codelet=add_codelet)
        for s in sources:
            assert isinstance(s, (OperandTemplate, IndexOperandTemplate))
            s.add_read(self.op_str)

    def __str__(self):
        return f"{self.output_operand.name} = {self.op_str}('{self.op_name}'; ARGS={self.operand_names}; TGT: {self.target}"

    @property
    def operand_names(self):
        names = []
        for o in self.sources:
            if isinstance(o, OperandTemplate):
                names.append(o.name)
            else:
                names.append(str(o))
        return tuple(names)


    @property
    def op_name(self):
        return self.param_map['op_name']

    @property
    def sources(self):
        return self.param_map['sources']

    @property
    def target(self):
        return self.param_map['target']

    @property
    def positional_args(self):
        return ComputeTemplate.PARAM_KEYS

    @property
    def arg_dummy_strings(self):
        return ComputeTemplate.USE_DUMMY_STRING

class TransferTemplate(MicroTemplate):
    PARAM_KEYS = ['src_op', 'src_offset', 'dst_op', 'dst_offset', 'size']
    USE_DUMMY_STRING = [False, True, True, False, True, True]

    def __init__(self, source: Union[OperandTemplate, IndexOperandTemplate],
                 destination: Union[OperandTemplate, IndexOperandTemplate],
                 size,
                 add_codelet=True,
                 **kwargs
                 ):
        assert isinstance(source, (OperandTemplate, IndexOperandTemplate))
        assert isinstance(destination, (str, DummyOp, OperandTemplate, IndexOperandTemplate))
        param_map = {}
        if isinstance(source, IndexOperandTemplate):
            src_offset = source.offsets
            src_op = source.operand
        else:
            src_offset = [Offset(0, [], [], [])]
            src_op = source

        if isinstance(destination, IndexOperandTemplate):
            dst_offset = destination.offsets
            dst_op = destination.operand
        else:
            dst_offset = [Offset(0, [], [], [])]
            dst_op = destination

        param_map['src_op'] = src_op
        param_map['src_offset'] = src_offset
        param_map['size'] = size
        param_map['dst_op'] = dst_op
        param_map['dst_offset'] = dst_offset

        super(TransferTemplate, self).__init__("transfer", {**param_map, **kwargs}, add_codelet=add_codelet)
        assert isinstance(src_op, OperandTemplate) and isinstance(dst_op, OperandTemplate)
        self.src_op.add_read(self.op_str)
        if self.dst_op.src_op is None:
            self.dst_op.set_source_op(self.op_str)
        self.set_output_operand(self.dst_op)

    def __str__(self):
        src_str = f"SRC:('{self.src_location}', {self.src_op.name}{self.src_offset_str})"
        dst_str = f"DST:('{self.dst_location}', {self.dst_op.name}{self.dst_offset_str}) "
        return f"{self.output_operand.name} = {self.op_str}({src_str} --> {dst_str})"

    @property
    def positional_args(self):
        return TransferTemplate.PARAM_KEYS

    @property
    def arg_dummy_strings(self):
        return TransferTemplate.USE_DUMMY_STRING

    @property
    def src_op(self) -> OperandTemplate:
        return self.param_map['src_op']

    @property
    def dst_op(self) -> OperandTemplate:
        return self.param_map['dst_op']

    @property
    def src_offset(self) -> OperandTemplate:
        return self.param_map['src_offset']

    @property
    def src_offset_str(self) -> List[str]:
        return [str(so) for so in self.src_offset]

    @property
    def dst_offset(self) -> OperandTemplate:
        return self.param_map['dst_offset']

    @property
    def dst_offset_str(self) -> List[str]:
        return [str(do) for do in self.dst_offset]

    @property
    def src_location(self):
        return self.src_op.location

    @property
    def dst_location(self):
        return self.dst_op.location

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