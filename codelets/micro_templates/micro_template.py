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
        self._op_id = copy(MicroTemplate.op_id_counters[operation_type])
        if operation_type == "loop":
            self._loop_id = self._op_id
        self._global_op_id = MicroTemplate.id_counter
        MicroTemplate.op_id_counters[operation_type] += 1
        MicroTemplate.id_counter += 1
        self._dependencies = dependencies or []
        self._operation_type = operation_type
        self._output_operand = None
        self._param_map = param_map
        self._param_options = {k: None for k in param_map.keys()}
        codelet = codelet or MicroTemplate.current_codelet
        if add_codelet:
            codelet.add_op(self)

        MicroTemplate.current_block.append(self.op_str)

    @property
    def global_op_id(self) -> int:
        return self._global_op_id

    @global_op_id.setter
    def global_op_id(self, global_op_id: int):
        self._global_op_id = global_op_id

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
    def param_options(self) -> Dict[str, Any]:
        return self._param_options

    @property
    def op_id(self) -> int:
        return self._op_id

    @op_id.setter
    def op_id(self, op_id: int):
        self._op_id = op_id

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

    def is_param_set(self, param_name: str):
        return self.param_map[param_name] is not None

    def has_param_options(self, param_name: str):
        return self.param_options[param_name] is not None

    def set_parameter(self, param_name: str, value):
        assert param_name in self.param_map and not self.is_param_set(param_name)
        self.param_map[param_name] = value

    def set_param_options(self, param_name: str, options):
        assert param_name in self.param_map and not self.is_param_set(param_name) and not self.has_param_options(param_name)
        self.param_options[param_name] = options


    @property
    def arg_dummy_strings(self):
        raise NotImplementedError

class ConfigureTemplate(MicroTemplate):
    PARAM_KEYS = ['config_key', 'config_value', 'target']
    USE_DUMMY_STRING = [False, False, False]

    def __init__(self, config_key,
                 config_value,
                 target,
                 add_codelet=True,
                 **kwargs
                 ):
        param_map = {}
        param_map['config_key'] = config_key
        param_map['config_value'] = config_value
        param_map['target'] = target
        super(ConfigureTemplate, self).__init__("config", {**param_map, **kwargs}, add_codelet=add_codelet)

    @property
    def positional_args(self):
        return ConfigureTemplate.PARAM_KEYS

    @property
    def arg_dummy_strings(self):
        return ConfigureTemplate.USE_DUMMY_STRING

    @property
    def target(self):
        return self.param_map['target']

    @property
    def config_key(self):
        return self.param_map['config_key']

    @property
    def config_value(self):
        return self.param_map['config_value']

    def is_target_set(self):
        return self.is_param_set('target')

