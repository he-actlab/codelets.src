from dataclasses import dataclass, field
from collections import defaultdict, deque
from codelets.adl.flex_template import Instruction, FlexTemplate
from typing import List, Dict, Union, Optional
from sympy import Basic, Idx, IndexedBase, Expr
from codelets.adl.flex_param import FlexParam

# INSTR_FN_TEMPLATE = """def param_fn{FN_ID}(hag, op, cdlt, relocation_table, program, fixed_val=None): return {FN_BODY}"""
OPERATION_TEMPLATE = """
<%inherit file="{CODELET_NAME}"/>
<%def field_name="emit(output_type)">
% if output_type == "op_string":
    ${{op.op_str}}
% else
    % for i in op.instruction_templates:
        ${{i.emit(output_type)}}
    % endfor
% endif
</%def>
"""

class Operation(object):
    id_counter = 0
    op_id_counters = defaultdict(int)
    loop_ctxt_level = 0
    loop_stack = deque()
    current_codelet = None

    def __init__(self, operation_type: str,
                 required_params: List[str],
                 codelet=None,
                 target: str = None,
                 instruction_template: List[FlexTemplate] = None,
                 extra_params: Dict[str, Union[str, int]] = None,
                 resolved_params: Dict[str, FlexParam] = None,
                 param_symbols: Dict[str, Basic] = None,
                 add_codelet=True,
                 loop_id=None,
                 loop_level=None,
                 global_op_id=None,
                 op_id=None,
                 copy_init=False):
        if copy_init:
            assert all([p is not None for p in [loop_id, loop_level, global_op_id, op_id]])
            self._loop_id = loop_id
            self._loop_level = loop_level
            self._global_op_id = global_op_id
            self._op_id = op_id
        else:
            self._loop_id = Operation.current_loop_id()
            self._loop_level = Operation.loop_ctxt_level
            self._global_op_id = Operation.id_counter
            self._op_id = Operation.op_id_counters[operation_type]
            if operation_type == "loop":
                self._loop_id = self._op_id

            Operation.op_id_counters[operation_type] += 1
            Operation.id_counter += 1
        self._param_symbols = param_symbols or {}
        self._target = target
        self._operation_type = operation_type
        self._instruction_template = instruction_template or []
        self._extra_params = extra_params or {}
        self._required_params = required_params
        self._resolved_params = resolved_params or {}
        assert self._loop_id >= 0 or self._operation_type == "config"
        codelet = codelet or Operation.current_codelet
        if add_codelet:
            codelet.add_op(self)

    @property
    def loop_id(self) -> int:
        return self._loop_id

    @property
    def required_params(self) -> List[str]:
        return self._required_params

    @property
    def extra_params(self) -> Dict[str, Union[str, int]]:
        return self._extra_params

    @property
    def target(self) -> str:
        return self._target

    @property
    def op_id(self) -> int:
        return self._op_id

    @property
    def resolved_params(self):
        return self._resolved_params

    @property
    def global_op_id(self) -> int:
        return self._global_op_id

    @property
    def instructions(self) -> List[FlexTemplate]:
        return self._instruction_template

    @property
    def loop_level(self):
        return self._loop_level

    @property
    def is_template(self) -> bool:
        return self.loop_id is not None

    @property
    def op_type(self) -> str:
        return self._operation_type

    @staticmethod
    def current_loop_id():
        return Operation.loop_stack[-1]

    @property
    def op_str(self):
        return f"{self.op_type}{self.op_id}"

    @property
    def param_symbols(self):
        return self._param_symbols

    def __str__(self):
        op_str = f"{self.op_type}{self.op_id} -> {self.target}, PARAMS: {list(self.required_params)}, " \
                 f"{self.op_type.upper()}PARAMS: {self.op_type_params()}"
        return op_str

    def instruction_str(self):
        instr_list = []
        for i in self.instructions:
            instr_list.append(str(i))
        return instr_list

    def op_args_copy(self, cdlt):
        kwargs = {}
        kwargs['extra_params'] = self.extra_params.copy()
        kwargs['loop_id'] = self.loop_id
        kwargs['loop_level'] = self.loop_level
        kwargs['global_op_id'] = self.global_op_id
        kwargs['op_id'] = self.op_id
        kwargs['target'] = self.target
        kwargs['instruction_template'] = [i.template_copy() for i in self.instructions]
        return kwargs

    def set_required_param(self, key, param):
        assert isinstance(param, FlexParam)
        value = param.value
        if key not in self.required_params:
            raise KeyError(f"Key {key} for updating param does not exist:\n"
                           f"All Keys: {self.required_params}\n"
                           f"Updated value: {value}")
        if key in self.resolved_params and self.resolved_params[key].value != value:
            raise RuntimeError(f"Param {key} has already been set:\n"
                               f"Previous value: {self.resolved_params[key]}\n"
                               f"New value: {value}")

        if value is None:
            raise ValueError(f"Cannot set required parameter to None value:\n"
                             f"Value: {value}\n"
                             f"Key: {key}")
        self.resolved_params[key] = param

    def unset_params(self):
        unset_params = []
        for k in self.required_params:
            if k not in self.resolved_params:
                unset_params.append(k)
        return unset_params

    @classmethod
    def copy_op(cls, cdlt, op):
        args = op.op_type_args_copy(cdlt)
        kwargs = op.op_args_copy(cdlt)
        return cls(*args, add_codelet=False, copy_init=True, **kwargs)

    def op_type_params(self):
        raise NotImplementedError

    def set_template(self, template: List[FlexTemplate]):
        self._instruction_template = template

    def emit(self, output_type):
        if output_type == "operations":
            op_str = f"{self.op_type}{self.op_id} -> {self.target}, PARAMS: {list(self.required_params)}, " \
                     f"{self.op_type.upper()}PARAMS: {self.op_type_params()}"
        else:
            op_str = []
            for ft in self.instructions:
                op_str += ft.emit(output_type)
        return op_str

    def evaluate_parameters(self, node, hag, cdlt):
        raise NotImplementedError

