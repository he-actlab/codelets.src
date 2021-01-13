from dataclasses import dataclass, field
from collections import defaultdict, deque
from codelets.adl import Instruction
from typing import List, Dict, Union, Optional

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
                 instruction_template: List[Instruction] = None,
                 extra_params: Dict[str, Union[str, int]]=None,
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
            self._loop_level = Operation.loop_ctxt_level
            self._loop_id = Operation.current_loop_id()
            self._global_op_id = Operation.id_counter
            self._op_id = Operation.op_id_counters[operation_type]

            Operation.op_id_counters[operation_type] += 1
            Operation.id_counter += 1

        self._target = target
        self._operation_type = operation_type
        self._instruction_template = instruction_template or []
        self._extra_params = extra_params or {}
        self._required_params = required_params

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
    def global_op_id(self) -> int:
        return self._global_op_id

    @property
    def instructions(self) -> List[Instruction]:
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

    def __str__(self):
        op_str = f"{self.op_type}{self.op_id} -> {self.target}, PARAMS: {list(self.required_params)}, " \
                 f"{self.op_type.upper()}PARAMS: {self.op_type_params()}"
        return op_str

    def op_args_copy(self, cdlt):
        kwargs = {}
        kwargs['extra_params'] = self.extra_params.copy()
        kwargs['loop_id'] = self.loop_id
        kwargs['loop_level'] = self.loop_level
        kwargs['global_op_id'] = self.global_op_id
        kwargs['op_id'] = self.op_id
        kwargs['target'] = self.target
        kwargs['instruction_template'] = [i.instruction_copy() for i in self.instructions]
        return kwargs

    @classmethod
    def copy_op(cls, cdlt, op):
        args = op.op_type_args_copy(cdlt)
        kwargs = op.op_args_copy(cdlt)
        return cls(*args, add_codelet=False, copy_init=True, **kwargs)

    def op_type_params(self):
        raise NotImplementedError

    def set_template(self, opcode, op_fields):
        pass


