from dataclasses import dataclass, field
from collections import defaultdict, deque
from codelets.adl import InstructionTemplate
from typing import List, Dict, Union, Optional

class Operation(object):
    id_counter = 0
    op_id_counters = defaultdict(int)
    loop_ctxt_level = 0
    loop_stack = deque()

    def __init__(self, name: str,
                 operation_type: str,
                 loop_id: str,
                 loop_level: int,
                 instruction_template: List[InstructionTemplate] = None,
                 extra_params: Dict[str, int]=None):
        self._name = name
        self._loop_id = loop_id

        self._operation_type = operation_type
        self._loop_level = loop_level
        self._instruction_template = instruction_template or []
        self._extra_params = extra_params or {}
        self._global_op_id = Operation.id_counter
        self._op_id = Operation.op_id_counters[operation_type]
        Operation.op_id_counters[operation_type] += 1
        Operation.id_counter += 1

    @property
    def loop_id(self):
        return self._loop_id

    @property
    def name(self):
        return self._name

    @property
    def loop_level(self):
        return self._loop_level

    @property
    def is_template(self) -> bool:
        return self.loop_id is not None

    @staticmethod
    def current_loop_id():
        return Operation.loop_stack[-1]

    def set_template(self, opcode, op_fields):
        pass
