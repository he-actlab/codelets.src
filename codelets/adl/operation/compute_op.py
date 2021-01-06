from .base import Operation
from dataclasses import field, dataclass
from collections import deque, defaultdict

class Compute(Operation):

    def __init__(self, op_name, target, inputs, outputs, template):
        name = f"{op_name}{Operation.op_id_counters['compute']}"
        loop_id = Operation.current_loop_id()
        loop_level = Operation.loop_ctxt_level
        self._op_name = op_name
        self._target = target
        self._inputs = inputs
        self._outputs = outputs
        super(Compute, self).__init__(name, 'compute', loop_id, loop_level, instruction_template=template)


    @property
    def target(self):
        return self._target

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def op_name(self):
        return self._op_name
