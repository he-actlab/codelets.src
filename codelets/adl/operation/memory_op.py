from .base import Operation
from typing import List
from dataclasses import field, dataclass
from collections import deque, defaultdict

class Comm(Operation):

    # TODO: Remove op name
    def __init__(self, op_name, sources, dests, src_offset, dst_offset, size=0, template=None):
        name = f"{op_name}{Operation.op_id_counters['memory']}"
        loop_id = Operation.current_loop_id()
        loop_level = Operation.loop_ctxt_level
        self._src_offset = src_offset
        self._dst_offset = dst_offset
        self._op_name = op_name
        self._sources = sources
        self._dests = dests
        self._size = size
        super(Comm, self).__init__(name, 'memory', loop_id, loop_level, instruction_template=template)

    @property
    def sources(self):
        return self._sources

    @property
    def dests(self):
        return self._dests

    @property
    def size(self):
        return self._size

    @property
    def op_name(self):
        return self._op_name
