from .base_op import Operation
from typing import List
from dataclasses import field, dataclass
from collections import deque, defaultdict

class Compute(Operation):

    def __init__(self, op_name: str, sources: List[str], dests: List[str],
                 target: str=None,
                 add_codelet=True,
                 **kwargs):
        self._op_name = op_name
        self._sources = sources
        self._dests = dests
        req_params = []

        for i in sources:
            if isinstance(i, str):
                req_params.append(i)

        for o in dests:
            if isinstance(o, str):
                req_params.append(o)

        super(Compute, self).__init__('compute', req_params,
                                      target=target,
                                      add_codelet=add_codelet,
                                      **kwargs)

    @property
    def sources(self):
        return self._sources

    @property
    def dests(self):
        return self._dests

    @property
    def op_name(self):
        return self._op_name

    def op_type_params(self):
        op_params = [f"OP: {self.op_name}", f"SRC: {self.sources}", f"DST: {self.dests}"]
        return op_params

    def op_type_args_copy(self, _):
        return (self.op_name, self.sources.copy(), self.dests.copy())

