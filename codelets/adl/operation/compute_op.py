from . import Operation, OperandTemplate
from typing import List, Dict, Union
from itertools import chain
from codelets.adl.flex_param import FlexParam
from dataclasses import field, dataclass
from collections import deque, defaultdict

class Compute(Operation):

    def __init__(self, op_name: str, sources: List[OperandTemplate], dests: List[OperandTemplate],
                 target: str=None,
                 add_codelet=True,
                 **kwargs):
        self._op_name = op_name
        self._sources = sources
        self._dests = dests
        req_params = []
        assert target is not None
        # TODO: Need to figure out if these need to be added
        # TODO: Remove these checks during copy
        dependencies = []
        for s in sources:
            if s.sorted_tile_keys[-1] != target:
                raise RuntimeError(f"Invalid codelet: {op_name} executed on {target} has input {s.name} which is currently stored in {s.sorted_tile_keys[-1]}."
                                   f"An additional transfer operation from {s.sorted_tile_keys[-1]} to {target} is required.")
            dependencies += s.dependencies

        for d in dests:
            dependencies += d.dependencies
        dependencies = list(set(dependencies))
        super(Compute, self).__init__('compute', req_params,
                                      target=target,
                                      add_codelet=add_codelet,
                                      dependencies=dependencies,
                                      **kwargs)
        for d in dests:
            if self.op_str not in d.dependencies:
                d.dependencies.append(self.op_str)

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

    def op_type_args_copy(self, cdlt):
        sources = [cdlt.get_operand(s.name) for s in self.sources]
        dests = [cdlt.get_operand(d.name) for d in self.dests]

        return (self.op_name, sources, dests)

    def evaluate_parameters(self, node, hag, cdlt):
        pass

    def emit(self, output_type):
        # TODO: Add template
        if output_type == "operations":
            source_names = [s.name for s in self.sources]
            dst_names = [d.name for d in self.dests]
            op_str = f"{self.op_str}: {self.target}-{self.op_name}({source_names})->{dst_names}"
        elif output_type == "json":
            op_str = {"op_type": self.op_type,
                      "op_id": self.global_op_id,
                      "operation_name": self.op_name,
                      "target": self.target,
                      "sources": self.sources,
                      "destinations": self.dests}
        else:
            op_str = []
            for ft in self.instructions:
                op_str += ft.emit(output_type)
        return op_str

    def copy(self, cdlt):
        obj = super(Compute, self).copy(cdlt)
        obj._op_name = self.op_name
        obj._sources = [cdlt.get_operand(s.name) for s in self.sources]
        obj._dests = [cdlt.get_operand(d.name) for d in self.dests]
        return obj

