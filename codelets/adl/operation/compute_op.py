from . import Operation, OperandTemplate, get_transfer_dim_sizes, IndexedOperandTemplate
from typing import List, Dict, Union, Callable
from itertools import chain
from codelets.adl.flex_param import FlexParam
from dataclasses import field, dataclass
from collections import deque, defaultdict

class Compute(Operation):

    def __init__(self, op_name: str,
                 sources: List[Union[OperandTemplate, IndexedOperandTemplate]],
                 dests: List[Union[OperandTemplate, IndexedOperandTemplate]],
                 target: str=None,
                 add_codelet=True,
                 **kwargs):
        self._op_name = op_name
        self._sources = []
        self._dests = []
        req_params = []
        assert target is not None

        # TODO: Need to figure out if these need to be added
        # TODO: Remove these checks during copy
        dependencies = []


        super(Compute, self).__init__('compute', req_params,
                                      target=target,
                                      add_codelet=add_codelet,
                                      dependencies=dependencies,
                                      **kwargs)
        for s_call in sources:
            s_call.test_add_compute_access(target, self.op_str, "source")
            if isinstance(s_call, OperandTemplate):
                s = s_call.add_data_access([], self.op_str, "read", [], path_key=(self.target,))
            else:
                s = s_call.operand_template.add_data_access(s_call.offsets, self.op_str, "read", [], path_key=(self.target,))

            self._dependencies += [dep for dep in s.dependencies if dep not in dependencies]
            self._sources.append(s)

        for d_call in dests:
            d_call.test_add_compute_access(target, self.op_str, "dest")

            if isinstance(d_call, OperandTemplate):
                d = d_call.add_data_access([], self.op_str, "write", [], access_node=self.target, path_key=(self.target,))
            else:
                d = d_call.operand_template.add_data_access(d_call.offsets, self.op_str, "write", [], access_node=self.target, path_key=(self.target,))
            self._dependencies += [dep for dep in d.dependencies if dep not in dependencies]
            d.dependencies.append(self.op_str)
            self._dests.append(d)

    @property
    def sources(self):
        return self._sources

    @property
    def dests(self):
        return self._dests

    @property
    def operands(self):
        return self._sources + self._dests

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
        # for d in self.sources:
        #     print(d.operand.tiling)
        #     print(d.operand.evaluated_tiling)
        #     print()
        # for d in self.dests:
        #     print(d.operand.tiling)
        #     print(d.operand.evaluated_tiling)
        # for s in self.sources:
        #     path_key = (s.data_source, self.target)
        #     src_shape, dst_shape = get_transfer_dim_sizes(s, path_key)
        #
        # for d in self.dests:
        #     path_key = (self.target)
        #     src_shape, dst_shape = get_transfer_dim_sizes(d, path_key)


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

    def copy(self, cdlt, op_name=None, sources=None, dests=None, **kwargs):
        obj = super(Compute, self).copy(cdlt, **kwargs)

        obj._op_name = op_name or self.op_name
        obj._sources = sources or [s.copy() for s in self.sources]
        obj._dests = dests or [d.copy() for d in self.dests]
        return obj

