from .base_op import Operation
from typing import Dict, Union
from codelets.adl.flex_param import FlexParam
from dataclasses import field, dataclass
from collections import deque, defaultdict

class Configure(Operation):

    def __init__(self, start_or_finish,
                 target=None,
                 add_codelet=True,
                 **kwargs
                 ):
        self._target_name = target
        self._start_or_finish = start_or_finish
        super(Configure, self).__init__('config', [],
                                        target=target,
                                        add_codelet=add_codelet, **kwargs)
    @property
    def target_name(self):
        return self._target_name

    @property
    def start_or_finish(self):
        return self._start_or_finish

    def op_type_params(self):
        op_params = [f"{self.start_or_finish}"]
        return op_params

    def evaluate_parameters(self, node, hag, cdlt):
        pass

    def emit(self, output_type):
        # TODO: Add template
        if output_type == "operations":
            op_str = f"{self.op_str}: {self.start_or_finish}-{self.target}"
        elif output_type == "json":
            op_str = {"op_type": self.op_type,
                      "op_id": self.global_op_id,
                      "start_or_finish": self.start_or_finish,
                      "target": self.target}
        else:
            op_str = []
            for ft in self.instructions:
                op_str += ft.emit(output_type)
        return op_str

    def copy(self, cdlt):
        obj = super(Configure, self).copy(cdlt)
        obj._target_name = self.target
        obj._start_or_finish = self.start_or_finish
        return obj

