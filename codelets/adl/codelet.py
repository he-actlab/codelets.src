import numpy as np
from dataclasses import asdict
import polymath as pm
from collections import namedtuple, defaultdict
from .capability import CapabilityTemplate, Capability
from typing import Callable, Any, List, Dict, Optional, Tuple, Set, Union
from .operand import Datatype
from .operand import Operand
CodeletOperand = namedtuple('CodeletOperand', ['field_name', 'dtypes'])
Field = namedtuple('Field', ['field_name', 'value', 'bitwidth'])

# Operand = namedtuple('Operand', ['component_name', 'datatype', 'dimensions'])
# TODO does different on-chip dataflow need to be considered?
#      if so, how should it be included?
#      current implementation considers a single type of on-chip dataflow!



class Codelet(object):
    """
    Base class for capability
    """
    CONST_VAL = 'CONST'
    CODELET_COUNTER = defaultdict(int)
    def __init__(self, name,
                 input_dtypes,
                 output_dtypes,
                 capability_sequence=None,
                 op_params=None,
                 latency=None):


        self._name = name

        # list of input_components should be identical regardless of dataflows
        # 'input field_name': 'dims'
        self.input_dtypes = input_dtypes
        self.output_dtypes = output_dtypes
        self._capability_sequence = capability_sequence or []
        self.op_params = op_params or {}
        # latency can be either a fixed value or a lambda function
        self.latency = latency or 0
        self._cdlt_id = Codelet.CODELET_COUNTER[self.name]
        Codelet.CODELET_COUNTER[self.name] += 1

    @property
    def is_atomic(self) -> bool:
        return self._is_atomic

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_dtypes(self) -> List[Datatype]:
        return self._input_dtypes

    @property
    def codelet_id(self):
        return self._cdlt_id

    @property
    def output_dtypes(self) -> List[str]:
        return self._output_dtypes


    @property
    def latency(self) -> int:
        return self._latency

    @property
    def op_params(self) -> Dict[str, Any]:
        return self._op_params

    @property
    def capability_sequence(self) -> List[CapabilityTemplate]:
        return self._capability_sequence

    @output_dtypes.setter
    def output_dtypes(self, output_dtypes):
        self._output_dtypes = output_dtypes

    @input_dtypes.setter
    def input_dtypes(self, input_dtypes):
        self._input_dtypes = input_dtypes

    @is_atomic.setter
    def is_atomic(self, is_atomic):
        self._is_atomic = is_atomic

    @name.setter
    def name(self, name):
        self._name = name

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @op_params.setter
    def op_params(self, op_params):
        self._op_params = op_params


    @capability_sequence.setter
    def capability_sequence(self, capability_sequence):
        if len(self.capability_sequence) > 0:
            raise RuntimeError(f"Cannot set capability sequence which has already been set.")
        self._capability_sequence = capability_sequence

    def set_name(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def set_latency(self, latency):
        self._latency = latency

    def get_latency(self):
        return self._latency

    def add_capability(self, capability):
        self._capability_sequence.append(capability)

    def get_op_param(self, key):
        if key not in self._op_params:
            raise KeyError(f"Key {key} not in op params: {self.op_params.keys()}")
        return self.op_params[key]

    def add_op_param(self, key, value):
        if key in self._op_params:
            raise KeyError(f"Key {key} already in op params: {self.op_params.keys()}")
        self.op_params[key] = value

    def to_json(self) -> Dict:
        blob = {}
        blob['codelet_name'] = self.name
        blob['op_params'] = self.op_params
        blob['latency'] = self.latency
        blob['input_dtypes'] = [f"{d.type.upper()}{d.bitwidth}" for d in self.input_dtypes]
        blob['output_dtypes'] = [f"{d.type.upper()}{d.bitwidth}" for d in self.input_dtypes]
        blob['capability_sequence'] = [c.template_json() for c in self.capability_sequence]
        return blob

    def get_text_instructions(self) -> List[str]:
        instr = [str(c) for c in self.capability_sequence]
        return instr

    def emit_template_string(self):
        instrs = []
        for c in self.capability_sequence:
            instrs.append(str(c))
        return "\n".join(instrs)

    def emit_template_json(self):
        instrs = []
        for c in self.capability_sequence:
            instrs.append(c.to_json())
        return instrs

    def set_input_types(self, node: pm.Node) -> None:
        for i in node.inputs:
            if "hag_dtype" not in i.kwargs:
                i.add_attribute("hag_dtype", self.input_dtypes[0])
            else:
                assert i.kwargs["hag_dtype"] in self.input_dtypes

    def set_output_types(self, node: pm.Node) -> None:
        for o in node.outputs:
            if "hag_dtype" not in o.kwargs:
                o.add_attribute("hag_dtype", self.output_dtypes[0])
            else:
                assert o.kwargs["hag_dtype"] in self.output_dtypes

    def instantiate_codelet(self, node, hag, program):
        instance_params = {}
        self.set_input_types(node)
        self.set_output_types(node)

        for k, v in self.op_params.items():
            if isinstance(v, Callable):
                instance_params[k] = v(node)

            # TODO: Need to make sure all nodes have input/output defined
        return CodeletInstance(node.inputs, node.outputs, self, op_params=instance_params)

class CodeletInstance(object):
    def __init__(self, inputs: List[pm.Node], outputs: List[pm.Node], codelet, op_params=None):
        self._codelet_template = codelet
        self._inputs = inputs
        self._outputs = outputs
        self._op_params = op_params

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def op_params(self) -> Dict:
        return self._op_params

    @property
    def codelet_template(self):
        return self._codelet_template


    def get_json_inputs(self) -> List[Dict]:
        blobs = []
        for i in self.inputs:
            blob = {}
            blob['name'] = i.name
            blob['dimensions'] = i.shape
            blob['dtype'] = i.kwargs['hag_dtype']
            blobs.append(blob)
        return blobs

    def get_json_outputs(self) -> List[Dict]:
        blobs = []
        for o in self.outputs:
            blob = {}
            blob['name'] = o.name
            blob['dimensions'] = o.shape
            blob['dtype'] = o.kwargs['hag_dtype']
            blobs.append(blob)
        return blobs

    def compiled_json(self) -> Dict:
        blob = {}
        blob['id'] = self.codelet_template.codelet_id
        blob['codelet_name'] = self.codelet_template.name
        blob['inputs'] = self.get_json_inputs()
        blob['outputs'] = self.get_json_outputs()
        blob['parameters'] = self.op_params
        blob['capability_sequence'] = [c.compiled_json() for c in self.codelet_template.capability_sequence]
        return blob

