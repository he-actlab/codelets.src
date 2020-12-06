import numpy as np
from collections import namedtuple
from typing import Callable, Any, List, Dict, Optional, Tuple, Set

Operand = namedtuple('Operand', ['component_name', 'datatype', 'dimensions'])
# TODO does different on-chip dataflow need to be considered?
#      if so, how should it be included?
#      current implementation considers a single type of on-chip dataflow!

class Codelet(object):
    """
    Base class for capability
    """
    CONST_VAL = 'CONST'

    def __init__(self, name,
                 input_dimension_names=None,
                 output_dimension_names=None,
                 input_dtypes=None,
                 output_dtypes=None,
                 input_components=None,
                 output_components=None,
                 subcapabilities=None,
                 op_params=None,
                 latency=None,
                 is_atomic=True):
        self._name = name
        self._is_atomic = is_atomic

        # list of input_components should be identical regardless of dataflows
        # 'input name': 'dims'
        self.input_dtypes = input_dtypes
        self.output_dtypes = output_dtypes
        self.input_components = input_components or {}
        self.output_components = output_components or {}
        self.subcapabilities = subcapabilities or []
        self.input_dimension_names = input_dimension_names or []
        self.output_dimension_names = output_dimension_names or []
        self.op_params = op_params or {}
        # latency can be either a fixed value or a lambda function
        self.latency = latency or 0

    @property
    def is_atomic(self) -> bool:
        return self._is_atomic

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_components(self) -> List[str]:
        return self._output_components

    @property
    def input_dtypes(self) -> List[str]:
        return self._input_dtypes

    @property
    def output_dtypes(self) -> List[str]:
        return self._output_dtypes

    @property
    def input_components(self) -> List[str]:
        return self._input_components

    @property
    def input_dimension_names(self) -> List[str]:
        return self._input_dimension_names

    @property
    def output_dimension_names(self) -> List[str]:
        return self._output_dimension_names

    @property
    def latency(self) -> int:
        return self._latency



    @property
    def op_params(self) -> Dict[str, Any]:
        return self._op_params

    @property
    def subcapabilities(self) -> List['Codelet']:
        return self._subcapabilities

    @property
    def all_dim_names(self) -> List[str]:
        dims = []
        for a in self.input_dimension_names:
            dims += a
        dims += self.output_dimension_names
        dims = list(set(dims))
        return dims

    @property
    def loop_dim_names(self) -> List[str]:
        dims = []
        for a in self.input_dimension_names:
            dims += a
        dims = list(set(dims))
        return dims

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


    @output_components.setter
    def output_components(self, output_components):
        self._output_components = output_components

    @input_components.setter
    def input_components(self, input_components):
        self._input_components = input_components

    @input_dimension_names.setter
    def input_dimension_names(self, input_dimension_names):
        self._input_dimension_names = input_dimension_names

    @output_dimension_names.setter
    def output_dimension_names(self, output_dimension_names):
        self._output_dimension_names = output_dimension_names

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @op_params.setter
    def op_params(self, op_params):
        self._op_params = op_params


    @subcapabilities.setter
    def subcapabilities(self, subcapabilities):
        self._subcapabilities = subcapabilities


    def set_name(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def get_input_sources(self):
        sources = []
        for k, v in self._input_components.items():
            for s in v['src']:
                if s not in sources and s != Codelet.CONST_VAL:
                    sources.append(s)
        return sources

    def get_output_dests(self):
        dests = []
        for k, v in self._output_components.items():
            for d in v['dst']:
                if d not in dests:
                    dests.append(d)
        return dests

    def add_input(self, name, src, dims):
        self._input_components[name] = {'src': src, 'dims': dims}
    
    def get_required_input_dims(self):
        # this one liner identifies unique dimensions that are required by the capability
        return set(sum([list(self._input_components[name]['dims'].keys()) for name in self._input_components.keys()], []))

    def get_required_input_ranges(self, dim):
        dim_ranges = [self._input_components[name]['dims'][dim] for name in self._input_components.keys()]
        equal = len(set(dim_ranges)) <= 1
        assert equal, 'ranges should be identically defined for same dim'
        return set(dim_ranges)[0]


    def add_output(self, name, dst, dims):
        self._output_components[name] = {'dst': dst, 'dims': dims}

    def get_required_output_dims(self):
        # this one liner identifies unique dimensions that are required by the capability
        return set(sum([list(self._output_components[name]['dims'].keys()) for name in self._output_components.keys()], []))

    def get_required_output_ranges(self, dim):
        dim_ranges = [self._output_components[name]['dims'][dim] for name in self._output_components.keys()]
        equal = len(set(dim_ranges)) <= 1
        assert equal, 'ranges should be identically defined for same dim'
        return set(dim_ranges)[0]
    

    def set_latency(self, latency):
        self._latency = latency

    def get_latency(self):
        return self._latency

    def get_output(self, name):
        assert name in self._output_components
        return self._output_components[name]

    def get_input(self, name):
        assert name in self._input_components
        return self._input_components[name]

    def add_sub_capability(self, capability):
        if len(self.subcapabilities) == 0:
            self._is_atomic = False
        self.subcapabilities.append(capability)

    def get_op_param(self, key):
        if key not in self._op_params:
            raise KeyError(f"Key {key} not in op params: {self.op_params.keys()}")
        return self.op_params[key]

    def add_op_param(self, key, value):
        if key in self._op_params:
            raise KeyError(f"Key {key} already in op params: {self.op_params.keys()}")
        self.op_params[key] = value


class Capability(object):
    op_num = 0
    def __init__(self, capability,
                 component_target,
                 inputs,
                 outputs,
                 loop_order,
                 executions,
                 tiling=None,
                 op_num=-1, **kwargs):
        if op_num:
            self.instance_name = f"{capability.name}{Capability.op_num}"
            Capability.op_num = op_num + 1
        else:
            Capability.op_num += 1
            self.instance_name = f"{capability.name}{Capability.op_num}"
        self.executions = executions
        self.tiling = tiling
        self.loop_order = loop_order
        self.outputs = outputs
        self.inputs = inputs
        self.dimension_values = {}
        self.component_target = component_target or None
        self.capability = capability
        self.op_params = kwargs

        for i in range(len(self.inputs)):
            dim_names = self.capability.input_dimension_names[i]
            for dim, dim_val in enumerate(self.inputs[i].dimensions):
                name = dim_names[dim]
                if name not in self.dimension_values:
                    self.dimension_values[name] = dim_val
                elif self.dimension_values[name] != dim_val:
                    raise RuntimeError(f"Dimension {name} is not equal to capability input:\n"
                                       f"Dim val: {dim_val}\tStored dim: {self.dimension_values[name]}")

        for o in range(len(self.outputs)):
            dim_names = self.capability.output_dimension_names
            for dim, dim_val in enumerate(self.outputs[o].dimensions):
                name = dim_names[dim]
                if name not in self.dimension_values:
                    self.dimension_values[name] = dim_val
                else:
                    assert self.dimension_values[name] == dim_val


        if not self.component_target:
            raise RuntimeError(f"Operation {capability.name} could not be instantiated because 'component_target' is not set!")
        elif len(self.capability.input_dimension_names) == 0:
            raise RuntimeError(f"Operation {capability.name} could not be instantiated because 'input_dimension_names' is not set!")
        elif len(self.capability.output_dimension_names) == 0:
            raise RuntimeError(f"Operation {capability.name} could not be instantiated because 'output_dimension_names' is not set!")
        elif set(self.loop_order) != set(self.capability.loop_dim_names):
            raise RuntimeError(f"Operation {capability.name} could not be instantiated because all dimensions are not included in loop order!\n"
                               f"All dimensions: {self.capability.loop_dim_names}\n"
                               f"Loop order: {self.loop_order}")
        elif self.tiling and set(list(self.tiling.keys())) != set(self.capability.loop_dim_names):
            raise RuntimeError(f"Operation {capability.name} could not be instantiated because all dimensions are not included in tiling!\n"
                               f"All dimensions: {self.capability.loop_dim_names}\n"
                               f"Tiling keys: {list(self.tiling.keys())}")

    @property
    def executions(self):
        return self._executions

    @property
    def component_target(self) -> str:
        return self._component_target

    @property
    def dimension_values(self) -> Dict[str, int]:
        return self._dimension_values

    @property
    def inputs(self):
        return self._inputs

    @property
    def input_dims(self):
        return (self.dimension_values[i] for i in self.capability.loop_dim_names)

    @property
    def outputs(self):
        return self._outputs

    @property
    def input_dtypes(self):
        return [i.datatype for i in self.inputs]

    @property
    def output_dtypes(self):
        return [o.datatype for o in self.outputs]

    @property
    def input_locations(self):
        return [i.component_name for i in self.inputs]

    @property
    def output_locations(self):
        return [o.component_name for o in self.outputs]

    @property
    def loop_order(self) -> List[str]:
        return self._loop_order

    @property
    def instance_name(self):
        return self._instance_name

    @property
    def tiling(self) -> Dict[str, int]:
        return self._tiling

    @executions.setter
    def executions(self, executions):
        self._executions = executions

    @dimension_values.setter
    def dimension_values(self, dimension_values):
        self._dimension_values = dimension_values

    @component_target.setter
    def component_target(self, component_target):
        self._component_target = component_target

    @outputs.setter
    def outputs(self, outputs):
        self._outputs = outputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs

    @loop_order.setter
    def loop_order(self, loop_order):
        self._loop_order = loop_order

    @instance_name.setter
    def instance_name(self, instance_name):
        self._instance_name = instance_name

    @tiling.setter
    def tiling(self, tiling):
        self._tiling = tiling

    def __repr__(self):
        return str(self.json_op())

    def json_op(self):
        blob = {'id': self.instance_name,
                'capability': self.capability.name,
                'executions': self.executions,
                'input_dimensions': self.capability.input_dimension_names,
                'output_dimensions': self.capability.output_dimension_names,
                'loop_order': self.loop_order,
                'dim_values': self.dimension_values,
                'tiling': self.tiling,
                'input_dtypes': self.input_dtypes,
                'output_dtypes': self.output_dtypes,
                'input_locations': self.input_locations,
                'output_locations': self.output_locations,
                'op_params': self.op_params
                }
        return blob

    def __str__(self):
        return str(self.json_op())
