from typing import List, Union, Dict, Tuple

from codelets.adl.flex_param import FlexParam
from codelets.adl.operation.operand import OperandTemplate, Datatype
from .operation import Operation, Loop, Transfer, Compute, Configure
from types import LambdaType
from pytools import memoize_method
from collections import defaultdict
from dataclasses import replace
from numbers import Integral
from copy import deepcopy
from sympy import symbols, Idx, Expr, Basic
from itertools import chain
import json
import numpy as np

import polymath as pm

class Codelet(object):
    codelet_instance_id = 0
    codelet_id = 0
    operations = []

    def __init__(self, op_name,
                 inputs: List[OperandTemplate],
                 outputs: List[OperandTemplate],
                 is_instance: bool =False,
                 cdlt_id: int = None,
                 required_params: Dict[str, Union[int, str, FlexParam, LambdaType]] = None):

        self._op_name = op_name
        self._inputs = inputs
        self._outputs = outputs
        self._ops = []
        self._op_map = {}
        self._global_op_map = {}
        self._num_instr = -1

        # Added, possibly need to consolidate
        self._domain_tiling = {}
        self._tile_levels = defaultdict(list)
        if required_params is not None:
            self._required_params = {}
            for k, v in required_params.items():
                self.add_required_param(k, v)
        else:
            self._required_params = {}
        self._is_instance = is_instance
        self._instance_id = None
        if self.is_instance:
            self._instance_id = Codelet.codelet_instance_id
            Codelet.codelet_instance_id += 1

        if cdlt_id:
            self._cdlt_id = cdlt_id
        else:
            self._cdlt_id = Codelet.codelet_id
            Codelet.codelet_id += 1

    def __enter__(self):
        Operation.current_codelet = self
        Operation.loop_stack.append(-1)
        Operation.id_counter = 0
        Operation.loop_ctxt_level = 0
        Operation.op_id_counters = defaultdict(int)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Operation.current_codelet = None
        last_id = Operation.loop_stack.pop()
        assert last_id == -1

    @property
    def is_instance(self):
        return self._is_instance

    @property
    def instance_id(self):
        return self._instance_id

    @property
    def cdlt_id(self):
        return self._cdlt_id

    @property
    def op_name(self):
        return self._op_name

    @property
    def required_params(self) -> Dict[str, Union[None, FlexParam]]:
        return self._required_params

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def ops(self) -> List[Operation]:
        return self._ops

    @property
    def op_map(self) -> Dict[str, Union[Loop, Compute, Transfer, Configure]]:
        return self._op_map

    @property
    def global_op_map(self) -> Dict[int, Union[Loop, Compute, Transfer, Configure]]:
        return self._global_op_map

    # TODO: Memoize this method
    @property
    def operands(self):
        return self.inputs + self.outputs

    @ops.setter
    def ops(self, ops):
        self._ops = ops

    @property
    def num_instr(self):
        return self._num_instr

    @memoize_method
    def operand_dimensions(self) -> List[str]:
        operands = self.inputs + self.outputs
        operand_dims = []
        for o in operands:
            operand_dims += o.shape_list
        return list(set(operand_dims))

    @property
    def domain_tiling(self):
        return self._domain_tiling

    @property
    def tile_levels(self):
        return self._tile_levels

    def operand_dim_mapping(self):
        operands = self.inputs + self.outputs
        operand_dims = {}
        for o in operands:
            operand_dims.update(o.shape_symbols)
        return operand_dims

    def unset_params(self):
        unset_params = []
        for k, v in self.required_params.items():
            if v is None:
                unset_params.append(k)
            else:
                assert isinstance(v, FlexParam)
                if not v.is_set():
                    unset_params.append(k)
        return unset_params

    def get_operand(self, op_name: str):
        for o in (self.inputs + self.outputs):
            if o.name == op_name:
                return o
        raise KeyError(f"Unable to find operand {op_name}: {self.inputs + self.outputs}")

    def copy(self):

        obj = type(self).__new__(self.__class__)
        obj._op_name = self.op_name
        obj._cdlt_id = self.cdlt_id
        obj._inputs = [i.copy() for i in self.inputs]
        obj._outputs = [o.copy() for o in self.outputs]
        obj._required_params = self.copy_required_params()
        obj._ops = []
        obj._op_map = {}
        obj._global_op_map = {}
        obj._num_instr = self._num_instr
        obj._cdlt_id = self._cdlt_id
        obj._instance_id = Codelet.codelet_instance_id
        obj._domain_tiling = deepcopy(self._domain_tiling)
        obj._tile_levels = deepcopy(self._tile_levels)
        for o in self.ops:
            obj.add_op(o.copy(obj))
        return obj

    def copy_required_params(self):
        params = {}
        for k, v in self.required_params.items():
            if isinstance(v, FlexParam):
                params[k] = v.copy()
            elif v is None:
                params[k] = v
            else:
                raise TypeError(f"Invalid type when copying params:\n"
                                f"Name: {k}\n"
                                f"Param: {v}")
        return params

    def get_op(self, global_op_id: int) -> Operation:
        for o in self.ops:
            if o.global_op_id == global_op_id:
                return o
        raise KeyError(f"Unable to find global op id {global_op_id}")


    def emit(self, output_type):
        if output_type == "operations":
            op_str = f"CODELET:\t{self.op_name}{self.instance_id}\n"
            for o in self.ops:
                ostr = f"\t" * (o.loop_level + 1)
                ostr += f"{o.emit(output_type)}\n"
                op_str += ostr
        elif output_type == "json":
            op_params = {}
            operand_dim_map = self.operand_dim_mapping()
            for k, v in self.required_params.items():
                if k not in operand_dim_map:
                    assert isinstance(v, FlexParam)
                    op_params[k] = v.value

            op_str = {}
            op_str['operand'] = self.op_name
            op_str['iterable_dimensions'] = [[k, v] for k, v in operand_dim_map.items()]
            op_str['operation_parameters'] = op_params
            op_str['inputs'] = [i.emit(output_type) for i in self.inputs]
            op_str['outputs'] = [o.emit(output_type) for o in self.outputs]
            op_str['operation_sequence'] = [op.emit(output_type) for op in self.ops]
        elif output_type not in ["decimal", "binary"]:
            op_str = f"CODELET:\t{self.op_name}{self.instance_id}\n"
            for o in self.ops:
                instr_list = o.emit(output_type)
                ostr = f"\t" * (o.loop_level + 1)
                instr_list = f"\n{ostr}".join(instr_list)
                ostr += f"{instr_list}\n"
                op_str += ostr
        else:
            op_str = []
            for o in self.ops:
                instr_list = o.emit(output_type)
                op_str += instr_list
            op_str = "\n".join(op_str)
        return op_str

    def add_op(self, op: Operation):
        for rp_key in op.required_params:
            if rp_key not in self.required_params:
                self.add_required_param(rp_key)
        self.ops.append(op)
        self.op_map[op.op_str] = op
        self.global_op_map[op.global_op_id] = op

    def add_required_param(self, key, value=None, check_key=True):
        if key in self.required_params:
            if check_key:
                raise KeyError(f"Key {key} already exists in params:\n"
                               f"Previous value: {self.required_params[key]}\n"
                               f"Updated value: {value}")
            else:
                return

        if isinstance(value, LambdaType):
            flex_param = FlexParam(key, fn=value)
            for a in flex_param.fn_args:
                if a not in self.required_params:
                    self.add_required_param(a)

            self.required_params[key] = flex_param
        elif isinstance(value, int):
            flex_param = FlexParam(key)
            flex_param.value = value
            self.required_params[key] = flex_param
        elif value is None:
            self.required_params[key] = FlexParam(key)
        elif isinstance(value, FlexParam):
            self.required_params[key] = value
        else:
            raise TypeError(f"Invalid type for required param:\n"
                            f"Name: {key}\n"
                            f"Value: {value}")

    def set_required_param(self, key: str, value: int):
        value = value.value if isinstance(value, FlexParam) else value
        if key not in self.required_params:
            raise KeyError(f"Key {key} for updating param does not exist:\n"
                           f"All Keys: {self.required_params.keys()}\n"
                           f"Updated value: {value}")


        # TODO: Check back on this
        if self.required_params[key].is_set() and self.required_params[key].value != value and\
                not isinstance(self.required_params[key].value, LambdaType):
            raise RuntimeError(f"Param {key} has already been set:\n"
                               f"Previous value: {self.required_params[key]}\n"
                               f"New value: {value}")
        if value is None:
            raise RuntimeError(f"Cannot self None value for required parameter:\n"
                               f"Value: {value}\n"
                               f"Key: {key}")
        self.required_params[key].value = value

    def configure(self, start_end, target_name, **kwargs):
        cfg = Configure(start_end, target_name,
                        add_codelet=False, **kwargs)
        self.add_op(cfg)


    def compute(self, op_name, sources, dests, **kwargs):
        comp = Compute(op_name, sources, dests,
                        add_codelet=False, **kwargs)
        self.add_op(comp)

    def transfer(self, operand, path, offsets, sizes=None, **kwargs):
        xfer = Transfer(operand, path, offsets, sizes,
                        add_codelet=False, **kwargs)
        self.add_op(xfer)

    def set_domain_tile(self, hag_node: str, domain_key: str, split_factor: int):
        if hag_node not in self.domain_tiling:
            self.domain_tiling[hag_node] = {}

        if domain_key in self.domain_tiling[hag_node] and self.domain_tiling[hag_node][domain_key] != split_factor:
            raise RuntimeError(f"The tile split factor has already been set for {hag_node} in domain"
                               f" {domain_key}:\n"
                               f"Previous value: {self.domain_tiling[hag_node][domain_key]}\n"
                               f"New value: {split_factor}")
        # TODO: Add other checks here to validate split
        self.domain_tiling[hag_node][domain_key] = split_factor

    def set_tile_level(self, level: int, node: str):
        if level in self.tile_levels:
            assert node not in self.tile_levels[level]
        self.tile_levels[level].append(node)

    def set_dim_values(self, node, operand):
        if not operand.is_instantiated():
            for j, s in enumerate(node.shape):
                key = operand.shape_list[j]
                operand.update_shape_symbols(key, s)
                if key not in self.required_params:
                    self.add_required_param(key, s)
                elif key in self.required_params and not self.required_params[key].is_set():
                    self.set_required_param(key, s)

            if len(operand.shape_list) != len(list(operand.shape_symbols.keys())):
                raise RuntimeError(f"All shape values were not set for node {node.name}, operand {operand.name}:\n"
                                   f"Node shape: {node.shape}\n"
                                   f"Operand shape variables: {operand.shape_list}")

    def get_tile_level(self, node_name: str):
        for i in self.tile_levels.keys():
            if node_name in self.tile_levels[i]:
                return i
        raise KeyError(f"Unable to find tile level for node {node_name}")


    def get_node_shape_map(self, op_tmplt: OperandTemplate, node: pm.Node) -> Dict[str, Dict]:
        shape_map = {}
        for i, s in enumerate(node.shape):

            key = op_tmplt.shape_symbols[i]
            shape_map[key] = {'value': s,
                              'dimension': i}
        return shape_map

    def set_dtype(self, node, operand):
        if "hag_dtype" not in node.kwargs:
            dtype = operand.supported_dtypes[0]
            node.add_attribute("hag_dtype", str(dtype))
        else:
            assert operand.is_dtype_supported(node.kwargs['hag_dtype'])
            dtype = Datatype.from_str(node.kwargs['hag_dtype'])
        operand.set_dtype(dtype)

    def set_op_node_name(self, node, operand):
        if operand.node_name is not None and operand.node_name != node.name:
            raise RuntimeError(f"Name already set to different value for operand:\n"
                               f"Previous name: {operand.node_name}\n"
                               f"New name: {node.name}")
        operand.set_node_name(node.name)

    def instantiate_operands(self, node, hag):
        all_cdlt_ops = self.inputs + self.outputs
        all_node_ops = node.inputs + node.outputs
        for i, n in enumerate(all_node_ops):
            operand = all_cdlt_ops[i]
            self.set_dim_values(n, operand)
            self.set_dtype(n, operand)
            self.set_op_node_name(n, operand)

    def instantiate_node_params(self, node, hag):
        fn_params = []
        for key, param in self.required_params.items():
            if key in node.kwargs:
                self.set_required_param(key, node.kwargs[key])
            elif isinstance(param, FlexParam) and param.value_type == "function" and not param.is_set():
                fn_params.append(key)

        for name in fn_params:
            flex_param = self.required_params[name]
            arg_vals = tuple([self.required_params[a].value for a in flex_param.fn_args])
            self.set_required_param(name, flex_param.evaluate_fn(*arg_vals))
        for k, v in self.required_params.items():
            if isinstance(v, FlexParam) and not v.is_set():
                raise RuntimeError(f"Unable to set parameter {v.name}\n"
                                   f"Key: {k}\n"
                                   f"Kwargs: {node.kwargs.keys()}")

        for operand in (self.inputs + self.outputs):
            operand.evaluate_operand(node, self.required_params, hag)



    def instantiate_operations(self, node: pm.Node, hag):
        self.instantiate_operands(node, hag)

        self.instantiate_node_params(node, hag)
        for o in self.ops:
            for rp in o.required_params:
                if rp in self.required_params and rp in o.unset_params():
                    o.set_required_param(rp, self.required_params[rp])
            assert isinstance(o, Operation)
            o.evaluate_parameters(node, hag, self)

    def get_operand_shapes(self):
        shape_dims = {}
        operands = (self.inputs + self.outputs)
        for o in operands:
            shape_dims.update(o.shape_symbols)

        return shape_dims






