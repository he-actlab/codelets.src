from typing import List, Union, Dict
from .operand import OperandTemplate
from .operation import Operation, Loop, Transfer, Compute, Configure
from dataclasses import replace
import polymath as pm

class Codelet(object):
    codelet_id = 0
    operations = []

    def __init__(self, op_name, inputs: List[OperandTemplate], outputs: List[OperandTemplate],
                 cdlt_id=None,
                 required_params=None,
                 **kwargs):

        self._params = kwargs
        self._op_name = op_name
        self._inputs = inputs
        self._outputs = outputs
        self._ops = []
        self._op_map = {}
        self._global_op_map = {}
        self._required_params = required_params or {}
        if cdlt_id:
            self._cdlt_id = cdlt_id
        else:
            self._cdlt_id = Codelet.codelet_id
            Codelet.codelet_id += 1

    def __enter__(self):
        Operation.current_codelet = self
        Operation.loop_stack.append(-1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Operation.current_codelet = None
        last_id = Operation.loop_stack.pop()
        assert last_id == -1

    @property
    def cdlt_id(self):
        return self._cdlt_id

    @property
    def op_name(self):
        return self._op_name

    @property
    def params(self) -> Dict[str, Union[str, int, List]]:
        return self._params

    @property
    def required_params(self) -> Dict[str, Union[str, int, List]]:
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

    @ops.setter
    def ops(self, ops):
        self._ops = ops

    @property
    def num_instr(self):
        num = 0
        for o in self.ops:
            num += len(o.instructions)
        return num

    def codelet_copy(self):
        inputs = [replace(i) for i in self.inputs]
        outputs = [replace(o) for o in self.outputs]

        cdlt = Codelet(self.op_name, inputs, outputs,
                       cdlt_id=self.cdlt_id,
                       required_params=self.required_params,
                       **self.params)
        for o in self.ops:
            cdlt.add_op(o.copy_op(cdlt, o))

        return cdlt

    def get_transfer_op(self, op_id: int) -> Transfer:
        return self.op_map[f'transfer{op_id}']

    def get_configure_op(self, op_id: int) -> Configure:
        return self.op_map[f'config{op_id}']

    def get_compute_op(self, op_id: int) -> Compute:
        return self.op_map[f'compute{op_id}']

    def get_loop_op(self, op_id: int) -> Loop:
        return self.op_map[f'loop{op_id}']

    def get_operation_instructions(self, op, hag):
        pass

    def emit_operations(self):
        op_str = f"CODELET:\t{self.op_name}\n"
        for o in self.ops:
            ostr = f"\t"*(o.loop_level + 1)
            ostr += f"{str(o)}\n"
            op_str += ostr

        return op_str

    def emit_str_instructions(self, hag):
        op_str = f"CODELET:\t{self.op_name}\n"
        for o in self.ops:
            instr_list = self.get_operation_instructions(o, hag)
            # ostr = f"\t" * (o.loop_level + 1)
            # ostr += f"{str(o)}\n"
            # op_str += ostr

        return op_str

    def emit_bin_instructions(self, arch):
        pass

    def add_op(self, op: Operation):
        for rp_key in op.required_params:
            if rp_key not in self.required_params:
                self.add_required_param(rp_key)
        self.ops.append(op)
        self.op_map[op.op_str] = op
        self.global_op_map[op.global_op_id] = op

    def add_required_param(self, key, value=None):
        if key in self.required_params:
            raise KeyError(f"Key {key} already exists in params:\n"
                           f"Previous value: {self.required_params[key]}\n"
                           f"Updated value: {value}")

        self.required_params[key] = value

    def set_required_param(self, key, value):
        if key not in self.required_params:
            raise KeyError(f"Key {key} for updating param does not exist:\n"
                           f"All Keys: {self.required_params.keys()}\n"
                           f"Updated value: {value}")
        if self.required_params[key] is not None:
            raise RuntimeError(f"Param {key} has already been set:\n"
                               f"Previous value: {self.required_params[key]}\n"
                               f"New value: {value}")

        self.required_params[key] = value

    def add_param(self, key, value):
        if key in self.params:
            raise KeyError(f"Key {key} already exists in params:\n"
                           f"Previous value: {self.params[key]}\n"
                           f"Updated value: {value}")

        self.params[key] = value

    def update_param(self, key, value):
        if key not in self.params:
            raise KeyError(f"Key {key} for updating param does not exist:\n"
                           f"All Keys: {self.params.keys()}\n"
                           f"Updated value: {value}")

        self.params[key] = value

    def configure(self, start_end, target_name, **kwargs):
        cfg = Configure(start_end, target_name,
                        add_codelet=False, **kwargs)
        self.add_op(cfg)


    def compute(self, op_name, sources, dests, **kwargs):
        comp = Compute(op_name, sources, dests,
                        add_codelet=False, **kwargs)
        self.add_op(comp)

    def transfer(self, operand, path, offsets, sizes, **kwargs):
        xfer = Transfer(operand, path, offsets, sizes,
                        add_codelet=False, **kwargs)
        self.add_op(xfer)

    def get_previous_transfer_size(self, op: Transfer):
        for o in range(op.op_id, -1, -1):
            if self.get_transfer_op(o).dest == op.source:
                pass

    def get_dim_values(self, node: pm.Node):
        all_cdlt_ops = self.inputs + self.outputs
        all_node_ops = node.inputs + node.outputs
        tiling_dims = {}
        for i, opt in enumerate(all_node_ops):
            tiling_dims.update(self.get_node_shape_map(all_cdlt_ops[i], opt))

        return tiling_dims

    def get_node_shape_map(self, op_tmplt: OperandTemplate, node: pm.Node) -> Dict[str, Dict]:
        shape_map = {}
        for i, s in enumerate(node.shape):

            key = op_tmplt.shape_symbols[i]
            shape_map[key] = {'value': s,
                              'dimension': i}
        return shape_map

    def instantiate_operands(self, node, dimensions):
        pass



