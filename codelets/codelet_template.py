
from numbers import Number
from typing import List, Dict
from pytools import memoize_method
from collections import defaultdict, deque
from copy import deepcopy

from codelets.common.flex_param import FlexParam
from codelets.micro_templates.dummy_op import DummyOp, DummyParam
from codelets.micro_templates import HAGPlaceholder
from codelets.micro_templates import NodePlaceholder
from codelets.micro_templates import OperandTemplate, IndexOperandTemplate
from codelets.micro_templates import MicroTemplate, LoopTemplate, ComputeTemplate, TransferTemplate

class CodeletTemplate(object):
    codelet_id = 0
    operations = []

    def __init__(self, op_name,
                 cdlt_id: int = None):
        self._op_name = op_name
        self._dummy_ops = {}
        self._node_placeholder = NodePlaceholder(self.op_name)
        self._hag_placeholder = HAGPlaceholder(self.op_name)
        self._inputs = []
        self._outputs = []
        self._temps = []
        self._constants = []
        self._ops = []
        self._op_map = {}
        self._operand_op_map = {}
        self._global_op_map = {}
        # Added, possibly need to consolidate
        self._id_counter = 0
        self._loop_ctxt_level = 0
        self._op_id_counters = defaultdict(int)
        self._compilation_params = {}
        self._loop_param_map = {}
        self._blocks = None

        if cdlt_id:
            self._cdlt_id = cdlt_id
        else:
            self._cdlt_id = CodeletTemplate.codelet_id
            CodeletTemplate.codelet_id += 1

    def __enter__(self):
        MicroTemplate.current_codelet = self
        MicroTemplate.loop_stack.append(-1)
        MicroTemplate.id_counter = 0
        MicroTemplate.loop_ctxt_level = 0
        MicroTemplate.op_id_counters = defaultdict(int)
        OperandTemplate.current_codelet = self
        MicroTemplate.finished_blocks = []
        MicroTemplate.current_block = []
        MicroTemplate.block = deque()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        MicroTemplate.current_codelet = None
        OperandTemplate.current_codelet = None
        last_id = MicroTemplate.loop_stack.pop()
        self._id_counter = deepcopy(MicroTemplate.id_counter)
        self._loop_ctxt_level = deepcopy(MicroTemplate.loop_ctxt_level)
        self._op_id_counters = deepcopy(MicroTemplate.op_id_counters)
        MicroTemplate.finished_blocks.append(MicroTemplate.current_block)
        self._blocks = list(reversed(MicroTemplate.finished_blocks.copy()))
        assert last_id == -1

    @property
    def blocks(self):
        return self._blocks

    def set_inputs(self, inputs: List[OperandTemplate]):
        if len(self.inputs) > 0:
            raise RuntimeError(f"Cannot overwrite existing inputs")
        self._inputs = inputs

    def set_outputs(self, outputs: List[OperandTemplate]):
        if len(self.outputs) > 0:
            raise RuntimeError(f"Cannot overwrite existing inputs")
        self._outputs = outputs

    def add_input(self, name, shape_list, dtype, location, **kwargs):
        assert all([isinstance(s, (int, DummyOp)) for s in shape_list])
        operand = OperandTemplate(name, location, "input",
                                  src_op=self.cdlt_uid,
                                  shape_list=shape_list,
                                  default_dtype=dtype)
        self.inputs.append(operand)
        return operand

    def add_output(self, name, shape_list, dtype, location, **kwargs):
        assert all([isinstance(s, (int, DummyOp)) for s in shape_list])
        operand = OperandTemplate(name, location, "output",
                                  shape_list=shape_list,
                                  default_dtype=dtype)
        self.outputs.append(operand)
        return operand

    def add_temp_operand(self, operand: OperandTemplate):
        self._temps.append(operand)

    def get_source_location(self, operand: OperandTemplate):
        source_op = self.op_map[operand.src_op]
        assert "transfer" in source_op.op_str and operand == source_op.dst_op
        return source_op.src_op.location

    @property
    def dummy_ops(self):
        return self._dummy_ops

    @property
    def compilation_params(self):
        return self._compilation_params

    @property
    def cdlt_id(self):
        return self._cdlt_id

    @property
    def op_name(self):
        return self._op_name

    @property
    def inputs(self) -> List[OperandTemplate]:
        return self._inputs

    @property
    def temps(self) -> List[OperandTemplate]:
        return self._temps

    @property
    def constants(self) -> List[OperandTemplate]:
        return self._constants


    @property
    def outputs(self) -> List[OperandTemplate]:
        return self._outputs

    @property
    def ops(self) -> List[MicroTemplate]:
        return self._ops

    @property
    def op_map(self) -> Dict[str, MicroTemplate]:
        return self._op_map

    @property
    def global_op_map(self) -> Dict[int, MicroTemplate]:
        return self._global_op_map

    @property
    def loop_param_map(self):
        return self._loop_param_map

    # TODO: Memoize this method
    @property
    def operands(self):
        return self.inputs + self.outputs

    @ops.setter
    def ops(self, ops):
        self._ops = ops

    @property
    def op_id_counters(self):
        return self._op_id_counters

    @property
    def id_counter(self):
        return self._id_counter

    @id_counter.setter
    def id_counter(self, id_counter):
        self._id_counter = id_counter

    @property
    def node_placeholder(self):
        return self._node_placeholder

    @property
    def hag_placeholder(self):
        return self._hag_placeholder

    @property
    def hag(self):
        return self._hag_placeholder

    @property
    def node(self):
        return self._node_placeholder

    @property
    def cdlt_uid(self):
        return f"{self.op_name}{self.cdlt_id}"

    def get_output_operand(self, op_name: str):
        for t in self.temps:
            if t.src_op == op_name:
                return t
        raise RuntimeError(f"Unable to find output for operation {op_name}")

    def get_operation_output(self, op: MicroTemplate):
        output_operand = self.get_output_operand(op.op_str)
        for t in self.ops:
            if t.op_type == "transfer" and t.src_op == output_operand:
                return t.dst_op.location
        raise RuntimeError(f"Unable to find output location for operation {op.op_str}")


    def __repr__(self):
        return f"codelet_template {self.op_name}{self.codelet_id}"

    def dummy_op(self, key: str, op: DummyOp, check_key=True):
        if key in self.dummy_ops:
            if check_key:
                raise KeyError(f"Key {key} already exists in dummy ops:\n"
                               f"Previous op: {self.dummy_ops[key]}\n"
                               f"Updated op: {op}")
            else:
                return self.dummy_ops[key]
        assert isinstance(op, DummyOp)
        op.flex_param.name = key
        self._dummy_ops[key] = op
        return op

    def update_dummy_op(self, key: str, op: DummyOp):
        if key not in self.dummy_ops:
            raise KeyError(f"Cannot update dummy op for {key} because it is not in current ops:\n"
                           f"Possible keys: {list(self.dummy_ops.keys())}")
        assert isinstance(op, DummyOp)
        prev_op = self._dummy_ops[key]
        prev_op.update(op)
        return prev_op

    def update_dummy_param(self, key: str, op: DummyOp):
        if key not in self.dummy_ops:
            raise KeyError(f"Cannot update dummy param for {key} because it is not in current ops:\n"
                           f"Possible keys: {list(self.dummy_ops.keys())}")
        assert isinstance(op, DummyParam)
        self._dummy_ops[key] = op
        return op

    def dummy_param(self, key: str, fn_str: str, fn_args: List[str], dummy_args):
        assert len(fn_args) == len(dummy_args)
        fp = FlexParam(key, fn_args=fn_args, fn_body_str=fn_str)
        dparam = DummyParam(fp, dummy_args)
        self._dummy_ops[key] = dparam
        return dparam

    def check_dummy_args(self, args):
        for a in args:
            if isinstance(a, (tuple, list)):
                self.check_dummy_args(a)
            elif not isinstance(a, DummyOp):
                raise RuntimeError

    @memoize_method
    def operand_dimensions(self) -> List[str]:
        operands = self.inputs + self.outputs
        operand_dims = []
        for o in operands:
            operand_dims += o.shape_list
        return list(set(operand_dims))

    def add_op(self, op: MicroTemplate):
        # TODO: iterate over dummy ops in param map and add to codelet
        self.ops.append(op)
        self.op_map[op.op_str] = op
        self.global_op_map[op.global_op_id] = op

    def loop(self, start, **kwargs):
        assert isinstance(start, (int, DummyOp))
        loop_op_template = LoopTemplate(start, add_codelet=False, **kwargs)
        self.add_op(loop_op_template)
        return loop_op_template

    def compute(self, op_name, source_ops, compute_target, **kwargs):
        compute_template = ComputeTemplate(op_name, source_ops, compute_target,
                                           add_codelet=False, **kwargs)
        self.add_op(compute_template)

        compute_output = OperandTemplate(f"temp{len(self.temps)}",
                                             compute_target,
                                         "intermediate")
        compute_output.set_source_op(compute_template.op_str)
        compute_template.set_output_operand(compute_output)

        self.add_temp_operand(compute_output)

        return compute_output

    def constant(self, value, location, dtype):
        assert isinstance(value, (Number, DummyOp))
        operand = OperandTemplate(f"constant{len(self.constants)}", location, "constant",
                                  src_op=self.cdlt_uid,
                                  shape_list=[0],
                                  default_dtype=dtype)

        if isinstance(value, Number):
            operand.init_value = value
        else:
            operand.init_dummy_op = value

        self.constants.append(operand)
        return operand


    def transfer(self, source, dest, dest_offset=None, size=None, **kwargs):
        if size is None:
            if isinstance(source, OperandTemplate):
                size = source.shape_list
            else:
                assert isinstance(source, IndexOperandTemplate)
                size = source.operand.shape_list

        if isinstance(dest, str):

            transfer_output = OperandTemplate(f"temp{len(self.temps)}", dest, "intermediate")

            if dest_offset is not None:
                assert isinstance(dest_offset, list)
                self.add_temp_operand(transfer_output)
                transfer_output = transfer_output[tuple(dest_offset)]
            else:
                self.add_temp_operand(transfer_output)
        else:
            assert isinstance(dest, (IndexOperandTemplate, OperandTemplate))
            transfer_output = dest

        transfer_template = TransferTemplate(source, transfer_output, size,
                                             add_codelet=False, **kwargs)

        self.add_op(transfer_template)
        return transfer_output


    def instantiate(self, instance_args):
        pass
        # TODO: Fix this
        # inputs = [i.instantiate(instance_args) for i in self.inputs]
        # outputs = [o.instantiate(instance_args) for o in self.outputs]
        # temps = [t.instantiate(instance_args) for t in self.temps]
        # contexts = deque()
        # with Codelet(self.op_name, inputs, outputs, instance_args['HAGPlaceholder']) as cdlt:
        #     cdlt._temps = temps
        #     instance_args['CodeletTemplate'] = cdlt
        #     for o in self.ops:
        #
        #         while len(contexts) > 0 and o.loop_level <= contexts[-1].loop_level:
        #             cm = contexts.pop()
        #             cm.exit_loop_body()
        #         if isinstance(o, LoopTemplate):
        #
        #             new_op = o.instantiate(instance_args).enter_loop_body()
        #             contexts.append(new_op)
        #         else:
        #             new_op = o.instantiate(instance_args)
        #         cdlt.add_op(new_op)
        #
        #     while len(contexts) > 0:
        #         cm = contexts.pop()
        #         cm.exit_loop_body()
        #
        # for k, v in self.compilation_params.items():
        #     cdlt.add_compilation_param(k, v)
        #
        # for key, do in self.dummy_ops.items():
        #     if not do.flex_param.is_set():
        #         do.evaluate(instance_args)
        #     if key not in cdlt.required_params:
        #         cdlt.add_required_param(key, do.value)
        #
        #     elif cdlt.required_params[key].is_set() and do.value is not None:
        #         if cdlt.required_params[key].value != do.value:
        #                 raise RuntimeError(f"Inconsistent values for dummy op:\n"
        #                                    f"{key}")
        #     elif not cdlt.required_params[key].is_set():
        #         cdlt.required_params[key].value = do.value
        # Codelet.codelet_instance_id += 1
        # cdlt._instance_id = Codelet.codelet_instance_id
        # return cdlt

    def emit(self, output_type):
        if output_type not in ["operations", "operations_idx"]:
            raise RuntimeError(f"Invalid output type for uninstantiated CodeletTemplate")

        if output_type == "operations_idx":
            input_str = ", ".join([f"{i.name}" for i in self.inputs])
            out_str = ", ".join([f"{o.name}" for o in self.outputs])
            operand_str = f"inputs={input_str}\n" \
                          f"outputs={out_str}\n"
            op_str = f"CODELET:\t{self.op_name}\n"
            op_str += operand_str
            for i, o in enumerate(self.ops):
                ostr = f"{i}" + f"\t" * (o.loop_level + 1)
                ostr += f"{str(o)}\n"
                op_str += ostr
        else:
            input_str = ", ".join([f"{i.name}" for i in self.inputs])
            out_str = ", ".join([f"{o.name}" for o in self.outputs])
            operand_str = f"inputs={input_str}\n" \
                          f"outputs={out_str}\n"
            op_str = f"CODELET:\t{self.op_name}\n"
            op_str += operand_str
            for o in self.ops:
                ostr = f"\t" * (o.loop_level + 1)
                ostr += f"{o.op_str}\n"
                op_str += ostr
        return op_str

    def add_compilation_param(self, key, value):
        self._compilation_params[key] = value