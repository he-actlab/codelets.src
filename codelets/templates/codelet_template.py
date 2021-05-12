from typing import List, Union, Dict, Tuple, Any, Callable
from types import LambdaType
from pytools import memoize_method
from collections import defaultdict, deque
from copy import deepcopy

from codelets.common import Datatype, get_full_obj_type
from codelets.adl.flex_param import FlexParam
from . import CLASS_TEMPLATE_MAP
from .dummy_op import DummyOp, DummyParam
from .hag_placeholder import HAGPlaceholder
from .node_placeholder import NodePlaceholder
from .operand_template import OperandTemplate
from .operation_template import OperationTemplate, LoopTemplate, ComputeTemplate, ConfigureTemplate, TransferTemplate
from codelets.adl.operation import Operation
from codelets.codelet_impl import Codelet

class CodeletTemplate(object):
    codelet_id = 0
    operations = []

    def __init__(self, op_name,
                 is_instance: bool = False,
                 cdlt_id: int = None,
                 required_params: Dict[str, Union[int, str, FlexParam, LambdaType]] = None):
        self._op_name = op_name
        self._dummy_ops = {}
        self._node_placeholder = NodePlaceholder(self.op_name)
        self._hag_placeholder = HAGPlaceholder(self.op_name)
        self._inputs = []
        self._outputs = []
        self._temps = []
        self._ops = []
        self._op_map = {}
        self._global_op_map = {}
        # Added, possibly need to consolidate
        self._id_counter = 0
        self._loop_ctxt_level = 0
        self._op_id_counters = defaultdict(int)
        self._compilation_params = {}
        self._loop_param_map = {}

        if required_params is not None:
            self._required_params = {}
            for k, v in required_params.items():
                self.add_required_param(k, v)
        else:
            self._required_params = {}
        self._is_instance = is_instance

        if cdlt_id:
            self._cdlt_id = cdlt_id
        else:
            self._cdlt_id = CodeletTemplate.codelet_id
            CodeletTemplate.codelet_id += 1

    def __enter__(self):
        OperationTemplate.current_codelet = self
        OperationTemplate.loop_stack.append(-1)
        OperationTemplate.id_counter = 0
        OperationTemplate.loop_ctxt_level = 0
        OperationTemplate.op_id_counters = defaultdict(int)
        OperandTemplate.current_codelet = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        OperationTemplate.current_codelet = None
        OperandTemplate.current_codelet = None
        last_id = OperationTemplate.loop_stack.pop()
        self._id_counter = deepcopy(OperationTemplate.id_counter)
        self._loop_ctxt_level = deepcopy(OperationTemplate.loop_ctxt_level)
        self._op_id_counters = deepcopy(OperationTemplate.op_id_counters)
        assert last_id == -1

    def set_inputs(self, inputs: List[OperandTemplate]):
        if len(self.inputs) > 0:
            raise RuntimeError(f"Cannot overwrite existing inputs")
        self._inputs = inputs

    def set_outputs(self, outputs: List[OperandTemplate]):
        if len(self.outputs) > 0:
            raise RuntimeError(f"Cannot overwrite existing inputs")
        self._outputs = outputs

    def create_operand_template(self, name: str, dtypes: List[Datatype], shape_list: List,
                                **kwargs):
        op_temp = OperandTemplate(name, dtypes, shape_list, **kwargs)
        return op_temp

    def create_temp_operand(self, shape_list, location):
        name = f"temp{len(self.temps)}"
        # TODO: Infer supported dtypes somehow
        supported_dtypes = self.inputs[0].supported_dtypes
        # TODO: Fix check for node existance
        temp_op = OperandTemplate(name, supported_dtypes, shape_list,
                                  start_location=location)
        self._temps.append(temp_op)
        return temp_op

    def add_temp_operand(self, operand: OperandTemplate):
        self._temps.append(operand)

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
    @property
    def dummy_ops(self):
        return self._dummy_ops

    @property
    def is_instance(self):
        return self._is_instance

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
    def required_params(self) -> Dict[str, Union[None, FlexParam]]:
        return self._required_params

    @property
    def inputs(self) -> List[OperandTemplate]:
        return self._inputs

    @property
    def temps(self) -> List[OperandTemplate]:
        return self._temps

    @property
    def outputs(self) -> List[OperandTemplate]:
        return self._outputs

    @property
    def ops(self) -> List[OperationTemplate]:
        return self._ops

    @property
    def op_map(self) -> Dict[str, OperationTemplate]:
        return self._op_map

    @property
    def global_op_map(self) -> Dict[int, OperationTemplate]:
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

    def add_op(self, op: OperationTemplate):
        # TODO: iterate over dummy ops in param map and add to codelet
        self.ops.append(op)
        self.op_map[op.op_str] = op
        self.global_op_map[op.global_op_id] = op

    def configure(self, start_end: str, target: str, **kwargs):
        cfg_op_template = ConfigureTemplate(start_end, target, add_codelet=False, **kwargs)
        self.add_op(cfg_op_template)
        return cfg_op_template

    def loop(self, start, **kwargs):
        loop_op_template = LoopTemplate(start, add_codelet=False, **kwargs)
        self.add_op(loop_op_template)
        return loop_op_template

    def compute(self, op_name, sources, dests, target, **kwargs):
        compute_op_template = ComputeTemplate(op_name, sources, dests, target, add_codelet=False, **kwargs)
        self.add_op(compute_op_template)
        return compute_op_template

    def transfer(self, operand, path, **kwargs):
        transfer_op_template = TransferTemplate(operand, path, add_codelet=False, **kwargs)
        self.add_op(transfer_op_template)
        return transfer_op_template

    def instantiate(self, instance_args):
        inputs = [i.instantiate(instance_args) for i in self.inputs]
        outputs = [o.instantiate(instance_args) for o in self.outputs]
        temps = [t.instantiate(instance_args) for t in self.temps]
        contexts = deque()
        with Codelet(self.op_name, inputs, outputs, instance_args['HAGPlaceholder']) as cdlt:
            cdlt._temps = temps
            instance_args['CodeletTemplate'] = cdlt
            for o in self.ops:

                while len(contexts) > 0 and o.loop_level <= contexts[-1].loop_level:
                    cm = contexts.pop()
                    cm.exit_loop_body()
                if isinstance(o, LoopTemplate):

                    new_op = o.instantiate(instance_args).enter_loop_body()
                    contexts.append(new_op)
                else:
                    new_op = o.instantiate(instance_args)
                cdlt.add_op(new_op)

            while len(contexts) > 0:
                cm = contexts.pop()
                cm.exit_loop_body()

        for k, v in self.compilation_params.items():
            cdlt.add_compilation_param(k, v)

        for key, do in self.dummy_ops.items():
            if not do.flex_param.is_set():
                do.evaluate(instance_args)
            if key not in cdlt.required_params:
                cdlt.add_required_param(key, do.value)

            elif cdlt.required_params[key].is_set() and do.value is not None:
                if cdlt.required_params[key].value != do.value:
                        raise RuntimeError(f"Inconsistent values for dummy op:\n"
                                           f"{key}")
            elif not cdlt.required_params[key].is_set():
                cdlt.required_params[key].value = do.value
        Codelet.codelet_instance_id += 1
        cdlt._instance_id = Codelet.codelet_instance_id
        return cdlt

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
                ostr += f"{o.op_str}\n"
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
