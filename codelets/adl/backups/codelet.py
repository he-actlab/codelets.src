from collections import namedtuple, defaultdict
from codelets.adl.util import get_lambda_source
from typing import Callable, Any, List, Dict, Union
from codelets.adl.backups.operand import Datatype
from dataclasses import dataclass
# OperandTemplate = namedtuple('OperandTemplate', ['field_name', 'supported_dtypes', 'mem_path', 'shape'])
Field = namedtuple('Field', ['field_name', 'value', 'bitwidth'])

# Operand = namedtuple('Operand', ['component_name', 'datatype', 'dimensions'])
# TODO does different on-chip dataflow need to be considered?
#      if so, how should it be included?
#      current implementation considers a single type of on-chip dataflow!

@dataclass(frozen=True)
class OperandTemplate:
    name: str
    dtypes: Union[List[Datatype], Datatype]
    memory_path: List[str]
    shape_symbols: Union[List[str], Dict[str, Dict]]
    iteration_domain: Union[List[str], Dict[str, Dict]]

    def is_dtype_supported(self, dtype_name) -> bool:
        return dtype_name in [str(dt) for dt in self.dtypes]

    def to_json(self):
        blob = {}
        blob['field_name'] = self.name
        if isinstance(self.dtypes, list):
            blob['supported_dtypes'] = [dt.to_json() for dt in self.dtypes]
        else:
            blob['supported_dtypes'] = self.dtypes.to_json()
        blob['memory_path'] = self.memory_path
        blob['shape_symbols'] = self.shape_symbols
        blob['iteration_domain'] = self.iteration_domain
        return blob

    def is_instantiated(self):
        return isinstance(self.shape_symbols, dict)

    @staticmethod
    def from_json(ot_obj: Dict):
        ot = ot_obj.copy()
        name = ot['field_name']
        dtypes = [Datatype.from_json(dt) for dt in ot['supported_dtypes']]
        memory_path = ot['memory_path']
        shape_symbols = ot['shape_symbols']
        iter_domain = ot['iteration_domain']
        return OperandTemplate(name=name,
                               dtypes=dtypes,
                               memory_path=memory_path,
                               shape_symbols=shape_symbols,
                               iteration_domain=iter_domain)


# @dataclass(frozen=True)
# class CodeletOperand:
#     field_name: str
#     supported_dtypes: List[Datatype]
#     memory_path: List[str]
#     shape_symbols: List[str]
#
#     def is_dtype_supported(self, dtype_name) -> bool:
#         return dtype_name in [str(dt) for dt in self.supported_dtypes]
#
#     def __dict__(self):
#         blob = {}
#         blob['field_name'] = self.field_name
#         blob['supported_dtypes'] = [dict(dt) for dt in self.supported_dtypes]
#         blob['memory_path'] = self.memory_path
#         blob['shape_symbols'] = self.shape_symbols
#         return blob


class Codelet(object):
    """
    Base class for primitive
    """
    CONST_VAL = 'CONST'
    def __init__(self, name,
                 loop_order=None,
                 inputs=None,
                 outputs=None,
                 capability_sequence=None,
                 op_params=None,
                 latency=None):


        self._name = name
        if inputs:
            assert all([isinstance(i, OperandTemplate) for i in inputs])
        else:
            inputs = []
        self._inputs = inputs

        if outputs:
            assert all([isinstance(o, OperandTemplate) for o in outputs])
        else:
            outputs = []
        self._outputs = outputs
        self._loop_order = loop_order or []
        self._capability_sequence = capability_sequence or []
        self.op_params = op_params or {}
        # latency can be either a fixed value or a lambda function
        self.latency = latency or 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def inputs(self) -> List[OperandTemplate]:
        return self._inputs

    @property
    def outputs(self) -> List[OperandTemplate]:
        return self._outputs

    @property
    def latency(self) -> int:
        return self._latency

    @property
    def op_params(self) -> Dict[str, Any]:
        return self._op_params

    @property
    def capability_sequence(self) -> List[Instruction]:
        return self._capability_sequence

    @property
    def loop_order(self):
        return self._loop_order

    @property
    def all_dims(self) -> List[str]:
        dims = []
        for i in self.inputs:
            dims += [ishp for ishp in i.shape_symbols if ishp not in dims]

        for o in self.outputs:
            dims += [oshp for oshp in o.shape_symbols if oshp not in dims]

        return dims

    @loop_order.setter
    def loop_order(self, loop_order):
        assert isinstance(loop_order, list)
        assert len(loop_order) == len(self.all_dims)
        for l in loop_order:
            assert l in self.all_dims
        self._loop_order = loop_order

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
            raise RuntimeError(f"Cannot set primitive sequence which has already been set.")
        self._capability_sequence = capability_sequence

    def add_input(self, name, dtypes, mem_path, shape, iter_domain=None, index=None):

        iteration_domain = iter_domain or shape
        inpt = OperandTemplate(name=name,
                               dtypes=dtypes,
                               memory_path=mem_path,
                               shape_symbols=shape,
                               iteration_domain=iteration_domain)
        if index:
            self.inputs.insert(index, inpt)
        else:
            self.inputs.append(inpt)

    def add_output(self, name: str, dtypes: List[Datatype],
                   mem_path: List[str],
                   shape: List[str],
                   iter_domain: List[str]=None,
                   index=None):
        iteration_domain = iter_domain or shape
        outpt = OperandTemplate(name=name, dtypes=dtypes, memory_path=mem_path, shape_symbols=shape, iteration_domain=iteration_domain)
        if index:
            self.outputs.insert(index, outpt)
        else:
            self.outputs.append(outpt)

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
        params = {}
        for k, v in self.op_params.items():
            if isinstance(v, Callable):
                v = get_lambda_source(v)
                param_type = "function"
            else:
                param_type = v.__class__.__name__
            params[k] = {'type': param_type, 'value': v}
        blob['op_params'] = params
        blob['latency'] = self.latency
        blob['source'] = [ipt.to_json() for ipt in self.inputs]
        blob['dest'] = [opt.to_json() for opt in self.outputs]
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


class CodeletInstance(object):
    CODELET_COUNTER = defaultdict(int)

    def __init__(self, capability_templates: List[InstructionTemplate],
                 inputs: List[OperandTemplate],
                 outputs: List[OperandTemplate],
                 codelet: Codelet,
                 compiler_params=None,
                 op_params=None,
                 loop_order=None):
        self._codelet_template = codelet
        self._capabilities = capability_templates
        self._inputs = inputs
        self._outputs = outputs
        self._op_params = op_params or {}
        self._compiler_params = compiler_params or {}
        self._cdlt_id = f"{codelet.name}{CodeletInstance.CODELET_COUNTER[codelet.name]}"
        self._loop_order = loop_order or codelet.loop_order
        CodeletInstance.CODELET_COUNTER[codelet.name] += 1

    @property
    def codelet_id(self):
        return self._cdlt_id

    @property
    def loop_order(self):
        return self._loop_order

    @property
    def inputs(self) -> List[OperandTemplate]:
        return self._inputs

    @property
    def outputs(self) -> List[OperandTemplate]:
        return self._outputs

    @property
    def op_params(self) -> Dict:
        return self._op_params

    @property
    def compiler_params(self) -> Dict:
        return self._compiler_params

    @property
    def codelet_template(self):
        return self._codelet_template

    @property
    def capabilities(self):
        return self._capabilities

    def get_text_instructions(self) -> List[str]:
        instr = [str(c) for c in self.capabilities]
        return instr

    def compiled_json(self) -> Dict:
        blob = {}
        blob['id'] = self.codelet_id
        blob['codelet_name'] = self.codelet_template.name
        blob['source'] = [ipt.to_json() for ipt in self.inputs]
        blob['dest'] = [opt.to_json() for opt in self.outputs]
        blob['parameters'] = self.op_params
        blob['capability_sequence'] = [c.compiled_json() for c in self.capabilities]
        return blob
