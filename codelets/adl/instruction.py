import numpy as np
from collections import namedtuple
from typing import Callable, List, Dict, Optional, Union
from codelets.adl.backups.operand import Operand, NullOperand
from itertools import count
from dataclasses import dataclass, asdict, field
from types import FunctionType

CodeletOperand = namedtuple('OperandTemplate', ['field_name', 'dtypes'])
field_cnt = count()
INSTR_FN_NAME_TEMPLATE = """param_fn{FN_ID}"""
INSTR_FN_TEMPLATE = """def param_fn{FN_ID}(hag, op, cdlt, relocation_table, program, fixed_val=None): return {FN_BODY}"""

@dataclass
class Field:
    field_name: str
    bitwidth: int
    field_id: int = field(default_factory=lambda: next(field_cnt))
    value: int = None
    value_names: Dict[str, int] = None
    value_str: str = None
    param_fn: Union[Callable, str] = None

    @property
    def isset(self) -> bool:
        return self.value is not None

    @property
    def value_name_list(self) -> List[str]:
        return list(self.value_names.keys())

    def set_param_fn(self, fn):
        self.param_fn = fn

    def set_value(self, value):
        if self.value is not None:
            raise ValueError(f"{self.field_name} has already been set to {self.value}")
        self.value = value

    def set_value_by_string(self, value_name) -> None:
        if value_name not in self.value_names:
            raise ValueError(f"{value_name} is not mapped to any values for this field.\n"
                             f"Possible values: {self.value_name_list}")
        elif self.value is not None and self.value_str != value_name:
            raise ValueError(f"Value for {self.field_name} has already been set to {self.value_str}, cannot set to {value_name}")
        else:
            self.value_str = value_name
            self.value = self.value_names[value_name]

    def get_string_value(self) -> Union[str, int]:
        if self.value_str is not None:
            return self.value_str
        return self.value

    def get_param_fnc(self):
        param_fnc_str = INSTR_FN_TEMPLATE.format(FN_ID=self.field_id, FN_BODY=self.param_fn)
        param_fnc_code = compile(param_fnc_str, "<string>", "exec")
        param_fnc = FunctionType(param_fnc_code.co_consts[0], globals(), INSTR_FN_NAME_TEMPLATE.format(FN_ID=self.field_id))
        return param_fnc

    def run_param_fnc(self, *args, **kwargs):
        param_fnc = self.get_param_fnc()
        return param_fnc(*args, **kwargs)


class InstructionTemplate(object):
    def __init__(self, target, fields: List[Field], field_values=None):
        self._target = target
        self._fields = fields
        self._field_values = field_values or {}
        self._field_map = {f.field_name: f for f in fields}

    def __str__(self):
        start = f"{self.fields[0].value_str} "
        rest = []
        for f in self.fields[1:]:
            if f.isset:
                val = f.value_str if isinstance(f.value_str, str) else str(f.value)
            else:
                val = f"$({f.field_name})"
            rest.append(val)
        return start + ", ".join(rest)

    @property
    def field_values(self) -> Dict[str, Field]:
        return self._field_values

    @property
    def target(self) -> str:
        return self._target

    @property
    def field_map(self) -> Dict[str, Field]:
        return self._field_map

    @property
    def fields(self) -> List[Field]:
        return self._fields

    @property
    def field_names(self) -> List[str]:
        return [f.field_name for f in self.fields]

    @property
    def unset_field_names(self) -> List[str]:
        return [f.field_name for f in self.fields if not f.isset]

    @property
    def field_widths(self) -> Dict[str, int]:
        return {k: v.bitwidth for k, v in self.field_values.items()}

    @property
    def set_fields(self) -> List[str]:
        return [k for k, v in self.field_values.items() if v.value is not None]

    @property
    def set_field_map(self) -> Dict[str, str]:
        return {k: v.get_string_value() for k, v in self.field_values.items() if v.isset}

    def get_field_value(self, name: str) -> int:
        if name not in self.field_names:
            raise ValueError(f"{name} is not a field for this capability:\n"
                             f"Fields: {self.fields}")
        elif self.field_values[name].value is None:
            raise KeyError(f"{name} is not currently set for this capability:\n"
                             f"Set fields: {self.set_fields}")

        return self.field_values[name].value

    def get_field(self, name: str) -> Field:
        if name not in self.field_names:
            raise ValueError(f"{name} is not a field for this capability:\n"
                             f"Fields: {self.field_names}")
        return self.field_map[name]

    def set_field_by_name(self, field_name: str, value_name: str):
        field = self.get_field(field_name)
        field.set_value_by_string(value_name)

    def set_field_value(self, name: str, value: int, value_str: Optional[str] = None):
        if name not in self.field_names:
            raise ValueError(f"{name} is not a field for this capability:\n"
                             f"Fields: {self.fields}")
        elif name in self.field_values:
            raise KeyError(f"{name} is already set for this capability:\n"
                             f"Set fields: {self.set_fields}")
        self.field_map[name].value = value

        if value_str:
            self.field_map[name] = value_str

    def compiled_json(self):
        blob = {}
        blob['target_name'] = self.fields[0].value_str
        blob['op_fields'] = []
        for f in self.fields[1:]:
            value = f.get_string_value()
            blob['op_fields'].append({'field_name': f.field_name, 'value': value})
        return blob

    def template_json(self):
        blob = {}
        blob['target_name'] = self.fields[0].value_str
        blob['op_fields'] = []
        for f in self.fields[1:]:
            value = f.get_string_value()
            blob['op_fields'].append({'field_name': f.field_name, 'value': value})
        return blob

    def copy(self):
        fields = [Field(**asdict(f)) for f in self.fields]
        return InstructionTemplate(self.target, fields, field_values=self.field_values)

    def to_json(self):
        return {self.fields[0].value_str: [asdict(f) for f in self.fields[1:]]}

class Instruction(object):

    # TODO: Add test for
    def __init__(self, opname, opcode, opcode_width, fields, target=None, field_values=None, latency=1, **kwargs):
        self._opname = opname
        self._opcode = opcode
        self._opcode_width = opcode_width
        self._extra_params = kwargs
        self._latency = latency
        self._target = target
        self._fields = fields
        self._field_values = field_values or {}
        self._field_map = {f.field_name: f for f in fields}

    @property
    def name(self) -> str:
        return self._opname

    @property
    def opcode(self) -> int:
        return self._opcode

    @property
    def target(self):
        return self._target

    @property
    def opcode_width(self) -> int:
        return self._opcode_width

    @property
    def extra_params(self) -> Dict:
        return self._extra_params

    def bin(self) -> str:
        return np.binary_repr(self.opcode, self.opcode_width)

    @property
    def latency(self) -> Union[int, Callable]:
        return self._latency

    @target.setter
    def target(self, target: str):
        self._target = target

    @property
    def opname(self):
        return self._opname

    def add_param_fn(self, field_name, fn: Callable):
        field = self.get_field(field_name)
        field.set_param_fn(fn)

    def instruction_copy(self):
        fields = [Field(**asdict(f)) for f in self.fields]
        instr = Instruction(self.opname, self.opcode, self.opcode_width, fields,
                            target=self.target,
                            field_values=self.field_values,
                            latency=self.latency)
        return instr

    def __str__(self):
        start = f"{self.opname} "
        rest = []
        for f in self.fields:
            if f.isset:
                val = f.value_str if isinstance(f.value_str, str) else str(f.value)
            else:
                val = f"$({f.field_name})"
            rest.append(val)
        return start + ", ".join(rest)

    @property
    def field_values(self) -> Dict[str, Field]:
        return self._field_values

    @property
    def field_map(self) -> Dict[str, Field]:
        return self._field_map

    @property
    def fields(self) -> List[Field]:
        return self._fields

    @property
    def field_names(self) -> List[str]:
        return [f.field_name for f in self.fields]

    @property
    def unset_field_names(self) -> List[str]:
        return [f.field_name for f in self.fields if not f.isset]

    @property
    def field_widths(self) -> Dict[str, int]:
        return {k: v.bitwidth for k, v in self.field_values.items()}

    @property
    def set_fields(self) -> List[str]:
        return [k for k, v in self.field_values.items() if v.value is not None]

    @property
    def set_field_map(self) -> Dict[str, str]:
        return {k: v.get_string_value() for k, v in self.field_values.items() if v.isset}

    def get_field_value(self, name: str) -> int:
        if name not in self.field_names:
            raise ValueError(f"{name} is not a field for this capability:\n"
                             f"Fields: {self.fields}")
        elif self.field_values[name].value is None:
            raise KeyError(f"{name} is not currently set for this capability:\n"
                             f"Set fields: {self.set_fields}")

        return self.field_values[name].value

    def get_field(self, name: str) -> Field:
        if name not in self.field_names:
            raise ValueError(f"{name} is not a field for this capability:\n"
                             f"Fields: {self.field_names}")
        return self.field_map[name]

    def set_field_by_name(self, field_name: str, value_name: str):
        field = self.get_field(field_name)
        field.set_value_by_string(value_name)

    def set_field_value(self, name: str, value: int, value_str: Optional[str] = None):
        if name not in self.field_names:
            raise ValueError(f"{name} is not a field for this capability:\n"
                             f"Fields: {self.fields}")
        elif name in self.field_values:
            raise KeyError(f"{name} is already set for this capability:\n"
                             f"Set fields: {self.set_fields}")
        self.field_map[name].value = value

        if value_str:
            self.field_map[name] = value_str

    def set_field_param_fn(self, name: str, param_fn: str):
        if name not in self.field_names:
            raise ValueError(f"{name} is not a field for this capability:\n"
                             f"Fields: {self.fields}")
        field = self.get_field(name)
        if field.param_fn is not None:
            raise ValueError(f"Param function for {name} is already set:\n"
                             f"Set param fn: {field.param_fn}\n"
                             f"New param fn: {param_fn}")
        field.set_param_fn(param_fn)

    def get_operand(self, opname) -> Operand:
        for o in self.operands:
            if o.name == opname:
                return o
        raise KeyError(f"{opname} not found in possible codelet source!")

    def get_operand_names(self) -> List[str]:
        return [o.name for o in self.operands]

    # TODO: handle preset field values
    def create_template(self) -> InstructionTemplate:
        fields = [Field('opcode', self.opcode_width, value=self.opcode, value_str=self.name)]
        for o in self.operands:
            if isinstance(o, NullOperand):
                f = Field(o.name, o.bits, value=0, value_str="X")
            else:
                f = Field(o.name, o.bits, value_names=o.str_value_map)
            fields.append(f)
            if o.index_size > 0:
                fields.append(Field(f"{o.name}_index", o.index_size))
        return InstructionTemplate(self.target, fields)

    def to_json(self):
        blob = {}
        blob['target_name'] = self.name
        blob['opcode'] = self.opcode
        blob['opcode_width'] = self.opcode
        blob['target'] = self.target
        # Removing this to prevent confusion temporarily
        # blob['opcode_width'] = self.opcode_width
        blob['latency'] = self.latency
        blob['extra_params'] = self.extra_params
        # blob['source'] = [o.to_json() for o in self.operands]
        return blob