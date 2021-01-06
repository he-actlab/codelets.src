from typing import Callable, Any, List, Dict, Optional, Tuple, Set, Union
from collections import namedtuple
from dataclasses import dataclass

@dataclass(frozen=True)
class Datatype:
    type: str
    bitwidth: int

    def __str__(self):
        return f"{self.type}{self.bitwidth}"

    def to_json(self):
        blob = {}
        blob['type'] = self.type
        blob['bitwidth'] = self.bitwidth
        return blob

    @staticmethod
    def from_json(dt_obj: Dict):
        return Datatype(type=dt_obj['type'], bitwidth=dt_obj['bitwidth'])

    @staticmethod
    def from_str(dt_str: str):
        idx = 0

        while not dt_str[idx].isdigit() and idx < len(dt_str):
            idx += 1
        type_part = dt_str[:idx].upper()
        bit_part = int(dt_str[idx:])
        return Datatype(type=type_part, bitwidth=bit_part)

OPERAND_TYPES = ['constant', 'storage', 'compute', 'fill']

class Operand(object):
    def __init__(self, name, operand_type, dtypes, bits,
                 components=None,
                 index_size=None,
                 value_names=None,
                 extra_params=None,
                 data_dims=None,
                 fill_value=None):
        self._data_dimensions = data_dims or [1]
        # TODO: If components are filled in and value_names, validate against each other
        self._name = name
        if operand_type not in OPERAND_TYPES:
            raise ValueError(f"Invalid operand type: {operand_type}."
                             f"\nPossible values: {OPERAND_TYPES}")

        self._operand_type = operand_type
        assert isinstance(dtypes, list)
        self._dtypes = dtypes
        self._bits = bits
        self._index_size = index_size if isinstance(index_size, int) else 0
        if extra_params:
            assert isinstance(extra_params, dict)
            self._extra_params = extra_params
        else:
            self._extra_params = {}
        self._components = components
        if operand_type in ["storage", "compute"]:
            assert components and isinstance(components, dict)
        elif operand_type == "fill":
            assert fill_value is not None
            self._fill_value = fill_value

        self._value_string_map = {}
        if value_names:
            for i, v in enumerate(value_names):
                self._value_string_map[i] = v

    @property
    def name(self) -> str:
        return self._name

    @property
    def bits(self) -> int:
        return self._bits

    @property
    def value_str_map(self) -> Dict[int, str]:
        return self._value_string_map

    @property
    def str_value_map(self) -> Dict[str, int]:
        return {v: k for k, v in self.value_str_map.items()}

    @property
    def components(self) -> Dict[str, int]:
        return self._components

    @property
    def index_size(self) -> int:
        return self._index_size

    @property
    def total_width(self) -> int:
        return self.index_size + self.bits

    @property
    def operand_type(self) -> str:
        return self._operand_type

    @property
    def dtypes(self):
        return self._dtypes

    def get_valid_components(self) -> List[str]:
        return list(self.components.keys())

    def is_fill_op(self) -> bool:
        return False

    def to_json(self) -> Dict:
        blob = {}
        blob['field_name'] = self.name
        blob['field_type'] = self.operand_type
        blob['bitwidth'] = self.bits
        blob['dtypes'] = [f"{d.type.upper()}{d.bitwidth}" for d in self._dtypes]
        blob['possible_values'] = list(self.str_value_map.keys())
        blob['index_size'] = self.index_size
        blob['components'] = self.components
        return blob

class NullOperand(Operand):
    def __init__(self, bitwidth, fill_value):
        super().__init__("null", "fill", [], bitwidth, fill_value=fill_value)

    def is_fill_op(self):
        return True
