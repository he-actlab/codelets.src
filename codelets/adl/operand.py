from typing import Callable, Any, List, Dict, Optional, Tuple, Set, Union
from collections import namedtuple
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class OperandTemplate:
    name: str
    dtypes: Union[List[Datatype], Datatype]
    shape_list: List[str]
    shape_symbols: Dict = field(default_factory=dict)

    def is_dtype_supported(self, dtype_name) -> bool:
        return dtype_name in [str(dt) for dt in self.dtypes]

    def to_json(self):
        blob = {}
        blob['name'] = self.name
        if isinstance(self.dtypes, list):
            blob['dtypes'] = [dt.to_json() for dt in self.dtypes]
        else:
            blob['dtypes'] = self.dtypes.to_json()
        blob['shape_symbols'] = self.shape_symbols
        return blob

    def is_instantiated(self):
        return isinstance(self.shape_symbols, dict)

    @property
    def shape(self):
        s = []
        for sname in self.shape_list:
            s.append(self.shape_symbols[sname])
        return tuple(s)

    @staticmethod
    def from_json(ot_obj: Dict):
        ot = ot_obj.copy()
        name = ot['name']
        dtypes = [Datatype.from_json(dt) for dt in ot['dtypes']]
        shape_symbols = ot['shape_symbols']
        return OperandTemplate(name=name,
                               dtypes=dtypes,
                               shape_symbols=shape_symbols)