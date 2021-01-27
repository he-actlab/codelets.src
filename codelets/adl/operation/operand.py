from typing import Callable, Any, List, Dict, Optional, Tuple, Set, Union
from collections import namedtuple
import polymath as pm
from numbers import Integral
from copy import deepcopy
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


@dataclass
class OperandTemplate:
    name: str
    supported_dtypes: Union[List[Datatype]]
    shape_list: List[str]
    shape_symbols: Dict = field(default_factory=dict)
    tiling: Dict[str, List[int]] = field(default_factory=dict)
    sorted_tile_keys: List[str] = field(default_factory=list)
    dtype: Datatype = field(default=None)
    node_name: str = field(default=None)
    evaluated_tiling: List[Tuple[str, List]] = field(default_factory=list, init=False)
    dependencies: List[str] = field(default_factory=list)

    def is_dtype_supported(self, dtype_name) -> bool:
        return dtype_name in [str(dt) for dt in self.supported_dtypes]

    def update_shape_symbols(self, shape_key, value):
        if shape_key in self.shape_symbols and self.shape_symbols[shape_key] != value:
            raise KeyError(f"Value for shape {shape_key} has already been set:\n"
                           f"Previous value: {self.shape_symbols[shape_key]}\n"
                           f"New value: {value}")
        assert isinstance(value, Integral)
        self.shape_symbols[shape_key] = value

    def is_instantiated(self):
        return len(list(self.shape_symbols.keys())) == len(self.shape_list)

    def set_dtype(self, dtype: Datatype):
        assert self.is_dtype_supported(str(dtype))
        self.dtype = dtype

    def set_node_name(self, name):
        self.node_name = name

    @property
    def shape(self):
        s = []
        for sname in self.shape_list:
            s.append(self.shape_symbols[sname])
        return tuple(s)

    @property
    def unset_tiling(self):
        return [k for k, v in self.tiling.items() if len(v) == 0]

    def is_tiling_set(self, path_key):
        if path_key in self.tiling:
            return len(self.tiling[path_key]) == len(self.shape_symbols)
        else:
            return False

    def add_transfer(self, op_str: str, path_key: Tuple[str, str], dim_tiling: Dict[str, Union[str, int, None]]):
        if op_str not in self.dependencies:
            self.dependencies.append(op_str)
        self.add_path_tiling(path_key, dim_tiling)

    def add_path_tiling(self, path_key: Tuple[str, str], dim_tiling: Dict[str, Union[str, int, None]]):
        if self.is_tiling_set(path_key):
            raise RuntimeError(f"Tiling for operand {self.name} has already been set!")

        if len(self.sorted_tile_keys) == 0:
            self.sorted_tile_keys.append(path_key[0])

        self.tiling[path_key] = []
        for k, v in dim_tiling.items():
            self.tiling[path_key].append(v)

        # TODO: Check logic here to make sure this is a valid data transfer
        # assert path_key[0] == self.sorted_tile_keys[-1]
        self.sorted_tile_keys.append(path_key[1])

    def evaluate_operand(self, node: pm.Node, params: Dict, hag):

        initial_size = [self.shape_symbols[symb] for symb in self.shape_list]
        first_tile = True
        # TODO: Add checks here


        for path_key, tiling_values in self.tiling.items():
            if first_tile:
                self.evaluated_tiling.append((path_key[0], initial_size))
                first_tile = False
            dest_tiling = []
            for k in tiling_values:
                if isinstance(k, str):
                    dest_tiling.append(params[k].value)
                else:
                    assert isinstance(k, Integral)
                    dest_tiling.append(k)

            self.evaluated_tiling.append((path_key[1], dest_tiling))

    def copy(self):
        # TODO: Fix this
        op_temp = OperandTemplate(name=self.name,
                               supported_dtypes=self.supported_dtypes,
                                shape_list=self.shape_list,
                                shape_symbols=self.shape_symbols.copy(),
                                dependencies=self.dependencies.copy(),
                                tiling=deepcopy(self.tiling),
                                sorted_tile_keys=self.sorted_tile_keys.copy(),
                                dtype=self.dtype,
                                node_name=self.node_name)
        op_temp.evaluated_tiling = deepcopy(self.evaluated_tiling)
        return op_temp

    def emit(self, output_type):
        if output_type == "json":
            blob = {"name": self.name,
                    "dtype": str(self.dtype),
                    "shape": [[k, v] for k, v in self.shape_symbols.items()],
                    "tiling": {t[0]: t[1] for t in self.evaluated_tiling},
                    }
        else:
            raise TypeError(f"Unable to support output type for operand: {output_type}")
        return blob

    @staticmethod
    def from_json(ot_obj: Dict):
        ot = ot_obj.copy()
        name = ot['field_name']
        dtypes = [Datatype.from_json(dt) for dt in ot['supported_dtypes']]
        shape_symbols = ot['shape_symbols']
        return OperandTemplate(name=name,
                               dtypes=dtypes,
                               shape_symbols=shape_symbols)

    def to_json(self):
        blob = {}
        blob['field_name'] = self.name
        if isinstance(self.supported_dtypes, list):
            blob['supported_dtypes'] = [dt.to_json() for dt in self.supported_dtypes]
        else:
            blob['supported_dtypes'] = self.supported_dtypes.to_json()
        blob['shape_symbols'] = self.shape_symbols
        return blob
