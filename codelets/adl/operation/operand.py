from typing import Callable, Any, List, Dict, Optional, Tuple, Set, Union, ClassVar
from collections import namedtuple
from functools import partial
from . import pairwise
import polymath as pm
from numbers import Integral
from copy import deepcopy
from sympy import Basic, Idx, symbols, Integer
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
class DataAccess:
    hag_node: str
    operand_name: str
    access_type: str
    shape_symbols: List[str]
    op_name: Union[str, None]
    shape_map: Dict[str, Integral]
    offset_map: Dict[str, List[Union[int, str, Basic]]] = field(default_factory=dict)
    src_node: str = field(default=None)
    dst_node: str = field(default=None)

    @property
    def shape(self):
        if all(v is not None for v in self.shape_map.values()):
            return [self.shape_map[s] for s in self.shape_symbols]
        else:
            raise RuntimeError(f"Shape is unevaluated")

    def is_set(self):
        return len(self.shape_map) == len(self.shape_symbols)

    def resolve_shape_map(self, sizes):
        self.shape_map = {self.shape_symbols[i]: sizes[i] for i in range(len(sizes))}

    def resolve_offsets(self, cdlt):
        for name, o in self.offset_map.items():
            if isinstance(o, Basic):
                indices = list(o.atoms(Idx))
                others = [i for i in list(o.free_symbols) if i not in indices]
                max_vals = {str(i): cdlt.op_map[str(i)].end - 1 for i in indices}
                max_vals.update({str(i): cdlt.required_params[str(i)].value for i in others})
                size = self.resolve_offset(o, max_vals) + 1
            else:
                size = o
            self.offset_map[name] = size


    def resolve_offset(self, expr: Basic, values: Dict[str, int]):
        for f_sym in list(expr.free_symbols):
            if str(f_sym) in values:
                expr = expr.subs(f_sym, values[str(f_sym)])
        if not isinstance(expr, (Integer, Integral)):
            raise TypeError(f"Unable to compute domain offsets because offset is not an integer:"
                            f"Offset: {expr}\tType: {type(expr)}")
        return int(expr)

    def is_compute(self):
        return "compute" in self.op_name

    def copy(self):
        return DataAccess(self.hag_node,
                          self.operand_name,
                          self.access_type,
                          deepcopy(self.shape_symbols),
                          self.op_name,
                          deepcopy(self.shape_map),
                          deepcopy(self.offset_map),
                          self.src_node,
                          self.dst_node
                          )

@dataclass
class DataMovement:
    src_node: str
    dst_node: str
    operand_name: str
    shape_symbols: List[str]
    op_name: Union[str, None]
    shape_map: Dict[str, Integral]
    offset_map: Dict[str, List[Union[int, str, Basic]]] = field(default_factory=dict)

    def copy(self):
        return DataMovement(self.src_node,
                          self.dst_node,
                          self.operand_name,
                          deepcopy(self.shape_symbols),
                          self.op_name,
                          deepcopy(self.shape_map),
                          deepcopy(self.offset_map)
                          )

@dataclass
class OperandTemplate:
    name: str
    supported_dtypes: Union[List[Datatype]]
    shape_list: List[str]
    shape_symbols: Dict = field(default_factory=dict)
    tiling: Dict[str, List[int]] = field(default_factory=dict)
    data_path: List[str] = field(default_factory=list)
    dtype: Datatype = field(default=None)
    node_name: str = field(default=None)
    evaluated_tiling: List[Tuple[str, List]] = field(default_factory=list, init=False)
    dependencies: List[str] = field(default_factory=list)
    data_moves: List[DataAccess] = field(default_factory=list)
    test_data_moves: List[DataMovement] = field(default_factory=list)
    required_params: List[str] = field(default_factory=list)
    current_codelet: ClassVar = field(default=None)

    @property
    def current_location(self):
        if len(self.data_path) == 0:
            return None
        else:
            return self.data_path[-1]

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            assert not isinstance(item, list)
            item = (item,)
        offsets = []
        for idx in item:
            if isinstance(idx, Basic):
                off = idx
                loop_idx = idx.atoms(Idx)
                self.dependencies += [str(l) for l in list(loop_idx) if str(l) not in self.dependencies]
            elif isinstance(idx, str):
                off = symbols(idx, integer=True)
                self.required_params.append(idx)
            elif isinstance(idx, int):
                off = idx
            else:
                try:
                    off = idx.param_symbols[idx.op_str]
                except TypeError:
                    off = idx
                self.dependencies.append(idx.op_str)
            offsets.append(off)
        assert len(offsets) == len(self.shape_list)
        return IndexedOperandTemplate(self, offsets)

    def get_access_offsets(self, offsets):
        if len(offsets) == 0:
            offsets = {self.shape_list[i]: None for i in range(len(self.shape_list))}
        else:
            offsets = {self.shape_list[i]: offsets[i] for i in range(len(self.shape_list))}
        return offsets

    def add_data_access(self, offsets, op_str, access_type, size, access_node=None, path_key=None):
        if "compute" in op_str:
            self.add_compute_access(offsets, op_str, access_type, size, access_node, path_key)
        elif "transfer" in op_str:
            self.add_transfer_access(offsets, op_str, access_type, size, access_node, path_key)
        # TODO: Check if the revious access was a write/read
        shapes = self.get_shape_map(size)

        if path_key is not None:
            if len(path_key) == 1:
                path_key = (path_key[0], self.current_location) if access_type == "read" else (self.current_location, path_key[0])
            self.add_path_tiling(path_key, shapes)
        return self

    def add_dependency(self, op_name):
        if op_name not in self.dependencies:
            self.dependencies.append(op_name)

    def add_transfer_access(self, offsets, op_str, access_type, size, access_node=None, path_key=None):
        shapes = self.get_shape_map(size)
        offsets = self.get_access_offsets(offsets)
        self.add_dependency(op_str)

        if len(self.data_moves) > 0 and self.data_moves[-1].is_compute() and self.data_moves[-1].hag_node is None:
            assert access_type == "read" and self.data_moves[-1].access_type == "write"
            self.data_moves[-1].hag_node = access_node
        assert access_node is not None
        data_move = DataAccess(access_node, self.name, access_type, self.shape_list.copy(), op_str, shapes, offsets)
        self.data_moves.append(data_move)


    def add_compute_access(self, offsets, op_str, access_type, size, access_node=None, path_key=None):
        shapes = self.get_shape_map(size)
        offsets = self.get_access_offsets(offsets)
        self.add_dependency(op_str)

        assert len(path_key) == 1
        # This is a compute node reading from an input
        if access_type == "read":
            if self.current_location != path_key[0]:
                data_move = DataAccess(self.current_location, self.name, "read", self.shape_list.copy(), op_str, shapes,
                                       offsets)
                self.data_moves.append(data_move)
                data_move = DataAccess(path_key[0], self.name, "write", self.shape_list.copy(), op_str, shapes,
                                       offsets)
                self.data_moves.append(data_move)

            else:
                assert len(self.data_moves) == 0 or (self.data_moves[-1].hag_node == path_key[0] and
                                                     self.data_moves[-1].access_type == "write")
            data_move = DataAccess(path_key[0], self.name, "read", self.shape_list.copy(), op_str, shapes,
                                   offsets)
            self.data_moves.append(data_move)

        else:

            if self.current_location != path_key[0]:
                data_move = DataAccess(self.current_location, self.name, "read", self.shape_list.copy(), op_str, shapes,
                                       offsets)
                self.data_moves.append(data_move)
            else:
                assert len(self.data_moves) == 0 or (self.data_moves[-1].hag_node == path_key[0] and
                                                     self.data_moves[-1].access_type == "write")

                data_move = DataAccess(None, self.name, "write", self.shape_list.copy(), op_str, shapes,
                                       offsets)
                self.data_moves.append(data_move)

    def test_add_transfer_access(self, path, op_name, sizes, offsets=None):
        offsets = offsets or []
        self.add_dependency(op_name)
        pairs = pairwise(path)
        for i, (src, dst) in enumerate(pairs):
            if src == path[0]:
                if len(self.test_data_moves) > 1 and self.test_data_moves[-1].dst_node is None:
                    self.test_data_moves[-1].dst_node = src
                    continue
                dm_offsets = self.get_access_offsets(offsets)
            else:
                dm_offsets = self.get_access_offsets([0] * len(self.shape_list))
            shape = self.get_shape_map(sizes[i])
            movement = DataMovement(src, dst, self.name, self.shape_list.copy(), op_name, shape, dm_offsets)
            self.test_data_moves.append(movement)

    def test_add_compute_access(self, target, op_name, operand_type, offsets=None):

        assert operand_type in ["source", "dest"]
        offsets = offsets or []
        offsets = self.get_access_offsets(offsets)
        self.add_dependency(op_name)
        shape = self.get_shape_map([0] * len(self.shape_list))
        if operand_type == "source":
            src = self.current_location
            dst = target
        else:
            src = target
            dst = None
        movement = DataMovement(src, dst, self.name, self.shape_list.copy(), op_name, shape, offsets)
        self.test_data_moves.append(movement)

    def is_dtype_supported(self, dtype_name) -> bool:
        return dtype_name in [str(dt) for dt in self.supported_dtypes]

    def update_shape_symbols(self, shape_key, value):
        if shape_key in self.shape_symbols and self.shape_symbols[shape_key] != value:
            raise KeyError(f"Value for shape_symbols {shape_key} has already been set:\n"
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

    def get_shape_map(self,  size):
        if len(size) == len(self.shape_list):
            shape_dict = {s: size[idx] for idx, s in enumerate(self.shape_list)}
        elif len(size) == 0:
            shape_dict = {}
        else:
            assert len(size) == 1
            shape_dict = {s: 1 for idx, s in enumerate(self.shape_list)}
            shape_dict[self.shape_list[-1]] = size[0]
        return shape_dict


    def add_path_tiling(self, path_key: Tuple[str, str], dim_tiling: Dict[str, Union[str, int, None]]):
        if self.is_tiling_set(path_key):
            raise RuntimeError(f"Tiling for operand {self.name} has already been set!")

        if len(self.data_path) == 0:
            self.data_path.append(path_key[0])

        self.tiling[path_key] = []
        for k, v in dim_tiling.items():
            self.tiling[path_key].append(v)

        # TODO: Check logic here to make sure this is a valid data transfer
        # assert path_key[0] == self.data_path[-1]
        if self.current_location != path_key[1]:
            self.data_path.append(path_key[1])


    def set_start_location(self, location: str):
        if len(self.data_path) > 0:
            raise RuntimeError(f"Unable to set default location for operand {self.name} because "
                               f"path has already been initialized: {self.data_path}")
        self.data_path.append(location)

    def evaluate_operand(self, node: pm.Node, hag, cdlt):

        initial_size = [self.shape_symbols[symb] for symb in self.shape_list]
        level_shapes = {}
        assert len(hag.node_levels[0]) == 1
        level_shapes[0] = initial_size

        for i, access in enumerate(self.data_moves):
            access_level = hag.get_node_level(access.hag_node)

            if access_level in level_shapes:
                access.resolve_shape_map(level_shapes[access_level])
            else:

                access.resolve_offsets(cdlt)


        first_tile = True
        # TODO: Add checks here
        for path_key, tiling_values in self.tiling.items():
            if first_tile:
                self.evaluated_tiling.append((path_key[0], initial_size))
                first_tile = False
            dest_tiling = []
            for k in tiling_values:
                if isinstance(k, str):
                    dest_tiling.append(cdlt.params[k].value)
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
                                  data_path=self.data_path.copy(),
                                  dtype=self.dtype,
                                  node_name=self.node_name,
                                  data_moves=deepcopy(self.data_moves),
                                  test_data_moves=deepcopy(self.test_data_moves))
        op_temp.evaluated_tiling = deepcopy(self.evaluated_tiling)
        return op_temp

    def emit(self, output_type):
        if output_type == "json":
            blob = {"name": self.name,
                    "dtype": str(self.dtype),
                    "shape_symbols": [[k, v] for k, v in self.shape_symbols.items()],
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

@dataclass
class IndexedOperandTemplate:
    operand_template: OperandTemplate
    offsets: List[Union[int, str, Basic]]

    def test_add_transfer_access(self, path, op_name, sizes):
        self.operand_template.test_add_transfer_access(path, op_name, sizes, self.offsets)

    def test_add_compute_access(self, target, op_name, operand_type):
        self.operand_template.test_add_compute_access(target, op_name, operand_type, self.offsets)

@dataclass(frozen=True)
class Offset:
    dim: int
    loop_id: int
    stride: int
    dim_size: int
    offset: int

# #TODO: Add strided view function
# @dataclass
# class OperandOffset:
#     operand: OperandTemplate
#     offsets: List[Union[int, str, Basic]]
#     data_source: str = field(default=None)
#
#     @property
#     def shape_list(self):
#         return self.operand.shape_list
#
#     @property
#     def dependencies(self):
#         return self.operand.dependencies
#
#     @property
#     def operand_name(self):
#         return self.operand.name
#
#     def add_transfer(self, op_str, key, shape_dict, offsets):
#         assert offsets == self.offsets
#         self.operand.add_transfer(op_str, key, shape_dict, self.offsets)
#
#     def add_compute(self, op_str, compute_target, offsets):
#         assert offsets == self.offsets
#         self.operand.add_compute(op_str, compute_target, self.offsets)
#
#     def copy(self, cdlt):
#         operand = cdlt.get_operand(self.operand_name)
#         return OperandOffset(operand, self.offsets.copy(), self.data_source)
#
#     def path_offset(self, xfer_path: List[str]):
#         if len(self.operand.data_path) == 0:
#             return xfer_path[0]
#         else:
#             return xfer_path[-1]




