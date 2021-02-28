from typing import Callable, Any, List, Dict, Optional, Tuple, Set, Union, ClassVar
from collections import namedtuple
from functools import partial
from pytools import memoize
from collections import defaultdict
import numpy as np
from . import pairwise
import polymath as pm
from numbers import Integral
from copy import deepcopy
from sympy import Basic, Idx, symbols, Integer
from codelets.adl import util
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

    def bytes(self):
        return self.bitwidth // 8

@dataclass
class DataMovement:
    src_node: str
    dst_node: str
    operand_name: str
    shape_symbols: List[str]
    op_name: Union[str, None]
    shape_map: Dict[str, Integral]
    offset_map: Dict[str, List[Union[int, str, Basic]]] = field(default_factory=dict)
    evaluated_offsets: Dict[str, List[Union[int, str, Basic]]] = field(default_factory=dict)
    evaluated_domain_offsets: Dict[str, List[Union[int, str, Basic]]] = field(default_factory=dict)

    def __str__(self):
        path = f"PATH: {self.src_node}->{self.dst_node}"
        op = f"OP: {self.op_name}"
        offsets = f"OFFSETS: {self.offset_map}"
        return ", ".join([path, op, offsets])

    @property
    def shape_list(self):
        return list(self.shape_map.values())

    def size(self):
        return np.prod(self.dim_sizes())

    def dim_sizes(self):
        return [self.shape_map[s] for s in self.shape_symbols]

    def domain_offsets(self):
        all_offsets = []
        for k, v in self.evaluated_domain_offsets.items():
            for i in v:
                all_offsets.append(i)
        return all_offsets

    def copy(self):
        return DataMovement(self.src_node,
                          self.dst_node,
                          self.operand_name,
                          deepcopy(self.shape_symbols),
                          self.op_name,
                          deepcopy(self.shape_map),
                          deepcopy(self.offset_map),
                          deepcopy(self.evaluated_offsets),
                          deepcopy(self.evaluated_domain_offsets)
                          )
    def substitute_offset_symbols(self, cdlt, dep_map, replacements):
        new_offset_map = {}
        for name, o in self.offset_map.items():
            if isinstance(o, Basic):
                indices = list(o.atoms(Idx))
                for idx, i in enumerate(indices):
                    if str(i) in replacements and str(i) in dep_map:
                        replacement_op = cdlt.op_map[dep_map[str(i)]]
                        assert replacement_op.op_type == "loop"
                        o.subs(i, replacement_op.get_symbol())
                new_offset_map[name] = o
            else:
                new_offset_map[name] = 0
        self.offset_map = new_offset_map
        self.set_size_from_splits(cdlt, cdlt.domain_tiling)
        self.set_offset_map(cdlt, cdlt.domain_loop_map)

    def get_size_from_splits(self, cdlt, splits):
        sizes = {}
        for name, o in self.offset_map.items():
            if isinstance(o, Basic):
                indices = list(o.atoms(Idx))
                others = [i for i in list(o.free_symbols) if i not in indices]
                max_vals = {}
                for idx, i in enumerate(indices):
                    assert cdlt.op_map[str(i)].end % splits[str(i)] == 0
                    max_vals[str(i)] = cdlt.op_map[str(i)].end // splits[str(i)] - 1
                max_vals.update({str(i): cdlt.required_params[str(i)].value for i in others})

                size = self.resolve_offset(o, max_vals) + 1
                # TODO: Add logic here to check for zero values
            else:
                size = o
            sizes[name] = size

        return sizes

    def get_size_from_loops(self, cdlt, loops):
        sizes = {}
        for name, o in self.offset_map.items():
            if isinstance(o, Basic):
                indices = list(o.atoms(Idx))
                others = [i for i in list(o.free_symbols) if i not in indices]
                max_vals = {}
                for idx, i in enumerate(indices):
                    assert str(i) in loops
                    max_vals[str(i)] = loops[str(i)] - 1

                max_vals.update({str(i): cdlt.required_params[str(i)].value for i in others})
                size = self.resolve_offset(o, max_vals) + 1
                # TODO: Add logic here to check for zero values
            else:
                size = o
            sizes[name] = size

        return sizes

    def set_size_from_splits(self, cdlt, split_levels):
        src_level = cdlt.get_tile_level(self.src_node)
        dst_level = cdlt.get_tile_level(self.dst_node)
        if src_level > dst_level:
            level = dst_level
        else:
            level = src_level
        splits = defaultdict(lambda: 1)
        for lev in range(level):
            for key, loop in split_levels[lev+1].items():
                splits[key] *= loop
        self.shape_map = self.get_size_from_splits(cdlt, splits)

    def set_offset_map(self, cdlt, loop_shapes):
        self.resolve_domain_offsets(cdlt, loop_shapes)

    def resolve_shape_map(self, sizes):
        self.shape_map = {self.shape_symbols[i]: sizes[i] for i in range(len(sizes))}

    def resolve_domain_offsets(self, cdlt, loop_shapes):
        for idx, (name, o) in enumerate(self.offset_map.items()):
            idx_offset = int(np.prod(self.shape_list[idx + 1:]))
            if isinstance(o, Basic):
                indices = list(o.atoms(Idx))
                dom_offsets = self.resolve_domain_offset(cdlt, o, indices, idx, loop_shapes)
            else:
                dom_offsets = [Offset(idx, -1, idx_offset, int(self.shape_list[idx]), o)]
            self.evaluated_domain_offsets[name] = dom_offsets

    def resolve_domain_offset(self, cdlt, expr, indices, dim, loop_shapes):
        offsets = []
        for f_sym in list(expr.free_symbols):
            if str(f_sym) in cdlt.required_params:
                expr = expr.subs(f_sym, cdlt.required_params[str(f_sym)].value)
        offset, coeffs, nonlin = util.split(expr, indices)
        dim_size = self.shape_list[dim]
        base_offset = int(np.prod(self.shape_list[dim + 1:]))

        for i, idx in enumerate(indices):
            coeff = coeffs[i]
            coeff *= base_offset
            if not isinstance(coeff, (Integer, Integral)) or not isinstance(dim_size, (Integer, Integral)):
                raise TypeError(f"Unable to compute domain offsets because coefficient is not an intenger:"
                                f"Coeff: {coeff}\tType: {type(coeff)}")
            o_info = Offset(dim, cdlt.op_map[str(idx)].loop_id, int(coeff), int(dim_size), offset)
            offsets.append(o_info)
        return offsets


    def resolve_offsets(self, cdlt):
        for idx, (name, o) in enumerate(self.offset_map.items()):
            if isinstance(o, Basic):
                indices = list(o.atoms(Idx))
                others = [i for i in list(o.free_symbols) if i not in indices]
                max_vals = {str(i): cdlt.op_map[str(i)].end - 1 for i in indices}
                max_vals.update({str(i): cdlt.required_params[str(i)].value for i in others})
                size = self.resolve_offset(o, max_vals) + 1
            else:
                size = o
            self.evaluated_offsets[name] = size

    def resolve_offset(self, expr: Basic, values: Dict[str, int]):
        for f_sym in list(expr.free_symbols):
            if str(f_sym) in values:
                expr = expr.subs(f_sym, values[str(f_sym)])
        if not isinstance(expr, (Integer, Integral)):
            raise TypeError(f"Unable to compute domain domain_offsets because offset is not an integer:"
                            f"Offset: {expr}\tType: {type(expr)}")
        return int(expr)




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
    data_moves: List[DataMovement] = field(default_factory=list)
    required_params: List[str] = field(default_factory=list)
    current_codelet: ClassVar = field(default=None)

    @property
    def current_location(self):
        if len(self.data_path) == 0:
            return None
        else:
            return self.data_path[-1]

    def transfer_tile(self, transfer_op):
        if transfer_op.path[0] not in self.tiling:
            movement = transfer_op.get_src_movement(transfer_op.path[0], transfer_op.path[1])
            self.tiling[transfer_op.path[0]] = movement.shape_map
        else:
            # TODO: Check if already set
            pass


    def compute_tile(self, compute_op, operand_type):
        if operand_type == "source":
            movement = compute_op.get_src_movement(self.name)
        elif operand_type == "dest":
            movement = compute_op.get_dest_movement(self.name)
        else:
            raise RuntimeError(f"Invalid operand type {operand_type} for tiling computation.\n"
                               f"Possible values: 'source', 'dest'")

        if movement.src_node not in self.tiling:
            self.tiling[movement.src_node] = movement.shape_map

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
            offsets = {self.shape_list[i]: 0 for i in range(len(self.shape_list))}
        else:
            offsets = {self.shape_list[i]: offsets[i] for i in range(len(self.shape_list))}
        return offsets

    def add_dependency(self, op_name):
        if op_name not in self.dependencies:
            self.dependencies.append(op_name)

    def add_transfer_access(self, path, op_name, sizes, offsets=None):
        offsets = offsets or []
        self.add_dependency(op_name)
        pairs = pairwise(path)
        for i, (src, dst) in enumerate(pairs):
            if self.current_location != src:
                self.data_path.append(src)

            if self.current_location != dst:
                self.data_path.append(dst)

            if src == path[0]:
                if len(self.data_moves) > 1 and self.data_moves[-1].dst_node is None:
                    if self.data_moves[-1].src_node == src:
                        self.data_moves[-1].dst_node = dst
                        # continue
                    else:
                        self.data_moves[-1].dst_node = src
                # dm_offsets = self.get_access_offsets(domain_offsets)

            # else:
                # dm_offsets = self.get_access_offsets([0] * len(self.shape_list))
            dm_offsets = self.get_access_offsets(offsets)
            shape = self.get_shape_map(sizes[i])
            movement = DataMovement(src, dst, self.name, self.shape_list.copy(), op_name, shape, dm_offsets)
            self.data_moves.append(movement)

        return self

    def add_compute_access(self, target, op_name, operand_type, offsets=None):

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
        self.data_moves.append(movement)

        if self.current_location != target:
            self.data_path.append(target)

        return self

    def update_transfer_access(self, new_op):
        pairs = list(pairwise(new_op.path))
        for a in self.data_moves:
            key = (a.src_node, a.dst_node)
            if key in pairs:
                a.op_name = new_op.op_str
        self.transfer_tile(new_op)

    def update_offset_maps(self, op, dep_map):
        if op.op_type == "compute":
            if self in op.sources:
                movement = op.get_src_movement(self.name)
            else:
                assert self in op.dests
                movement = op.get_dest_movement(self.name)
        elif op.op_type == "transfer":
            movement = op.get_src_movement(op.path[0], op.path[1])

    def update_op_accesses(self, cdlt, op, dep_map: Dict[str, str]):
        accesses = self.get_op_accesses(op.op_str)
        for a in accesses:
            a.substitute_offset_symbols(cdlt, dep_map, op.dependencies)

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

    def get_op_accesses(self, op_name: str):
        all_accesses = []
        for dm in self.data_moves:
            if dm.op_name == op_name:
                all_accesses.append(dm)
        if len(all_accesses) == 0:
            raise KeyError(f"Unablet to find movement for operation {op_name}\n"
                           f"Names: {[dm.op_name for dm in self.data_moves]}")
        return all_accesses


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
            access_level = hag.get_node_level(access.dst_node)

            if access_level in level_shapes:
                access.resolve_shape_map(level_shapes[access_level])
            else:
                #TODO: Come back to this code
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
                                  data_moves=deepcopy(self.data_moves))
        op_temp.evaluated_tiling = deepcopy(self.evaluated_tiling)
        return op_temp

    def emit(self, output_type):
        if output_type == "json":
            blob = {"name": self.name,
                    "dtype": str(self.dtype),
                    "shape_symbols": [[k, v] for k, v in self.shape_symbols.items()],
                    "data_path": self.data_path,
                    "tiling": [v for _, v in self.tiling.items()],
                    }
        else:
            raise TypeError(f"Unable to support output type for operand: {output_type}")
        return blob

@dataclass
class IndexedOperandTemplate:
    operand_template: OperandTemplate
    offsets: List[Union[int, str, Basic]]

    @property
    def data_moves(self):
        return self.operand_template.data_moves

    def add_transfer_access(self, path, op_name, sizes):
        return self.operand_template.add_transfer_access(path, op_name, sizes, self.offsets)

    def add_compute_access(self, target, op_name, operand_type):
        return self.operand_template.add_compute_access(target, op_name, operand_type, self.offsets)

@dataclass(frozen=True)
class Offset:
    dim: int
    loop_id: int
    stride: int
    dim_size: int
    offset: int





