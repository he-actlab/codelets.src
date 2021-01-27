from .base_op import Operation
from typing import List, Union, Dict, Tuple
from codelets.adl.operation.operand import OperandTemplate
from . import Loop
import numpy as np
from numbers import Integral
from codelets.adl import util
from dataclasses import dataclass, replace, field
from copy import copy, deepcopy
from sympy import Basic, symbols, Idx, IndexedBase, Integer

@dataclass(frozen=True)
class Offset:
    dim: int
    loop_id: int
    stride: int
    dim_size: int
    offset: int

@dataclass
class TransferInfo:
    src: str
    dst: str
    _src_offset: List[Basic]
    _dst_offset: List[Basic]
    size: List[Union[str, Integral]]
    _src_dim_sizes: List[Integral] = field(default_factory=list)
    _dst_dim_sizes: List[Integral] = field(default_factory=list)
    src_domain_offsets: List[Offset] = field(default_factory=list)
    dst_domain_offsets: List[Offset] = field(default_factory=list)

    def __copy__(self):
        return replace(self, _src_offset=deepcopy(self._src_offset),
                       _dst_offset=deepcopy(self._dst_offset),
                       _src_dim_sizes=self._src_dim_sizes.copy(),
                       _dst_dim_sizes=self._dst_dim_sizes.copy(),
                       src_domain_offsets=[replace(sdo) for sdo in self.src_domain_offsets],
                       dst_domain_offsets=[replace(ddo) for ddo in self.dst_domain_offsets]
                       )
    @property
    def src_offset(self):
        return self.src_domain_offsets

    @property
    def dst_offset(self):
        return self.dst_domain_offsets


    def compute_domain_offsets(self, cdlt, dim_sizes, offsets):
        if len(dim_sizes) > 0:
            d_offsets = []
            dom_offsets = []
            for pos, dim_size in enumerate(dim_sizes):
                base_offset = int(np.prod(dim_sizes[pos + 1:]))
                expr = offsets[pos]
                if isinstance(expr, Basic):
                    for f_sym in list(expr.free_symbols):
                        if str(f_sym) in cdlt.required_params:
                            expr = expr.subs(f_sym, cdlt.required_params[str(f_sym)].value)

                    indices = list(expr.atoms(Idx))
                    offset, coeffs, nonlin = util.split(expr, indices)

                    for i, idx in enumerate(indices):
                        coeff = coeffs[i]

                        coeff *= base_offset
                        if not isinstance(coeff, (Integer, Integral)):
                            raise TypeError(f"Unable to compute domain offsets because coefficient is not an intenger:"
                                            f"Coeff: {coeff}\tType: {type(coeff)}")
                        d_offsets.append(int(coeff))
                        o_info = Offset(pos, cdlt.op_map[str(idx)].loop_id, int(coeff), dim_size, offset)
                        dom_offsets.append(o_info)
                else:
                    d_offsets.append(base_offset)
                    o_info = Offset(pos, -1, base_offset, dim_size, expr)
                    dom_offsets.append(o_info)
        else:
            dom_offsets = []

        return dom_offsets

    def set_src_domain_offsets(self, cdlt):
        self.src_domain_offsets = self.compute_domain_offsets(cdlt, self._src_dim_sizes, self._src_offset)

    def set_dst_domain_offsets(self, cdlt):
        self.dst_domain_offsets = self.compute_domain_offsets(cdlt, self._dst_dim_sizes, self._dst_offset)

    def compute_size_from_splits(self, symbol_splits: Dict[str, int]):

        pass


# TODO: Check to make sure there are edges for the entire path
class Transfer(Operation):

    def __init__(self, operand: OperandTemplate, path: List[str], offsets: List[List[Loop]],
                 sizes=None,
                 add_codelet=True,
                 **kwargs):

        # TODO: Add type checking for offset lists
        self._path = path
        self._offsets = []
        req_params = []
        dependencies = []
        for i, o in enumerate(offsets):
            if isinstance(o, (list, tuple)):
                assert len(o) == len(operand.shape_list)
                off = o
            else:
                # TODO: Add check for numpy type as well
                assert isinstance(o, int)
                off = [0] * len(operand.shape_list)
                off[-1] = o
            arr_idx_symbols = []
            for dim, idx in enumerate(off):
                if isinstance(idx, Loop):
                    arr_idx_symbol = idx.param_symbols[idx.op_str]
                    dependencies.append(idx.op_str)
                elif isinstance(idx, Basic):
                    arr_idx_symbol = idx
                    loop_idx = idx.atoms(Idx)
                    dependencies += [str(l) for l in list(loop_idx) if str(l) not in dependencies]
                elif isinstance(idx, str):
                    req_params.append(idx)
                    arr_idx_symbol = symbols(idx, integer=True)
                elif isinstance(idx, int):
                    arr_idx_symbol = idx
                else:
                    raise TypeError(f"Invalid type for loop index: {idx}, type: {type(idx)}")

                arr_idx_symbols.append(arr_idx_symbol)

            self._offsets.append(arr_idx_symbols)
        assert len(path) >= 2
        assert len(path) == len(offsets)
        self._sizes = sizes or [[]]*(len(path)-1)
        assert all([not isinstance(s, Loop) for s in self._sizes])
        assert len(path) == (len(self._sizes) + 1)
        self._operand = operand
        self._transfers = {}
        if len(self.operand.sorted_tile_keys) > 0 and path[0] != self.operand.sorted_tile_keys[-1]:
            raise RuntimeError(f"Invalid transfer operation for operand {self.operand.name}:\n"
                               f"{path[0]} is the first architecture node, but the operand is currently stored in "
                               f"{self.operand.sorted_tile_keys[-1]}.")
        for i, (pt, ofs) in enumerate(zip(path[1:], self.offsets[1:])):
            xfer_key = (path[i], pt)
            self._transfers[xfer_key] = TransferInfo(path[i], pt, self.offsets[i], ofs, self._sizes[i])


        for s in self._sizes:
            if isinstance(s, str):
                req_params.append(s)
        dependencies += [d for d in self._operand.dependencies if d not in dependencies]

        super(Transfer, self).__init__('transfer', req_params,
                                       add_codelet=add_codelet,
                                       dependencies=dependencies,
                                       **kwargs)

        for key, tinfo in self.transfers.items():
            if len(tinfo.size) == len(self._operand.shape_list):
                shape_dict = {s: tinfo.size[idx] for idx, s in enumerate(self._operand.shape_list)}
            elif len(tinfo.size) == 0:
                shape_dict = {}
            else:
                assert len(tinfo.size) == 1
                shape_dict = {s: 1 for idx, s in enumerate(self._operand.shape_list)}
                shape_dict[self._operand.shape_list[-1]] = tinfo.size[0]

            self._operand.add_transfer(self.op_str, key, shape_dict)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def offsets(self):
        return self._offsets

    @offsets.setter
    def offsets(self, offsets):
        self._offsets = offsets

    @property
    def sizes(self) -> List[Union[str, int]]:
        return self._sizes

    @sizes.setter
    def sizes(self, sizes):
        self._sizes = sizes

    @property
    def operand(self) -> OperandTemplate:
        return self._operand

    @property
    def transfers(self) -> Dict[Tuple[str, str],TransferInfo]:
        return self._transfers

    @transfers.setter
    def transfers(self, transfers):
        self._transfers = transfers

    def op_type_params(self):
        op_params = []
        for i, offset in enumerate(self.offsets):
            offset_str = ",".join([o.op_str if isinstance(o, Operation) else f"{o}" for o in offset])
            op_params.append(f"{self.path[i]}[{offset_str}]")

        return op_params

    def get_transfer_dim_sizes(self, path_key, hag):
        src_shape = None
        dst_shape = None

        for i, p in enumerate(self.operand.evaluated_tiling[:-1]):
            # TODO: FIx this to validate against path key
            if p[0] == path_key[0] and self.operand.evaluated_tiling[i+1][0] == path_key[1]:
                src_shape = p[1]
                dst_shape = self.operand.evaluated_tiling[i+1][1]
                break

        # if src_shape is None or not all([isinstance(i, Integral) for i in src_shape]) or len(src_shape) == 0:
        #     raise RuntimeError
        #
        # if dst_shape is None or not all([isinstance(i, Integral) for i in dst_shape]) or len(dst_shape) == 0:
        #     raise RuntimeError

        return src_shape, dst_shape

    def evaluate_parameters(self, node, hag, cdlt):

        for path_key, tinfo in self.transfers.items():
            src_shape, dst_shape = self.get_transfer_dim_sizes(path_key, hag)
            tinfo._src_dim_sizes = src_shape
            tinfo._dst_dim_sizes = dst_shape
            tinfo.set_src_domain_offsets(cdlt)
            tinfo.set_dst_domain_offsets(cdlt)
            self.operand.tiling[path_key] = dst_shape

        all_sizes = []
        for size_list in self.sizes:
            new_sizes = []
            for s in size_list:
                if isinstance(s, str):
                    size = cdlt.required_params[s].value
                elif isinstance(s, Integral):
                    size = s
                else:
                    raise TypeError(f"Invalid type for size: {s}\n"
                                    f"Type: {type(s)}")
                new_sizes.append(size)


            if len(new_sizes) == 0:
                all_sizes.append(None)
            else:
                all_sizes.append(int(np.prod(new_sizes)))

        self._sizes = all_sizes

    def emit(self, output_type):
        # TODO: Add template
        if output_type == "operations":
            op_str = f"{self.op_str}: OPERAND: {self.operand.name}[{'->'.join(self.path)}], SIZES: {self.sizes}"
        elif output_type == "json":
            transfer_info = {}
            for path_key, xfer in self.transfers.items():
                text_key = "->".join(path_key)
                transfer_info[text_key] = {}
                transfer_info[text_key]['size'] = xfer.size
                transfer_info[text_key]['src_address'] = xfer._src_offset
                transfer_info[text_key]['dst_address'] = xfer._dst_offset

            op_str = {"op_type": self.op_type,
                      "op_id": self.global_op_id,
                      "operand": self.operand.name,
                      "transfer_path": self.path,
                      "transfers": transfer_info}
        else:
            op_str = []
            for ft in self.instructions:
                op_str += ft.emit(output_type)
        return op_str


    def copy(self, cdlt):
        obj = super(Transfer, self).copy(cdlt)
        offsets = []
        for offset in self.offsets:
            offset_copy = []
            for op in offset:
                if isinstance(op, Operation):
                    offset_copy.append(cdlt.global_op_map[op.global_op_id])
                else:
                    offset_copy.append(op)
            offsets.append(offset_copy)
        obj._operand = cdlt.get_operand(self.operand.name)
        obj._path = self.path.copy()
        obj._offsets = offsets
        obj._sizes = deepcopy(self.sizes)
        obj._transfers = {k: copy(v) for k, v in self.transfers.items()}
        return obj



