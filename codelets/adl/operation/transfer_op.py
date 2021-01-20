from .base_op import Operation
from typing import List, Union, Dict
from codelets.adl.operation.operand import OperandTemplate
from . import Loop, DataIndex
import numpy as np
from numbers import Integral
from codelets.adl import util
from dataclasses import dataclass, replace, field
from sympy import Basic, symbols, Idx, IndexedBase, Integer

@dataclass
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
    size: Union[str, Integral]
    _src_dim_sizes: List[Integral] = field(default_factory=list)
    _dst_dim_sizes: List[Integral] = field(default_factory=list)
    src_domain_offsets: List[Offset] = field(default_factory=list)
    dst_domain_offsets: List[Offset] = field(default_factory=list)


    @property
    def src_offset(self):
        return self.src_domain_offsets

    @property
    def dst_offset(self):
        return self.dst_domain_offsets

    def compute_domain_offsets(self, cdlt, dim_sizes, offsets):
        assert len(dim_sizes) > 0
        d_offsets = []
        dom_offsets = []
        for pos, dim_size in enumerate(dim_sizes):
            base_offset = int(np.prod(dim_sizes[pos + 1:]))
            expr = offsets[pos]
            if isinstance(expr, Basic):
                for f_sym in list(expr.free_symbols):
                    if str(f_sym) in cdlt.required_params:
                        expr = expr.subs(f_sym, cdlt.required_params[str(f_sym)].value)

                offset_added = False
                indices = list(expr.atoms(Idx))
                offset, coeffs, nonlin = util.split(expr, indices)

                for i, idx in enumerate(indices):
                    coeff = coeffs[i]

                    ### TESTING ####
                    if not offset_added:
                        coeff *= base_offset
                        offset_added = True
                    ################
                    assert isinstance(coeff, (Integer, Integral))
                    d_offsets.append(int(coeff))
                    o_info = Offset(pos, cdlt.op_map[str(idx)].loop_id, int(coeff), dim_size, offset)
                    dom_offsets.append(o_info)
            else:
                d_offsets.append(base_offset)
                o_info = Offset(pos, -1, base_offset, dim_size, expr)
                dom_offsets.append(o_info)

        return dom_offsets

    def set_src_domain_offsets(self, cdlt):
        self.src_domain_offsets = self.compute_domain_offsets(cdlt, self._src_dim_sizes, self._src_offset)

    def set_dst_domain_offsets(self, cdlt):
        self.dst_domain_offsets = self.compute_domain_offsets(cdlt, self._dst_dim_sizes, self._dst_offset)


class Transfer(Operation):

    def __init__(self, operand: OperandTemplate, path: List[str], offsets: List[List[Loop]], sizes,
                 add_codelet=True,
                 **kwargs):
        # TODO: Add type checking for offset lists
        self._path = path
        self._offsets = []
        req_params = []

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
                elif isinstance(idx, Basic):
                    arr_idx_symbol = idx
                elif isinstance(idx, str):
                    req_params.append(idx)
                    arr_idx_symbol = symbols(idx, integer=True)
                elif isinstance(idx, int):
                    arr_idx_symbol = idx
                else:
                    raise TypeError(f"Invalid type for loop index: {idx}, type: {type(idx)}")

                arr_idx_symbols.append(arr_idx_symbol)

            self._offsets.append(arr_idx_symbols)

        self._sizes = sizes
        assert len(path) >= 2
        assert len(path) == len(offsets)
        assert len(path) == (len(sizes) + 1)
        self._operand = operand
        self._transfers = {}

        for i, (pt, ofs) in enumerate(zip(path[1:], self.offsets[1:])):
            xfer_key = (path[i], pt)
            self._transfers[xfer_key] = TransferInfo(path[i], pt, self.offsets[i], ofs, sizes[i])
            if len(sizes[i]) == len(self._operand.shape_list):
                shape_dict = {s: sizes[i][idx] for idx, s in enumerate(self._operand.shape_list)}
            else:
                assert len(sizes[i]) == 1
                shape_dict = {s: 1 for idx, s in enumerate(self._operand.shape_list)}
                shape_dict[self._operand.shape_list[-1]] = sizes[i][0]

            self._operand.add_path_tiling(xfer_key, shape_dict)

        for s in sizes:
            if isinstance(s, str):
                req_params.append(s)

        super(Transfer, self).__init__('transfer', req_params,
                                       add_codelet=add_codelet,
                                       **kwargs)
    @property
    def path(self):
        return self._path

    @property
    def offsets(self):
        return self._offsets

    @property
    def sizes(self) -> List[Union[str, int]]:
        return self._sizes

    @property
    def operand(self):
        return self._operand

    @property
    def transfers(self):
        return self._transfers

    def create_index(self, offsets):
        pass

    def op_type_args_copy(self, cdlt):
        offsets = []
        for offset in self.offsets:
            offset_copy = []
            for op in offset:
                if isinstance(op, Operation):
                    offset_copy.append(cdlt.global_op_map[op.global_op_id])
                else:
                    offset_copy.append(op)
            offsets.append(offset_copy)
        return (cdlt.get_operand(self.operand.name), self.path, offsets, self.sizes.copy())

    def op_type_params(self):
        op_params = []
        for i, offset in enumerate(self.offsets):
            offset_str = ",".join([o.op_str if isinstance(o, Operation) else f"{o}" for o in offset])
            op_params.append(f"{self.path[i]}[{offset_str}]")

        return op_params

    def compute_loop_idx_stride(self, loop, offsets):
        dim_size = 1
        lidx = 0
        final_offset = None
        for dim, idx in zip(reversed(self.operand.shape), reversed(offsets)):
            if isinstance(idx.stride, Operation):
                raise RuntimeError(f"Unable to handle variable stride at the moment")
            if isinstance(idx.offset, Operation):
                lidx += idx.stride*dim_size
                final_offset = idx.offset
            else:
                lidx += idx.stride*dim_size + idx.offset
            if loop == idx:
                break
            dim_size *= dim
        return lidx, final_offset

    def get_transfer_dim_sizes(self, path_key, hag):
        src_shape = None
        dst_shape = None

        for i, p in enumerate(self.operand.evaluated_tiling[:-1]):
            # TODO: FIx this to validate against path key
            if p[0] == path_key[0] and self.operand.evaluated_tiling[i+1][0] == path_key[1]:
                src_shape = p[1]
                dst_shape = self.operand.evaluated_tiling[i+1][1]
                break

        if src_shape is None or not all([isinstance(i, Integral) for i in src_shape]) or len(src_shape) == 0:
            raise RuntimeError


        if dst_shape is None or not all([isinstance(i, Integral) for i in dst_shape]) or len(dst_shape) == 0:
            raise RuntimeError
        return src_shape, dst_shape

    def evaluate_parameters(self, node, hag, cdlt):
        for path_key, tinfo in self.transfers.items():
            src_shape, dst_shape = self.get_transfer_dim_sizes(path_key, hag)
            tinfo._src_dim_sizes = src_shape
            tinfo._dst_dim_sizes = dst_shape
            tinfo.set_src_domain_offsets(cdlt)
            tinfo.set_dst_domain_offsets(cdlt)


    def get_offset_stride(self, dim, node_name):
        offset_idx = self.path.index(node_name)
        offsets = self.offsets[offset_idx]
        loop = offsets[dim]
        return self.compute_loop_idx_stride(loop, offsets)



