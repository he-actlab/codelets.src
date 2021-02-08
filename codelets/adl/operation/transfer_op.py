from .base_op import Operation
from typing import List, Union, Dict, Tuple, Callable
from codelets.adl.operation.operand import OperandTemplate, Offset, IndexedOperandTemplate
from . import Loop
import numpy as np
from itertools import chain
from numbers import Integral
from codelets.adl import util
from dataclasses import dataclass, replace, field
from copy import copy, deepcopy
from sympy import Basic, symbols, Idx, IndexedBase, Integer
from . import size_from_offsets, get_transfer_dim_sizes

OffsetType = Union[str, Integral, Basic]

@dataclass
class TransferInfo:
    src: str
    dst: str
    _src_offset: List[OffsetType]
    _dst_offset: List[OffsetType]
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

    def copy(self, cdlt):
        src_off_copy = []
        for op in self._src_offset:
            if isinstance(op, Operation):
                src_off_copy.append(cdlt.global_op_map[op.global_op_id])
            else:
                src_off_copy.append(op)
        dst_off_copy = []
        for op in self._dst_offset:
            if isinstance(op, Operation):
                dst_off_copy.append(cdlt.global_op_map[op.global_op_id])
            else:
                dst_off_copy.append(op)
        return replace(self, _src_offset=src_off_copy,
                       _dst_offset=dst_off_copy,
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

    def compute_src_size(self, cdlt):
        self.size = size_from_offsets(cdlt, self._src_offset)
        self._src_dim_sizes = self.size
        self.set_src_domain_offsets(cdlt)

    def compute_dst_size(self, cdlt):
        self.size = size_from_offsets(cdlt, self._dst_offset)
        self._dst_dim_sizes = self.size
        self.set_dst_domain_offsets(cdlt)

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


# TODO: Check to make sure there are edges for the entire path
class Transfer(Operation):

    def __init__(self, operand: Union[OperandTemplate, IndexedOperandTemplate], path: List[str],
                 sizes=None,
                 add_codelet=True,
                 **kwargs):

        # TODO: Add type checking for offset lists
        # Set path, make sure there is at least 2 locations to transfer to/from
        assert len(path) >= 2
        self._path = path


        sizes = sizes or [[]]*(len(path)-1)

        # For each pair of nodes in the path, there needs to be a size
        assert len(path) == (len(sizes) + 1)

        self._transfers = {}
        self._access_indices = []
        req_params = []
        dependencies = []

        for s in sizes:
            if isinstance(s, str):
                req_params.append(s)

        super(Transfer, self).__init__('transfer', req_params,
                                       add_codelet=add_codelet,
                                       dependencies=dependencies,
                                       **kwargs)
        operand.test_add_transfer_access(path, self.op_str, sizes)
        if isinstance(operand, OperandTemplate):
            operand = operand.add_data_access([], self.op_str, "read", sizes[0], access_node=path[0])
        else:
            operand = operand.operand_template.add_data_access(self.offsets, self.op_str, "read", sizes[0], access_node=path[0])

        self._access_indices.append(len(operand.data_moves) - 1)

        self._operand = operand
        self._dependencies += [d for d in self._operand.dependencies if d not in self.dependencies]
        self._required_params += [r for r in self.operand.required_params if r not in self.required_params]

        for i, pt in enumerate(path[1:]):
            xfer_key = (path[i], pt)
            if i > 0:
                self.operand.add_data_access([0] * len(self.operand.shape_list), self.op_str, "read", sizes[i], access_node=path[i])
                self._access_indices.append(len(operand.data_moves) - 1)

            self.operand.add_data_access([0]*len(self.operand.shape_list), self.op_str, "write", sizes[i], access_node=pt, path_key=(xfer_key))
            self._access_indices.append(len(operand.data_moves) - 1)

            src_off = list(chain(self.operand.data_moves[-2].offset_map.values()))
            dst_off = list(chain(self.operand.data_moves[-1].offset_map.values()))
            self._transfers[xfer_key] = TransferInfo(path[i], pt, src_off, dst_off, sizes[i])



    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def offsets(self):
        nodes = []
        offsets = []
        for k, v in self.transfers.items():

            if k[0] not in nodes:
                offsets.append(v._src_offset)
                nodes.append(k[0])

            if k[1] not in nodes:
                offsets.append(v._dst_offset)
                nodes.append(k[1])
        return offsets

    @property
    def sizes(self) -> List[List[Union[str, Integral]]]:
        return [v.size for _, v in self.transfers.items()]

    @property
    def data_transfer_sizes(self) -> List[Union[str, Integral]]:
        return [np.prod(s) for s in self.sizes]

    @property
    def operand(self) -> OperandTemplate:
        return self._operand

    @property
    def transfers(self) -> Dict[Tuple[str, str], TransferInfo]:
        return self._transfers

    @property
    def access_indices(self):
        return self._access_indices

    @transfers.setter
    def transfers(self, transfers):
        self._transfers = transfers

    def initialize_offsets(self, offset):
        if isinstance(offset, (list, tuple)):
            assert len(offset) == len(self.operand.shape_list)
            off = offset
        else:
            # TODO: Add check for numpy type as well
            assert isinstance(offset, int)
            off = [0] * len(self.operand.shape_list)
            off[-1] = offset
        arr_idx_symbols = []
        for dim, idx in enumerate(off):
            if isinstance(idx, Loop):
                arr_idx_symbol = idx.param_symbols[idx.op_str]
                self.dependencies.append(idx.op_str)
            elif isinstance(idx, Basic):
                arr_idx_symbol = idx
                loop_idx = idx.atoms(Idx)
                self.dependencies += [str(l) for l in list(loop_idx) if str(l) not in self.dependencies]
            elif isinstance(idx, str):
                self.required_params.append(idx)
                arr_idx_symbol = symbols(idx, integer=True)
            elif isinstance(idx, int):
                arr_idx_symbol = idx
            else:
                raise TypeError(f"Invalid type for loop index: {idx}, type: {type(idx)}")

            arr_idx_symbols.append(arr_idx_symbol)
        return arr_idx_symbols

    def op_type_params(self):
        op_params = []
        for i, off in enumerate(self.offsets):
                offset_str = ",".join([o.op_str if isinstance(o, Operation) else f"{o}" for o in off])
                op_params.append(f"{self.path[i]}[{offset_str}]")

        return op_params


    def evaluate_parameters(self, node, hag, cdlt):
        pass
        # print(self.required_params)
        # for a_idx in self.access_indices:
        #     access = self.operand.data_moves[a_idx]

        # for path_key, tinfo in self.transfers.items():
        #     src_shape, dst_shape = get_transfer_dim_sizes(self.operand, path_key)
        #     tinfo._src_dim_sizes = src_shape
        #     tinfo._dst_dim_sizes = dst_shape
        #     tinfo.set_src_domain_offsets(cdlt)
        #     tinfo.set_dst_domain_offsets(cdlt)
        #     self.operand.tiling[path_key] = dst_shape
        #
        #     new_sizes = []
        #     for s in tinfo.size:
        #         if isinstance(s, str):
        #             size = cdlt.required_params[s].value
        #         elif isinstance(s, Integral):
        #             size = s
        #         else:
        #             raise TypeError(f"Invalid type for size: {s}\n"
        #                             f"Type: {type(s)}")
        #         new_sizes.append(size)
        #
        #     if len(new_sizes) == 0:
        #         tinfo.size = []
        #     else:
        #         tinfo.size = new_sizes

    def emit(self, output_type):
        # TODO: Add template
        if output_type == "operations":
            op_str = f"{self.op_str}: OPERAND: {self.operand.name}[{'->'.join(self.path)}], SIZES: {self.sizes}," \
                     f"OFFSETS: {self.offsets[1]}"
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


    def copy(self, cdlt, operand=None, path=None, access_indices=None, transfers=None, **kwargs):
        obj = super(Transfer, self).copy(cdlt, **kwargs)
        obj._operand = operand or self.operand.copy()
        obj._path = path or self.path.copy()
        obj._access_indices = access_indices or self._access_indices.copy()
        obj._transfers = transfers or {k: v.copy(cdlt) for k, v in self.transfers.items()}
        return obj



