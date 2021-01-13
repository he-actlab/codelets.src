from .base_op import Operation
from typing import List, Union, Tuple
from codelets.adl.operand import OperandTemplate
from dataclasses import field, dataclass, replace
import numpy as np
from collections import deque, defaultdict

@dataclass
class TransferInfo:
    src: str
    dst: str
    src_offset: List[Union[Operation, str]]
    dst_offset: List[Union[Operation, str]]
    size: Union[str, int]

class Transfer(Operation):

    def __init__(self, operand: OperandTemplate, path: List[str], offsets: List[List[Operation]], sizes,
                 add_codelet=True,
                 **kwargs):
        # TODO: Add type checking for offset lists
        self._path = path
        self._offsets = []
        for o in offsets:
            if isinstance(o, (list, tuple)):
                assert len(o) == len(operand.shape_list)
                off = o
            else:
                # TODO: Add check for numpy type as well
                assert isinstance(o, int)
                off = [0] * len(operand.shape_list)
                off[-1] = o
            self._offsets.append(off)

        self._sizes = sizes
        assert len(path) >= 2
        assert len(path) == len(offsets)
        assert len(path) == (len(sizes) + 1)
        self._operand = operand
        self._transfers = {}

        for i, (pt, ofs) in enumerate(zip(path[1:], offsets[1:])):
            self._transfers[(path[i], pt)] = TransferInfo(path[i], pt, offsets[i], ofs, sizes[i])

        # self._src_offset = src_offset if isinstance(src_offset, list) else [src_offset]
        # self._dst_offset = dst_offset if isinstance(dst_offset, list) else [dst_offset]
        # self._source = source
        # self._dest = dest
        # for s in self.src_offset:
        #     if isinstance(s, str):
        #         req_params.append(s)
        #
        # for d in self.dst_offset:
        #     if isinstance(d, str):
        #         req_params.append(d)

        req_params = []
        for o in self._offsets:
            for l in o:
                if isinstance(l, str):
                    req_params.append(l)

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


    # @property
    # def source(self):
    #     return self._source
    #
    # @property
    # def dest(self):
    #     return self._dest
    #
    # @property
    # def src_offset(self):
    #     return self._src_offset
    #
    # @property
    # def dst_offset(self):
    #     return self._dst_offset

    @property
    def sizes(self) -> List[Union[str, int]]:
        return self._sizes

    @property
    def operand(self):
        return self._operand

    @property
    def transfers(self):
        return self._transfers


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
        # src_offset = []
        # dest_offset = []
        # for s in self.src_offset:
        #     if isinstance(s, Operation):
        #         src_offset.append(cdlt.global_op_map[s.global_op_id])
        #     else:
        #         src_offset.append(s)
        # for d in self.dst_offset:
        #     if isinstance(d, Operation):
        #         dest_offset.append(cdlt.global_op_map[d.global_op_id])
        #     else:
        #         dest_offset.append(d)

        return (replace(self.operand), self.path, offsets, self.sizes.copy())

    def op_type_params(self):
        op_params = []
        for i, offset in enumerate(self.offsets):
            offset_str = ",".join([o.op_str if isinstance(o, Operation) else f"{o}" for o in offset])
            op_params.append(f"{self.path[i]}[{offset_str}]")

        # src_offset_str = ",".join([so.op_str if isinstance(so, Operation) else f"{so}" for so in self.src_offset])
        # dst_offset_str = ",".join([so.op_str if isinstance(so, Operation) else f"{so}" for so in self.dst_offset])
        # op_params.append(f"SRC: {self.source}[{src_offset_str}]")
        # op_params.append(f"DST: {self.dest}[{dst_offset_str}]")
        return op_params

    def compute_loop_idx_stride(self, loop, offsets):
        dim_size = 1
        lidx = 0
        for dim, idx in zip(reversed(self.operand.shape), reversed(offsets)):
            lidx += idx.stride*dim_size
            if loop == idx:
                break
            dim_size *= dim
        return lidx


    def get_offset_stride(self, dim, node_name):
        offset_idx = self.path.index(node_name)
        offsets = self.offsets[offset_idx]
        loop = offsets[dim]
        return self.compute_loop_idx_stride(loop, offsets)

    # def get_src_offset_stride(self, dim):
    #     offsets = self.src_offset
    #     loop = offsets[dim]
    #     return self.compute_loop_idx_stride(loop, offsets)
    #
    # def get_dst_offset_stride(self, dim):
    #     offsets = self.dst_offset
    #     loop = offsets[dim]
    #     return self.compute_loop_idx_stride(loop, offsets)


