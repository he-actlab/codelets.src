from dataclasses import dataclass, field
from typing import Union, List
from types import FunctionType
from . import Operation, OperandTemplate
from functools import partial
from cvxpy import Variable, Parameter

@dataclass
class IndexParam:
    operation: str
    val: Union[str, Operation, int]
    loop_id: int
    stride: int

@dataclass
class DataIndex:
    base_iter: Operation
    dim: int = field(default=-1)
    offsets: List[IndexParam] = field(default_factory=list)
    strides: List[IndexParam] = field(default_factory=list)
    operations: List[IndexParam] = field(default_factory=list)
    _domains: List[Operation] = field(default_factory=list)

    def add_operation(self, op: Union[None,str], val: Union[str, Operation, int]):
        if isinstance(val, Operation) and val.op_type == "loop":
            off = IndexParam(op, val, val.loop_id, val.stride)
            self._domains.append(val)
            self.operations.append(off)
        elif isinstance(val, DataIndex):
            self._domains += val.domains
            self.operations += val.operations
        else:
            assert isinstance(val, (int, str))
            self.operations.append(IndexParam(op, val, self.base_iter.loop_id, self.base_iter.stride))

    @property
    def domains(self):
        return [self.base_iter] + self._domains

    def __str__(self):
        return f"{self.base_iter.op_str}"


# class IndexExpr(object):
#     def __init__(self, domain_map, extent, offset=0, scale=1):
#         self._extent = extent
#         self._offset = [offset]
#         self._scale = [scale]
#         self._domain_map = domain_map
#
#     @property
#     def offset(self):
#         return self._offset
#
#     def scale(self):
#         return self._scale
#
#
#     def combine_scalar_scale(self, val: Union[str, int]):
#
#         self._scale.append(val)
#
#     def combine_scalar_offset(self, val: Union[str, int]):
#         self._offset.append(val)
#
#     def combine_loop_offset(self, val: Operation):
#         if val.op_str in self._domain_map:
#             self._domain_map[val.op_str].append(val.op_str)
#         else:
#             self._domain_map[val.op_str] = []
#
#
#     def combine_offset(self, val):
#         if isinstance(val, (str, int)):
#             self.combine_scalar_offset(val)
#         else:
#             assert isinstance(val, Operation)
#             self.combine_loop_offset(val)
#
# class IndexExprList(object):
#     def __init__(self, expr_list: List[IndexExpr]):
#         self._expr_list = expr_list
#
#     def merge_index(self, idx, pos, op):
#         if op == "*":
#             assert isinstance(idx, (str, int))
#             self._expr_list[pos].combine_scalar_scale(idx)
#         else:
#             assert op == "+"
#             self._expr_list[pos].combine_offset(idx)
