from .base_op import Operation
from functools import partial
from dataclasses import field, dataclass
from collections import deque, defaultdict

class Loop(Operation):

    loop_ids = 0

    def __init__(self, start,
                 end=None,
                 stride=1,
                 offset=0,
                 loop_op_params=None,
                 add_codelet=True,
                 **kwargs
                 ):
        if end is not None:
            self._start = start
            self._end = end
        else:
            self._start = 0
            self._end = start
        self._stride = stride
        self._offset = offset

        req_params = []
        if loop_op_params:
            req_params += loop_op_params
        if isinstance(self.start, str):
            req_params.append(self.start)

        if isinstance(self.end, str):
            req_params.append(self.end)

        if isinstance(stride, str):
            req_params.append(stride)

        super(Loop, self).__init__("loop", req_params,
                                   add_codelet=add_codelet,
                                   **kwargs)

    def op_type_args_copy(self, _):
        return (self.start, self.end, self.stride, self.offset)

    def __enter__(self):
        Operation.loop_ctxt_level += 1
        Operation.loop_stack.append(self.loop_id)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        Operation.loop_ctxt_level -= 1
        Operation.loop_stack.pop()

    def set_loop_level(self, level):
        pass

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def stride(self):
        return self._stride

    @property
    def iter_count(self) -> int:
        return (self.end - self.start)//self.stride

    @property
    def loop_domains(self):
        return [self.loop_id]

    @property
    def offset(self):
        return self._offset

    def __add__(self, other):
        return LoopArithmetic(self, other, '+')

    def __radd__(self, other):
        return LoopArithmetic(other, self, '+')

    def __sub__(self, other):
        return LoopArithmetic(self, other, '-')

    def __rsub__(self, other):
        return LoopArithmetic(other, self, '-')

    def __mul__(self, other):
        return LoopArithmetic(self, other, '*')

    def __rmul__(self, other):
        return LoopArithmetic(other, self, '*')

    def __div__(self, other):
        return LoopArithmetic(self, other, '/')

    def __rdiv__(self, other):
        return LoopArithmetic(other, self, '/')

    def __truediv__(self, other):
        return LoopArithmetic(self, other, '/')

    def __rtruediv__(self, other):
        return LoopArithmetic(other, self, '/')

    def __floordiv__(self, other):
        return LoopArithmetic(self, other, '//')

    def __rfloordiv__(self, other):
        return LoopArithmetic(other, self, '//')

    def __mod__(self, other):
        return LoopArithmetic(self, other, '%')

    def __rmod__(self, other):
        return LoopArithmetic(other, self, '%')

    def __lshift__(self, other):
        return LoopArithmetic(self, other, '<<')

    def __rlshift__(self, other):
        return LoopArithmetic(other, self, '<<')

    def __rshift__(self, other):
        return LoopArithmetic(self, other, '>>')

    def __rrshift__(self, other):
        return LoopArithmetic(other, self, '>>')

    def __and__(self, other):
        return LoopArithmetic(self, other, '&')

    def __rand__(self, other):
        return LoopArithmetic(other, self, '&')

    def __or__(self, other):
        return LoopArithmetic(self, other, '|')

    def __ror__(self, other):
        return LoopArithmetic(other, self, '|')

    def __xor__(self, other):
        return LoopArithmetic(self, other, '^')

    def __rxor__(self, other):
        return LoopArithmetic(other, self, '^')

    def op_type_params(self):
        op_params = [f"LO: {self.start}", f"HI: {self.end}", f"stride: {self.stride}"]
        return op_params


class LoopArithmetic(Loop):

    # def __init__(self, start,
    #              end=None,
    #              stride=1,
    #              offset=0,
    #              loop_op_params=None,
    #              add_codelet=True,
    #              **kwargs
    #              ):

    def __init__(self, l1, l2, bin_op, **kwargs):

        self._l1 = l1
        self._l2 = l2
        self._bin_op = bin_op
        loop_op_params = []

        if isinstance(l1, str):
            loop_op_params.append(l1)

        if isinstance(l2, str):
            loop_op_params.append(l2)

        super(LoopArithmetic, self).__init__(None, loop_op_params=loop_op_params, **kwargs)

    def op_type_args_copy(self, cdlt):
        l1 = cdlt.global_op_map[self.l1.global_op_id] if isinstance(self.l1, Operation) else self.l1
        l2 = cdlt.global_op_map[self.l2.global_op_id] if isinstance(self.l2, Operation) else self.l2
        return (l2, l1, self.bin_op)

    @property
    def l1(self):
        return self._l1

    @property
    def l2(self):
        return self._l2

    @property
    def bin_op(self):
        return self._bin_op

    @property
    def loop_domains(self):
        l1_d = []
        l2_d = []

        if isinstance(self.l1, (LoopArithmetic, Loop)):
            l1_d += self.l1.loop_domains

        if isinstance(self.l2, (LoopArithmetic, Loop)):
            l2_d += self.l2.loop_domains

        return l1_d + l2_d

    def op_type_params(self):
        op_params = []

        if isinstance(self.l1, (LoopArithmetic, Loop)):
            op_params.append(f"L1: {self.l1.op_str}")
        else:
            op_params.append(f"L1: {self.l1}")

        if isinstance(self.l2, (LoopArithmetic, Loop)):
            op_params.append(f"L2: {self.l2.op_str}")
        else:
            op_params.append(f"L2: {self.l1}")
        op_params.append(f"BIN_OP: {self.bin_op}")

        return op_params
