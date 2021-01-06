from .base import Operation
from dataclasses import field, dataclass
from collections import deque, defaultdict

class Loop(Operation):

    loop_level_names = defaultdict(list)

    def __init__(self, name, start, end=None, stride=1, template=None):
        if end is not None:
            self._start = start
            self._end = end
        else:
            self._start = 0
            self._end = start
        assert name not in Loop.loop_level_names[Loop.loop_ctxt_level]
        Loop.loop_level_names[Loop.loop_ctxt_level].append(name)
        self._stride = stride
        super(Loop, self).__init__(name, "loop", name, Loop.loop_ctxt_level, instruction_template=template)

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


class LoopArithmetic(Loop):
    def __init__(self, l1, l2, bin_op):
        self._l1 = l1
        self._l2 = l2
        self._bin_op = bin_op

        name = f"{self._l1.name}{self._l2.name}"
        # name, start, end = None, stride = 1, template = None
        super(LoopArithmetic, self).__init__(name, )

    @property
    def l1(self):
        return self._l1

    @property
    def l2(self):
        return self._l2


