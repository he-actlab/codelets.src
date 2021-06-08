from typing import List, Dict, Union

from functools import singledispatch
from .dummy_op import DummyOp, DummyParam
from .operand_template import OperandTemplate, IndexOperandTemplate, Offset
from .micro_template import MicroTemplate


class TransferTemplate(MicroTemplate):
    PARAM_KEYS = ['src_op', 'src_offset', 'dst_op', 'dst_offset', 'size']
    USE_DUMMY_STRING = [False, True, True, False, True, True]

    def __init__(self, source: Union[OperandTemplate, IndexOperandTemplate],
                 destination: Union[OperandTemplate, IndexOperandTemplate],
                 size,
                 add_codelet=True,
                 **kwargs
                 ):
        assert isinstance(source, (OperandTemplate, IndexOperandTemplate))
        if destination is not None:
            assert isinstance(destination, (str, DummyOp, OperandTemplate, IndexOperandTemplate))
        param_map = {}
        if isinstance(source, IndexOperandTemplate):
            src_offset = source.offsets
            src_op = source.operand
        else:
            src_offset = [Offset(0, [], [], [])]
            src_op = source

        if isinstance(destination, IndexOperandTemplate):
            dst_offset = destination.offsets
            dst_op = destination.operand
        else:
            dst_offset = [Offset(0, [], [], [])]
            dst_op = destination

        param_map['src_op'] = src_op
        param_map['src_offset'] = src_offset
        param_map['size'] = size
        param_map['dst_op'] = dst_op
        param_map['dst_offset'] = dst_offset

        super(TransferTemplate, self).__init__("transfer", {**param_map, **kwargs}, add_codelet=add_codelet)
        assert isinstance(src_op, OperandTemplate)
        assert isinstance(dst_op, OperandTemplate)
        self.src_op.add_read(self.op_str)
        self.dst_op.add_write(self.op_str)
        self.set_output_operand(self.dst_op)

    def __str__(self):
        src_str = f"SRC:('{self.src_location}', {self.src_op.name}{self.src_offset_str})"
        dst_str = f"DST:('{self.dst_location}', {self.dst_op.name}{self.dst_offset_str}) "
        return f"{self.output_operand.name} = {self.op_str}({src_str} --> {dst_str})"

    @property
    def positional_args(self):
        return TransferTemplate.PARAM_KEYS

    @property
    def arg_dummy_strings(self):
        return TransferTemplate.USE_DUMMY_STRING

    @property
    def src_op(self) -> OperandTemplate:
        return self.param_map['src_op']

    @src_op.setter
    def src_op(self, op) -> OperandTemplate:
        self.param_map['src_op'] = op

    @property
    def dst_op(self) -> OperandTemplate:
        return self.param_map['dst_op']

    @dst_op.setter
    def dst_op(self, op) -> OperandTemplate:
        self.param_map['dst_op'] = op

    @property
    def src_offset(self) -> OperandTemplate:
        return self.param_map['src_offset']

    @property
    def src_offset_str(self) -> List[str]:
        return [str(so) for so in self.src_offset]

    @property
    def dst_offset(self) -> OperandTemplate:
        return self.param_map['dst_offset']

    @property
    def dst_offset_str(self) -> List[str]:
        return [str(do) for do in self.dst_offset]

    @property
    def src_location(self):
        return self.src_op.location

    @property
    def dst_location(self):
        return self.dst_op.location
