from typing import List, Dict, Union

from .operand_template import OperandTemplate, IndexOperandTemplate
from .micro_template import MicroTemplate


class ComputeTemplate(MicroTemplate):
    PARAM_KEYS = ['op_name', 'sources', 'target']
    USE_DUMMY_STRING = [True, False, False, False]
    def __init__(self, op_name: str,
                 sources: List[OperandTemplate],
                 compute_target: str,
                 add_codelet=True,
                 **kwargs
                 ):
        param_map = {}
        param_map['op_name'] = op_name
        param_map['target'] = compute_target
        param_map['sources'] = sources
        super(ComputeTemplate, self).__init__("compute", {**param_map, **kwargs}, add_codelet=add_codelet)
        for s in sources:
            if isinstance(s, OperandTemplate):
                s.add_read(self.op_str)
            elif isinstance(s,  IndexOperandTemplate):
                s.operand.add_read(self.op_str)
            else:
                raise RuntimeError

    def __str__(self):
        return f"{self.output_operand.name} = {self.op_str}('{self.op_name}'; ARGS={self.operand_names}; TGT: {self.target}"

    @property
    def operand_names(self):
        names = []
        for o in self.sources:
            if isinstance(o, OperandTemplate):
                names.append(o.name)
            elif isinstance(o, IndexOperandTemplate):
                names.append(o.operand.name)
            else:
                names.append(str(o))
        return tuple(names)


    @property
    def op_name(self):
        return self.param_map['op_name']

    @property
    def sources(self):
        return self.param_map['sources']

    @sources.setter
    def sources(self, sources):
        self.param_map['sources'] = sources

    @property
    def target(self):
        return self.param_map['target']

    @target.setter
    def target(self, target):
        self.param_map['target'] = target

    @property
    def positional_args(self):
        return ComputeTemplate.PARAM_KEYS

    @property
    def arg_dummy_strings(self):
        return ComputeTemplate.USE_DUMMY_STRING

    def is_target_set(self):
        return self.target is not None



