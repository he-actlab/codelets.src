from dataclasses import dataclass, field
from .dummy_op import DummyOp
from codelets.common.flex_param import FlexParam
from codelets.micro_templates import TEMPLATE_CLASS_ARG_MAP


@dataclass
class PlaceholderStructure:
    name: str
    template_type: str = field(default=None)

    def __post_init__(self):
        self.template_type = self.__class__.__name__

    def __getattr__(self, name):
        # TODO: Add back a check here which verifies the attribute is a part of the class
        # assert name in CLASS_ATTR[self.template_type] or self.template_type == "NodePlaceholder"
        args = TEMPLATE_CLASS_ARG_MAP[self.template_type]
        fp = FlexParam(f"{self.template_type}_{self.name}_dummy", args, f"{args[0]}.{name}")
        return DummyOp([self.template_type], fp)

@dataclass
class HAGPlaceholder(PlaceholderStructure):
    pass

@dataclass
class NodePlaceholder(PlaceholderStructure):
    pass