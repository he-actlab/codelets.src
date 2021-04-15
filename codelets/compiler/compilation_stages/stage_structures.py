from typing import List
from codelets.adl.flex_param import FlexParam
from dataclasses import dataclass, field


@dataclass
class TilingInfo:
    name: str
    dimensions: List[str]
    constraint_str: str
    src_node: str
    dst_node: str
    constraint_fp: FlexParam = field(default=None)

    def __post_init__(self):
        if self.constraint_fp is None:
            self.constraint_fp = FlexParam(self.name, ["size"], self.constraint_str)


