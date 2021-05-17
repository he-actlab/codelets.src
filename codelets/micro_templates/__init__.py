TEMPLATE_CLASS_ARG_MAP = {'NodePlaceholder': ['node'],
                          'HAGPlaceholder': ['hag']
                          }
CLASS_ATTR = {}
# CLASS_ATTR['NodePlaceholder'] = dir

from .dummy_op import DummyOp, DummyParam
from .placeholders import HAGPlaceholder, NodePlaceholder
from .operand_template import OperandTemplate, IndexOperandTemplate
from .micro_template import MicroTemplate
from .control import LoopTemplate
from .compute import ComputeTemplate
from .transfer import TransferTemplate