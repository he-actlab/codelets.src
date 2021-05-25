from codelets.micro_templates import TransferTemplate, ComputeTemplate, LoopTemplate
from .graphs import form_cdlt_block_map
import networkx as nx
from codelets.codelet_template import CodeletTemplate

RULES = {
    ""
}

def create_pattern(cdlt: CodeletTemplate):
    form_cdlt_block_map(cdlt)
    # ssa_list = []
    # cfg = nx.MultiDiGraph()
    # dfg = nx.MultiDiGraph()
    # cfg.add_node(cdlt.cdlt_uid)
    # for o in cdlt.ops:
    #     if isinstance(o, ComputeTemplate):
    #         res = o.output_operand.name
    #         stmt = f"{o.op_name}" + ", ".join([s.name for s in o.sources])
    #         ssa_list.append(f"{res}")