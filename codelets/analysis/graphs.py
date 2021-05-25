from codelets.micro_templates import TransferTemplate, ComputeTemplate, LoopTemplate
from collections import deque, defaultdict
import networkx as nx
from codelets.codelet_template import CodeletTemplate


def form_cdlt_block_map(cdlt: CodeletTemplate):
    block_map = {cdlt.cdlt_uid: []}
    blocks = deque([cdlt.cdlt_uid])
    levels = deque([0])
    for o in cdlt.ops:
        if isinstance(o, LoopTemplate):
            block_map[blocks[-1]].append(o.op_str)
            levels.append(o.loop_level + 1)
            blocks.append(o.op_str)
            block_map[o.op_str] = []
        elif levels[-1] > o.loop_level:
            while levels[-1] > o.loop_level:
                _ = levels.pop()
                _ = blocks.pop()
            block_map[blocks[-1]].append(o.op_str)
        else:
            block_map[blocks[-1]].append(o.op_str)
    return block_map

def join_hag_nodes(node1, node2):
    pass