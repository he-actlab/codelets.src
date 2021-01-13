import numpy as np
from types import FunctionType
from codelets.graph import Node
from codelets.adl.architecture_graph import ArchitectureGraph
from typing import List, Dict, Union, TYPE_CHECKING
from codelets.adl import Codelet, Instruction
from pygraphviz import AGraph
from collections import namedtuple
from .operation import Operation, Loop, Compute, Configure, Transfer

if TYPE_CHECKING:
    from .compute_node import ComputeNode
    from .storage_node import StorageNode
    from .communication_node import CommunicationNode
Edge = namedtuple('Edge', ['src', 'dst', 'attributes', 'transfer_fn_map'])
OpTemplate = namedtuple('OpTemplate', ['instructions', 'functions'])


class UtilFuncs(object):

    def __init__(self):
        self._funcs = {}
        self._func_def_names = {}

    @property
    def funcs(self):
        return self._funcs

    @property
    def func_def_names(self):
        return self._func_def_names

    def __getattr__(self, item):
        return self.funcs[item]

    def add_fn(self, name, arg_vars: List[str], body):
        arg_str = ", ".join(arg_vars)
        self.func_def_names[name] = f"util_fn{len(list(self.funcs.keys()))}"
        self.funcs[name] = f"def {self.func_def_names[name]}({arg_str}):\n\t" \
                           f"return {body}"

    def get_util_fnc(self, name):
        util_fnc_code = compile(self.funcs[name], "<string>", "exec")
        util_fnc = FunctionType(util_fnc_code.co_consts[0], globals(), self.func_def_names[name])
        return util_fnc

    def run_param_fnc(self, fn_name, *args, **kwargs):
        util_fnc = self.get_util_fnc(fn_name)
        return util_fnc(*args, **kwargs)

class ArchitectureNode(Node):
    """
    Base class for Architecture Node
    Inherited from Node
    """

    def __init__(self, name, index=None):
        super(ArchitectureNode, self).__init__(index=index)
        self._has_parent = None
        self._subgraph = ArchitectureGraph()
        self._name = name
        if self.name:
            self.set_attr("field_name", self.name)
        # type
        self._anode_type = type(self).__name__
        self._all_subgraph_nodes = {}
        self._subgraph_nodes = {}
        self._subgraph_edges = []
        self._in_edges = {}
        self._out_edges = {}

        # primitives
        self._primitives = {}

        # capability_sequence
        self._codelets = {}

        # occupied: [(op_node, primitive, begin_cycle, end_cycle)]
        # later consider changing to custom Heap because this needs to be accessed very frequently
        self._occupied = [] # NOTE state in TABLA compiler...
        self._operation_mappings = {"config": {},
                                    "transfer": {},
                                    "loop": {},
                                    "compute": {}}
        self._util_fns = UtilFuncs()

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exec_value, exec_traceback):
        pass
    
    def __str__(self):
        return f'op {self.index} ({self.get_type()}): \
                 preds={self.get_preds_indices()} ({self._attrs["in_degree"]}), \
                 succs={self.get_succs_indices()} ({self._attrs["out_degree"]})'
    
    # two categories of operation
    # modifying anode type arbitrarily should not be permitted

    @property
    def name(self):
        return self._name

    @property
    def subgraph(self):
        return self._subgraph

    @property
    def primitives(self) -> Dict[str, Instruction]:
        return self._primitives

    @property
    def codelets(self) -> Dict[str, Codelet]:
        return self._codelets

    @property
    def util_fns(self):
        return self._util_fns

    @property
    def operation_mappings(self):
        return self._operation_mappings

    @property
    def all_codelet_names(self) -> List[str]:
        names = [] + list(self.codelets.keys())
        for n in self.get_subgraph_nodes():
            names += n.all_codelet_names
        return names

    @property
    def node_type(self):
        raise NotImplementedError

    def networkx_visualize(self, filename):
        dgraph = AGraph(strict=False, directed=True)

        for n in self.subgraph.get_nodes():
            self.subgraph._add_nx_subgraph(dgraph, n)

        dgraph.add_edges_from(self.subgraph.get_viz_edge_list())

        dgraph.layout("dot")
        dgraph.draw(f"{filename}.pdf", format="pdf")
    #

    def add_util_fn(self, name, arg_vars: List[str], body):
        self.util_fns.add_fn(name, arg_vars, body)

    def run_util_fn(self, fn_name, *args):
        return self.util_fns.run_param_fnc(fn_name, *args)

    def set_parent(self, node_id):
        self._has_parent = node_id

    def get_viz_attr(self):
        raise NotImplementedError

    def has_primitive(self, name):
        if name in self.primitives:
            return True
        else:
            for n in self.get_subgraph_nodes():
                if n.has_primitive(name):
                    return True
        return False

    def has_codelet(self, name):
        return name in self.all_codelet_names

    def add_subgraph_edge(self, src, dst, attributes=None, transfer_fn_map=None):

        if self._has_parent:
            raise RuntimeError("Already added node to graph, cannot continue to add subgraph edges")
        if attributes:
            assert isinstance(attributes, dict)
            attr = []
            for k,v in attributes.items():
                attr.append({k: v})
        else:
            attr = []

        if isinstance(src, (int, str)):
            src = self.get_subgraph_node(src)

        if isinstance(dst, (int, str)):
            dst = self.get_subgraph_node(dst)

        if src.index not in self.subgraph._nodes:
            self.add_in_edge(src)

        if dst.index not in self.subgraph._nodes:
            self.add_out_edge(dst)

        edge = Edge(src=src.index, dst=dst.index, attributes=attr, transfer_fn_map=transfer_fn_map)
        self._subgraph_edges.append(edge)
        self.subgraph.add_edge(src, dst)

    def add_subgraph_node(self, node: 'ArchitectureNode'):
        if self._has_parent:
            raise RuntimeError("Already added node to graph, cannot continue to add subgraph nodes")


        self.merge_subgraph_nodes(node)
        node.set_parent(self.index)
        self.subgraph._add_node(node)
        self._subgraph_nodes[node.name] = node
        self._all_subgraph_nodes[node.name] = node

    def add_composite_node(self, node: 'ArchitectureNode', sub_nodes):
        for s in sub_nodes:
            s_node = self.get_subgraph_node(s)
            s_node.set_parent(None)
            node.add_subgraph_node(s_node)
            self.subgraph._nodes.pop(s_node.index)
            s_node.set_parent(node)
        self.add_subgraph_node(node)

    def get_operation_template(self, op):

        if isinstance(op, Transfer):
            template = self.operation_mappings['transfer'][(op.source, op.dest)]
        elif isinstance(op, Configure):
            template = self.operation_mappings['config'][op.target_name][op.start_or_finish]
        elif isinstance(op, Compute):
            template = self.operation_mappings['compute'][op.op_name]
        elif isinstance(op, Loop):
            template = self.operation_mappings['loop']
        else:
            raise TypeError(f"Invalid type for getting operation template: {type(op)}")
        return template


    # TODO: Need to validate that each parameter is correctly mapped
    def add_start_template(self, target, template, template_fns=None):
        if target not in self.operation_mappings['config']:
            self.operation_mappings['config'][target] = {}
        self.operation_mappings['config'][target]['start'] = OpTemplate(instructions=template, functions=template_fns)

    def add_end_template(self, target, template, template_fns=None):
        if target not in self.operation_mappings['config']:
            self.operation_mappings['config'][target] = {}
        self.operation_mappings['config'][target]['end'] = OpTemplate(instructions=template, functions=template_fns)

    def add_transfer_template(self, src, dst, template, template_fns=None):
        self.operation_mappings['transfer'][(src, dst)] = OpTemplate(instructions=template, functions=template_fns)

    def add_compute_template(self, target, op_name, template, template_fns=None):
        if target not in self.operation_mappings['compute']:
            self.operation_mappings['compute'][target] = {}
        self.operation_mappings['compute'][target][op_name] = OpTemplate(instructions=template, functions=template_fns)

    def add_loop_template(self, target, template, template_fns=None):
        self.operation_mappings['loop'][target] = OpTemplate(instructions=template, functions=template_fns)

    def add_in_edge(self, src):
        self._in_edges[src.field_name] = src
        self.subgraph.add_input(src)

    def add_out_edge(self, dst):
        self._out_edges[dst.field_name] = dst
        self.subgraph.add_output(dst)

    def get_subgraph_node(self, name) -> Union['ComputeNode', 'StorageNode', 'CommunicationNode']:
        if isinstance(name, str):
            if name in self._in_edges:
                return self._in_edges[name]
            elif name in self._out_edges:
                return self._out_edges[name]
            elif name in self._all_subgraph_nodes:
                return self._all_subgraph_nodes[name]
        else:
            assert isinstance(name, int)
            for n, v in self._all_subgraph_nodes.items():
                if v.index == name:
                    return v
        raise KeyError(f"{name} not found in subgraph or input_components")

    def get_type(self):
        return self._anode_type

    def merge_subgraph_nodes(self, node):
        intersection = node.subgraph._nodes.keys() & self.subgraph._nodes.keys()
        if len(intersection) > 0:
            raise RuntimeError(f"Overlapping keys when merging nodes for {self.name} and {node.field_name}")
        for name, n in node._all_subgraph_nodes.items():
            self._all_subgraph_nodes[n.name] = n
        self.subgraph._nodes.update(node.subgraph._nodes)


    def add_primitive(self, primitive: Instruction):
        if primitive.target is None:
            primitive.target = self.name
        self._primitives[primitive.name] = primitive

    def get_primitive_template(self, name) -> Instruction:
        if name in self.primitives:
            return self.primitives[name].instruction_copy()
        else:
            for n in self.get_subgraph_nodes():
                if n.has_primitive(name):
                    return n.get_primitive_template(name)
        raise KeyError(f"Primitive {name} not found!")

    def get_primitives(self) -> List[Instruction]:
        return list(self._primitives.keys())

    def add_codelet(self, codelet: Codelet):
        # TODO: Validate memory paths
        self._codelets[codelet.op_name] = codelet.codelet_copy()

    def get_codelet_template(self, name) -> Codelet:
        if name in self.codelets:
            return self.codelets[name].codelet_copy()
        else:
            for n in self.get_subgraph_nodes():
                if n.has_codelet(name):
                    return n.get_codelet_template(name)
        raise KeyError(f"Codelet {name} not found!")

    def get_codelets(self):
        return self._codelets.keys()

    def is_compatible(self, op_name):
        return op_name in self._primitives.keys()
    
    def set_occupied(self, op_code, primitive, begin_cycle, end_cycle):
        
        # check for overlaps, "o" is occupied and "n" is new
        n = (begin_cycle, end_cycle)
        overlaps = [o for o in self._occupied if o[2] > n[0] and o[2] < n[1] or o[3] > n[0] and o[3] < n[1]]
        assert len(overlaps) == 0, 'this op_node cannot be mapped here, check before using set_occupied'

        # append to _occupied
        self._occupied.append((op_code, primitive, begin_cycle, end_cycle))

    def get_occupied(self):
        return self._occupied

    def is_available(self, begin_cycle, end_cycle):

        # check for overlaps, "o" is occupied and "n" is new
        n = (begin_cycle, end_cycle)
        overlaps = [o for o in self._occupied if o[2] > n[0] and o[2] < n[1] or o[3] > n[0] and o[3] < n[1]]
        return len(overlaps) == 0

    @property
    def viz_color(self):
        raise NotImplementedError

    def get_subgraph_nodes(self) -> List['ComputeNode']:
        return list(self._subgraph_nodes.values())

    def get_subgraph_edges(self):
        return self._subgraph_edges

    def get_graph_node_count(self):
        count = 0
        for n in self.get_subgraph_nodes():
            count += (1 + n.get_graph_node_count())
        return count

    def get_graph_edge_count(self):
        count = len(self.get_subgraph_edges())
        for n in self.get_subgraph_nodes():
            count += n.get_graph_edge_count()
        return count

    def print_subgraph_edges(self, tabs=""):
        edge_pairs = [f"SRC: {self.subgraph.get_node_by_index(e.src).name}\t" \
                      f"DST:{self.subgraph.get_node_by_index(e.dst).name}" for e in self.get_subgraph_edges()]
        print(f"Total edges: {len(edge_pairs)}\n"
              f"Unique: {len(set(edge_pairs))}")
        print("\n".join(edge_pairs))
        tabs = tabs+"\t"
        for n in self.get_subgraph_nodes():
            n.print_subgraph_edges(tabs=tabs)

    def to_json(self):
        raise NotImplementedError

    def initialize_json(self):
        blob = {}
        blob['node_id'] = self.index
        blob['name'] = self.name
        blob['node_type'] = self.get_type()
        blob['node_color'] = self.viz_color
        blob['attributes'] = {}
        return blob

    def finalize_json(self, blob):
        blob['subgraph'] = {}
        blob['subgraph']['nodes'] = [sg.to_json() for sg in self.get_subgraph_nodes()]
        blob['subgraph']['edges'] = []
        for e in self.get_subgraph_edges():
            e_attr = {list(k.keys())[0]:  list(k.values())[0] for k in e.attributes}
            sub_edge = {'src': e.src, 'dest': e.dst, 'attributes': e_attr}
            blob['subgraph']['edges'].append(sub_edge)
        return blob
