import numpy as np

from codelets.graph import Node
from codelets.adl.architecture_graph import ArchitectureGraph
from typing import List, Dict, Union
from codelets.adl import Codelet, Instruction
from pygraphviz import AGraph
from collections import namedtuple

Edge = namedtuple('Edge', ['src', 'dests', 'attributes'])

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

        # capabilities
        self._capabilities = {}

        # capability_sequence
        self._codelets = {}

        # occupied: [(op_node, capability, begin_cycle, end_cycle)]
        # later consider changing to custom Heap because this needs to be accessed very frequently
        self._occupied = [] # NOTE state in TABLA compiler...

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
    def capabilities(self) -> Dict[str, Instruction]:
        return self._capabilities

    @property
    def codelets(self) -> Dict[str, Codelet]:
        return self._codelets

    @property
    def all_codelet_names(self) -> List[str]:
        names = [] + list(self.codelets.keys())
        for n in self.get_subgraph_nodes():
            names += n.all_codelet_names
        return names

    def networkx_visualize(self, filename):
        dgraph = AGraph(strict=False, directed=True)

        for n in self.subgraph.get_nodes():
            self.subgraph._add_nx_subgraph(dgraph, n)

        dgraph.add_edges_from(self.subgraph.get_viz_edge_list())

        dgraph.layout("dot")
        dgraph.draw(f"{filename}.pdf", format="pdf")
    #
    def set_parent(self, node_id):
        self._has_parent = node_id

    def get_viz_attr(self):
        raise NotImplementedError

    def has_capability(self, name):
        if name in self.capabilities:
            return True
        else:
            for n in self.get_subgraph_nodes():
                if n.has_capability(name):
                    return True
        return False

    def has_codelet(self, name):
        return name in self.all_codelet_names

    def add_subgraph_edge(self, src, dst, attributes=None):

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

        edge = Edge(src=src.index, dst=dst.index, attributes=attr)
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


    def add_in_edge(self, src):
        self._in_edges[src.field_name] = src
        self.subgraph.add_input(src)

    def add_out_edge(self, dst):
        self._out_edges[dst.field_name] = dst
        self.subgraph.add_output(dst)

    def get_subgraph_node(self, name) -> 'ArchitectureNode':
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


    def add_capability(self, capability: Instruction):
        if capability.target is None:
            capability.target = self.name
        self._capabilities[capability.name] = capability

    def get_capability(self, name) -> Instruction:
        if name in self.capabilities:
            return self.capabilities[name]
        else:
            for n in self.get_subgraph_nodes():
                if n.has_capability(name):
                    return n.get_capability(name)
        raise KeyError(f"Capability {name} not found!")

    def get_capabilities(self) -> List[Instruction]:
        return list(self._capabilities.keys())



    def add_codelet(self, codelet: Codelet):
        # TODO: Validate memory paths
        self._codelets[codelet.name] = codelet

    def get_codelet(self, name) -> Codelet:
        if name in self.codelets:
            return self.codelets[name]
        else:
            for n in self.get_subgraph_nodes():
                if n.has_codelet(name):
                    return n.get_codelet(name)
        raise KeyError(f"Codelet {name} not found!")

    def get_codelets(self):
        return self._codelets.keys()

    def is_compatible(self, op_name):
        return op_name in self._capabilities.keys()
    
    def set_occupied(self, op_code, capability, begin_cycle, end_cycle):
        
        # check for overlaps, "o" is occupied and "n" is new
        n = (begin_cycle, end_cycle)
        overlaps = [o for o in self._occupied if o[2] > n[0] and o[2] < n[1] or o[3] > n[0] and o[3] < n[1]]
        assert len(overlaps) == 0, 'this op_node cannot be mapped here, check before using set_occupied'

        # append to _occupied
        self._occupied.append((op_code, capability, begin_cycle, end_cycle))

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

    def get_subgraph_nodes(self) -> List['ArchitectureNode']:
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
            sub_edge = {'src': e.src, 'dests': e.dst, 'attributes': e_attr}
            blob['subgraph']['edges'].append(sub_edge)
        return blob
