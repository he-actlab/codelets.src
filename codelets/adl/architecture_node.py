import numpy as np

from codelets.graph import Node
from codelets.adl.architecture_graph import ArchitectureGraph
from pygraphviz import AGraph

class ArchitectureNode(Node):
    """
    Base class for Architecture Node
    Inherited from Node
    """

    def __init__(self, name=None):
        super().__init__()
        self._has_parent = None
        self._subgraph = ArchitectureGraph()
        self._name = name
        if self.name:
            self.set_attr("name", self.name)
        # type
        self._anode_type = type(self).__name__
        self._subgraph_nodes = {}

        # capabilities: {name: {'delay': lambda function, 'encoding': TODO}}
        # NOTE maybe other information need to be added such as inputs and outputs... and 
        # their sources and destinations
        self._capabilities = {}

        # occupied: [(op_node, capability, begin_cycle, end_cycle)]
        # later consider changing to custom Heap because this needs to be accessed very frequently
        self._occupied = [] # NOTE state in TABLA compiler...
    
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

    def networkx_visualize(self, filename):
        dgraph = AGraph(strict=False, directed=True)

        for n in self.subgraph.get_nodes():
            self.subgraph._add_nx_subgraph(dgraph, n)
        dgraph.add_edges_from(self.subgraph.get_viz_edge_list())

        dgraph.layout("fdp")
        dgraph.draw(f"{filename}.pdf", format="pdf")
    #
    def set_parent(self, node_id):
        self._has_parent = node_id

    def add_subgraph_edge(self, src, dest):
        if self._has_parent:
            raise RuntimeError("Already added node to graph, cannot continue to add subgraph edges")
        self.subgraph._add_edge(src, dest)

    def add_subgraph_node(self, node: 'ArchitectureNode'):
        if self._has_parent:
            raise RuntimeError("Already added node to graph, cannot continue to add subgraph nodes")
        self.merge_subgraph_nodes(node)
        node.set_parent(self.index)
        self.subgraph._add_node(node)
        self._subgraph_nodes[node.name] = node

    def get_subgraph_node(self, name):
        if name not in self._subgraph_nodes:
            raise KeyError
        return self._subgraph_nodes[name]

    def get_type(self):
        return self._anode_type

    def merge_subgraph_nodes(self, node):
        intersection = node.subgraph._nodes.keys() & self.subgraph._nodes.keys()
        if len(intersection) > 0:
            raise RuntimeError(f"Overlapping keys when merging nodes")
        self.subgraph._nodes.update(node.subgraph._nodes)

    def set_capabilities(self, name, args):
        assert 'delay' in args.keys(), 'delay is mandatory information'
        self._capabilities[name] = args

    def get_capabilities(self):
        return self._capabilities

    def is_compatible(self, op_name):
        return op_name in self._capabilities.keys()

    # inputs are dictionary
    def use_capability(self, name, inputs):

        # identify inputs and outputs?
        # get begin_cycle and end_cycle
        # check is_available
        # set_occupied
        pass
    
    def set_occupied(self, op_node, capability, begin_cycle, end_cycle):
        
        # check for overlaps, "o" is occupied and "n" is new
        n = (begin_cycle, end_cycle)
        overlaps = [o for o in self._occupied if o[2] > n[0] and o[2] < n[1] or o[3] > n[0] and o[3] < n[1]]
        assert len(overlaps) == 0, 'this op_node cannot be mapped here, check before using set_occupied'

        # append to _occupied
        self._occupied.append((op_node, capability, begin_cycle, end_cycle))

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


