import networkx as nx
from codelets.adl.anode import CommunicationNode, StorageNode, ComputeNode

class ArchitectureDescription(object):

    def __init__(self, name, filepath=None):
        self.name = name
        self._nodes = {}
        self._delays = {}
        self.graph = nx.DiGraph()
        if filepath:
            self.load_from_file(filepath)

    @property
    def nodes(self):
        return self._nodes

    @property
    def delays(self):
        return self._delays

    def get_edge_delay(self, src_id, dst_id):
        src = self.nodes[src_id]
        dst = self.nodes[dst_id]
        return self.delays[src.ntype][dst.ntype]

    def load_from_file(self, filepath):
        pass

    def add_node(self, node, in_edges, out_edges):
        self.nodes[node.uid] = node
        self.graph.add_node(node.uid)

        for ie in in_edges:
            assert ie in self.nodes
            self.add_edge(ie, node.uid, self.get_edge_delay(ie, node.uid))

        for oe in out_edges:
            assert oe in self.nodes
            self.add_edge(node.uid, oe, self.get_edge_delay(node.uid, oe))

        return node

    def add_compute_node(self, in_edges, out_edges):
        node = ComputeNode(len(self.nodes))
        return self.add_node(node, in_edges, out_edges)

    def add_storage_node(self, in_edges, out_edges):
        node = StorageNode(len(self.nodes))
        return self.add_node(node, in_edges, out_edges)

    def add_communication_node(self, in_edges, out_edges):
        node = CommunicationNode(len(self.nodes))
        return self.add_node(node, in_edges, out_edges)

    def add_edge(self, node_a_id, node_b_id, delay):
        pass

    def add_node_attribute(self, node_id, key, value):
        pass

    def add_edge_attribute(self, node_id, key, value):
        pass