from codelets.graph import Graph
from codelets.visualizer import GraphVisualizer

class ArchitectureGraph(Graph):
    """
    Class for ArchitectureGraph
    """

    def __init__(self, old_name=None):
        super().__init__()

        # store the name of the graph from the .pb or .onnx
        self._old_name = old_name

    
    def visualize(self, filename='output'):
        
        digraph = Digraph(format='pdf')

        # draw nodes
        for node in self.get_nodes():
            attrs = None
            color = 'white'
            self._draw_node(digraph, node, attrs=attrs, color=color)
        
        # draw edges
        for src_node in self.get_nodes():
            for dst_node in src_node.get_succs():
                self._draw_edge(digraph, src_node, dst_node)

        # save as pdf
        digraph.render(filename)

    def _draw_node(self, digraph, node, attrs=None, shape='record', style='rounded,filled', color='white'):
        
        name = f'node_{node.index}'
        label = f'{type(node).__name__}'
        if attrs is not None:
            label += '|'
            for key, value in attrs.items():
                label += '{key}: {value}\l'
            label = '{{' + label + '}}'
        digraph.node(name, label, shape=shape, style=style, fillcolor=color)
    
    def _draw_edge(self, digraph, src_node, dst_node):
        
        src_name = f'node_{src_node.index}'
        dst_name = f'node_{dst_node.index}'

        digraph.edge(src_name, dst_name)


