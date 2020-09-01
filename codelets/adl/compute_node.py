from codelets.adl.architecture_node import ArchitectureNode

# NOTE originally, there were implementations for occupancy. This has been 
#      removed because this can be managed by the compiler during the scheduling
#      phase as a scoreboard or something instead of instance of architecture graph.

class ComputeNode(ArchitectureNode):

    def __init__(self, name, index=None):
        super(ComputeNode, self).__init__(name, index=index)
        self.set_attr("node_color", self.viz_color)
        
        self._capabilities = {}


    @property
    def viz_color(self):
        return "#BFBFFF"
