from codelets.adl.architecture_node import ArchitectureNode

class ComputeNode(ArchitectureNode):

    def __init__(self, name=None):
        super(ComputeNode, self).__init__(name=name)
        self.set_attr("node_color", self.viz_color)

    @property
    def viz_color(self):
        return "#BFBFFF"