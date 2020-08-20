from codelets.adl.architecture_node import ArchitectureNode

class CommunicationNode(ArchitectureNode):

    def __init__(self, name=None):
        super(CommunicationNode, self).__init__(name=name)
        self.set_attr("node_color", self.viz_color)

    @property
    def viz_color(self):
        return "#BFFFBF"