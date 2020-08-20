from codelets.adl.architecture_node import ArchitectureNode

class CommunicationNode(ArchitectureNode):

    def __init__(self, name, comm_type, bandwidth,  index=None):
        super(CommunicationNode, self).__init__(name, index=index)
        self._comm_type = comm_type
        self._bandwidth = bandwidth
        self.set_attr("node_color", self.viz_color)

    @property
    def viz_color(self):
        return "#BFFFBF"

    def get_comm_type(self):
        return self._comm_type

    def get_bandwidth(self):
        return self._bandwidth