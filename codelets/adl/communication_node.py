from codelets.adl.architecture_node import ArchitectureNode

class CommunicationNode(ArchitectureNode):

    def __init__(self, name):
        super(CommunicationNode, self).__init__(name, index=index)
        self.set_attr("node_color", self.viz_color)

    def __init__(self, name, comm_type, latency, bw, index=None):
        super(CommunicationNode, self).__init__(name, index=index)
        self.set_attr("node_color", self.viz_color)

        self._comm_type = comm_type
        self._latency = latency
        self._bw = bw

    @property
    def viz_color(self):
        return "#BFFFBF"


    def set_comm_type(self, comm_type):
        self._comm_type = comm_type

    def get_comm_type(self):
        return self._comm_type


    def set_latency(self, latency):
        self._latency = latency

    def get_latency(self):
        return self._latency


    def set_bw(self, bw):
        self._bw = bw

    def get_bw(self):
        return self._bw
