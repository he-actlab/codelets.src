from codelets.adl.architecture_node import ArchitectureNode
from codelets.adl.codelet import Codelet

# NOTE originally, there were implementations for occupancy. This has been 
#      removed because this can be managed by the compiler during the scheduling
#      phase as a scoreboard or something instead of instance of architecture graph.

class ComputeNode(ArchitectureNode):

    def __init__(self, name, dimensions=None, capabilities=None, index=None):
        super(ComputeNode, self).__init__(name, index=index)
        self.set_attr("node_color", self.viz_color)
        self._capabilities = {}
        self.dimensions = dimensions or [1]
        if capabilities:
            for c in capabilities:
                if isinstance(c, dict):
                    cap = self.parse_capability_json(c)
                else:
                    cap = c
                self.add_capability(cap)

    @property
    def viz_color(self):
        return "#BFBFFF"

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def capabilities(self):
        return self._capabilities

    @dimensions.setter
    def dimensions(self, dimensions):
        assert isinstance(dimensions, list)
        self._dimensions = dimensions

    def parse_capability_json(self, cap_dict):
        name = cap_dict.pop("name")
        cap = Codelet(name, **cap_dict)
        return cap

    def get_viz_attr(self):
        caps = list(self.get_capabilities())
        if len(caps) > 5:
            return f"Capabilities: {caps[:5]}"
        else:
            return f"Capabilities: {caps}"

