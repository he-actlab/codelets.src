from codelets.adl.architecture_node import ArchitectureNode

from codelets.adl.codelet import Codelet
from codelets.adl.instruction import Instruction
from typing import Dict

# NOTE originally, there were implementations for occupancy. This has been 
#      removed because this can be managed by the compiler during the scheduling
#      phase as a scoreboard or something instead of instance of architecture graph.

class ComputeNode(ArchitectureNode):

    def __init__(self, name, dimensions=None, codelets=None, capabilities=None, index=None):
        super(ComputeNode, self).__init__(name, index=index)
        self.set_attr("node_color", self.viz_color)
        self._capabilities = {}
        self._codelets = {}
        self.dimensions = dimensions or [1]
        if capabilities:
            for c in capabilities:
                if isinstance(c, dict):
                    cap = self.parse_capability_json(c)
                else:
                    cap = c
                self.add_capability(cap)

        # TODO: Check codelet capabilities for support
        if codelets:
            for c in codelets:
                if isinstance(c, dict):
                    cdlt = self.parse_codelet_json(c)
                else:
                    cdlt = c
                self.add_codelet(cdlt)

    @property
    def viz_color(self):
        return "#BFBFFF"

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def capabilities(self) -> Dict[str, Instruction]:
        return self._capabilities

    @property
    def codelets(self) -> Dict[str, Codelet]:
        return self._codelets

    @dimensions.setter
    def dimensions(self, dimensions):
        assert isinstance(dimensions, list)
        self._dimensions = dimensions

    def parse_capability_json(self, cap_dict):
        name = cap_dict.pop("field_name")
        cap = Codelet(name, **cap_dict)
        return cap

    def parse_codelet_json(self, cdlt_dict):
        name = cdlt_dict.pop("field_name")
        cap = Codelet(name, **cdlt_dict)
        return cap

    def get_viz_attr(self):
        caps = list(self.get_capabilities())
        if len(caps) > 5:
            return f"Capabilities: {caps[:5]}"
        else:
            return f"Capabilities: {caps}"

    def to_json(self) -> Dict:
        blob = self.initialize_json()
        blob['attributes']['dimensions'] = self.dimensions
        blob['attributes']['capabilities'] = [c.to_json() for k, c in self.capabilities.items()]
        blob['attributes']['codelets'] = [c.to_json() for k, c in self.codelets.items()]
        blob = self.finalize_json(blob)
        return blob