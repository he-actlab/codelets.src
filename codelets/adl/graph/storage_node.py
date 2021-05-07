from codelets.adl.graph.architecture_node import ArchitectureNode
from typing import Dict


class StorageNode(ArchitectureNode):
    ACCESS_TYPES = ["FIFO", "RAM"]
    BUFF_SCHEMES = {"single": 1,
                    "double": 2,
                    "quadruple": 4}

    def __init__(self, name,
                 access_type=None,
                 banks=-1,
                 buffering_scheme=None,
                 latency=0,
                 input_ports=1,
                 output_ports=1,
                 width=-1,
                 depth=-1,
                 indirection=False,
                 on_chip=True,
                 index=None):
        super(StorageNode, self).__init__(name=name, index=index)
        # Configuration Attributes
        self._access_type = access_type
        self._banks = banks
        self._width = width
        self._depth = depth
        self._input_ports = input_ports
        self._output_ports = output_ports
        self._buffering_scheme = buffering_scheme or "single"
        self._indirection = indirection
        self._latency = latency
        self._on_chip = on_chip
        # Visualization Attributes
        self._node_color = "#7FFFFF"

    @property
    def attribute_names(self):
        return ["access_type", "banks", "width", "depth", "input_ports", "output_ports", "buffering_scheme",
                "indirection", "latency", "on_chip"]

    # Configuration Attributes
    @property
    def access_type(self):
        return self._access_type

    @access_type.setter
    def access_type(self, access_type):
        assert access_type in StorageNode.ACCESS_TYPES
        self._access_type = access_type

    @property
    def banks(self):
        return self._banks

    @banks.setter
    def banks(self, banks):
        self._banks = banks

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        self._width = width

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, depth):
        self._depth = depth

    @property
    def input_ports(self):
        return self._input_ports

    @input_ports.setter
    def input_ports(self, input_ports):
        self._input_ports = input_ports

    @property
    def output_ports(self):
        return self._output_ports

    @output_ports.setter
    def output_ports(self, output_ports):
        self._output_ports = output_ports

    @property
    def buffering_scheme(self):
        return self._buffering_scheme

    @buffering_scheme.setter
    def buffering_scheme(self, scheme):
        if isinstance(scheme, int):
            self._buffering_scheme = scheme
        else:
            assert isinstance(scheme, str) and scheme in StorageNode.BUFF_SCHEMES
            self._buffering_scheme = StorageNode.BUFF_SCHEMES[scheme]

    @property
    def indirection(self):
        return self._indirection

    @indirection.setter
    def indirection(self, indirection):
        self._indirection = indirection

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @property
    def on_chip(self):
        return self._on_chip

    @on_chip.setter
    def on_chip(self, on_chip):
        self._on_chip = on_chip

    # Visualization Attributes
    @property
    def viz_color(self):
        return self._node_color

    @property
    def node_color(self):
        return self._node_color

    # Derived / Other Attributes
    @property
    def size(self):
        return self._depth * self._width * self._banks

    @property
    def num_elements(self):
        return self._depth * self._banks

    @property
    def size_bytes(self):
        return self.size // 8

    @property
    def node_type(self):
        return 'storage'

    # Class methods
    def get_viz_attr(self):
        return f"Access Type: {self.access_type}\\n" \
               f"Size: {self.size}"

    def to_json(self) -> Dict:
        blob = self.initialize_json()
        blob['attributes']['access_type'] = self.access_type
        blob['attributes']['banks'] = self.banks
        blob['attributes']['width'] = self.width
        blob['attributes']['depth'] = self.depth
        blob['attributes']['input_ports'] = self.input_ports
        blob['attributes']['output_ports'] = self.output_ports
        blob['attributes']['buffering_scheme'] = self.buffering_scheme
        blob['attributes']['indirection'] = self.indirection
        blob['attributes']['latency'] = self.latency
        blob['attributes']['on_chip'] = self.on_chip
        blob = self.finalize_json(blob)
        return blob

    def from_json(self, blob):
        self.initialize_from_json(blob)
        self.access_type = blob['attributes']['access_type']
        self.banks = blob['attributes']['banks']
        self.width = blob['attributes']['width']
        self.depth = blob['attributes']['depth']
        self.input_ports = blob['attributes']['input_ports']
        self.output_ports = blob['attributes']['output_ports']
        self.buffering_scheme = blob['attributes']['buffering_scheme']
        self.indirection = blob['attributes']['indirection']
        self.latency = blob['attributes']['latency']
        self.on_chip = blob['attributes']['on_chip']
