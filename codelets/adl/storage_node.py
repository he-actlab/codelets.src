from codelets.adl.architecture_node import ArchitectureNode
from typing import Dict

class StorageNode(ArchitectureNode):
    ACCESS_TYPES = ["FIFO", "RAM"]
    BUFF_SCHEMES = {"single": 1,
                    "double": 2,
                    "quadruple": 4}
    def __init__(self, name,
                 read_bw=-1,
                 write_bw=-1,
                 access_type=None,
                 size=-1,
                 buffering_scheme=None,
                 latency=0,
                 input_ports=1,
                 output_ports=1,
                 width=-1,
                 indirection=False,
                 on_chip=True,
                 index=None):
        super(StorageNode, self).__init__(name=name, index=index)
        self.set_attr("node_color", self.viz_color)
        self.read_bw = read_bw
        self.write_bw = write_bw
        self.access_type = access_type
        self.size = size
        self.width = width
        self.input_ports = input_ports
        self.output_ports = output_ports
        self.buffering_scheme = buffering_scheme or "single"
        self.indirection = indirection
        self.latency = latency
        self.on_chip = on_chip

    @property
    def viz_color(self):
        return "#7FFFFF"

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
    def width(self):
        return self._width

    @property
    def latency(self):
        return self._latency

    @property
    def on_chip(self):
        return self._on_chip

    @property
    def input_ports(self):
        return self._input_ports

    @property
    def output_ports(self):
        return self._output_ports

    @property
    def read_bw(self):
        return self._read_bw

    @property
    def write_bw(self):
        return self._write_bw

    @property
    def access_type(self):
        return self._access_type

    @property
    def size(self):
        return self._size

    @property
    def size_bytes(self):
        return self._size * 1000

    @property
    def indirection(self):
        return self._indirection

    @width.setter
    def width(self, width):
        self._width = width

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @on_chip.setter
    def on_chip(self, on_chip):
        self._on_chip = on_chip

    @size.setter
    def size(self, size):
        self.set_size(size)

    @read_bw.setter
    def read_bw(self, read_bw):
        self.set_read_bw(read_bw)

    @write_bw.setter
    def write_bw(self, write_bw):
        self.set_write_bw(write_bw)

    @access_type.setter
    def access_type(self, access_type):
        self.set_access_type(access_type)


    @indirection.setter
    def indirection(self, indirection):
        self._indirection = indirection

    @input_ports.setter
    def input_ports(self, input_ports):
        self._input_ports = input_ports

    @output_ports.setter
    def output_ports(self, output_ports):
        self._output_ports = output_ports

    @property
    def node_type(self):
        return 'storage'

    def set_read_bw(self, bw):
        self._read_bw = bw

    def set_buffer_scheme(self, scheme):
        self._buffering_scheme = scheme

    def set_input_ports(self, input_buffers):
        self._input_ports = input_buffers

    def set_output_ports(self, output_buffers):
        self._output_ports = output_buffers

    def get_read_bw(self):
        return self._read_bw
    
    def set_write_bw(self, bw):
        self._write_bw = bw

    def get_write_bw(self):
        return self._write_bw

    def set_access_type(self, access_type):
        assert access_type in StorageNode.ACCESS_TYPES
        self._access_type = access_type

    def get_access_type(self):
        return self._access_type
    
    def set_size(self, size):
        self._size = size

    def get_size(self):
        return self._size

    def get_viz_attr(self):
        return f"R/W Bandwidth: {self.get_read_bw()}/{self.get_write_bw()}\\n" \
               f"Access Type: {self.get_access_type()}\\n" \
               f"Size: {self.get_size()}"

    def to_json(self) -> Dict:
        blob = self.initialize_json()
        blob['attributes']['read_bw'] = self.read_bw
        blob['attributes']['write_bw'] = self.write_bw
        blob['attributes']['access_type'] = self.access_type
        blob['attributes']['size'] = self.size
        blob['attributes']['input_ports'] = self.input_ports
        blob['attributes']['output_ports'] = self.output_ports
        blob['attributes']['buffering_scheme'] = self.width
        blob['attributes']['width'] = self.buffering_scheme
        blob['attributes']['latency'] = self.latency
        blob['attributes']['on_chip'] = self.on_chip
        blob = self.finalize_json(blob)
        return blob

