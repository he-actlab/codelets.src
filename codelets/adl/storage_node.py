from codelets.adl.architecture_node import ArchitectureNode

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
                 buff_scheme=None,
                 latency=0,
                 input_ports=1,
                 output_ports=1,
                 index=None):
        super(StorageNode, self).__init__(name=name, index=index)
        self.set_attr("node_color", self.viz_color)
        self.read_bw = read_bw
        self.write_bw = write_bw
        self.access_type = access_type
        self.size = size
        self.input_ports = input_ports
        self.output_ports = output_ports
        self.buffering_scheme = buff_scheme or "single"
        self.latency = latency

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
    def latency(self):
        return self._latency

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

    @latency.setter
    def latency(self, latency):
        self._latency = latency

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

    @input_ports.setter
    def input_ports(self, input_ports):
        self._input_ports = input_ports

    @output_ports.setter
    def output_ports(self, output_ports):
        self._output_ports = output_ports

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

