from codelets.adl.architecture_node import ArchitectureNode

class StorageNode(ArchitectureNode):
    ACCESS_TYPES = ["FIFO", "RAM"]
    def __init__(self, name, read_bw, write_bw, access_type, capacity, index=None):
        super(StorageNode, self).__init__(name=name, index=index)
        self._read_bw = read_bw
        self._write_bw = write_bw
        assert access_type in StorageNode.ACCESS_TYPES
        self._access_type = access_type
        self._capacity = capacity
        self.set_attr("node_color", self.viz_color)

    @property
    def viz_color(self):
        return "#7FFFFF"

    def get_read_bw(self):
        return self._read_bw

    def get_write_bw(self):
        return self._write_bw

    def get_access_type(self):
        return self._access_type

    def get_capacity(self):
        return self._capacity
