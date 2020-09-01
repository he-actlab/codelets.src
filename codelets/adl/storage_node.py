from codelets.adl.architecture_node import ArchitectureNode

class StorageNode(ArchitectureNode):
    ACCESS_TYPES = ["FIFO", "RAM"]

    def __init__(self, name, read_bw, write_bw, access_type, size, index=None):
        super(StorageNode, self).__init__(name=name, index=index)
        self.set_attr("node_color", self.viz_color)
        self.set_read_bw(read_bw)
        self.set_write_bw(write_bw)
        self.set_access_type(access_type)
        self.set_size(size)

    @property
    def viz_color(self):
        return "#7FFFFF"
    
    def set_read_bw(self, bw):
        self._read_bw = bw

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
