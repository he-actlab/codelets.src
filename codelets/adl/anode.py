
class ArchitectureNode(object):

    def __index__(self, ntype, uid, in_edges, out_edges):
        self.uid = uid
        self.ntype = ntype
        self.out_nodes = dict([(oe.ntype, oe.uid) for oe in out_edges])
        self.in_nodes = dict([(ie.ntype, ie.uid) for ie in in_edges])
        self.busy = []

    def is_available(self, cycle):
        raise NotImplementedError

class ComputeNode(ArchitectureNode):
    def __init__(self, operations, *args):
        self.operations = operations
        super(ComputeNode, self).__init__("compute", *args)

class StorageNode(ArchitectureNode):
    def __init__(self, storage_type, capacity, *args):
        self.capacity = capacity
        self.storage_type = storage_type
        self.stored_data = []
        super(StorageNode, self).__init__("storage", *args)

    def add_data(self, data_id):
        self.stored_data.append(data_id)

    def has_data(self, data_id):
        return data_id in self.stored_data

class CommunicationNode(ArchitectureNode):
    def __init__(self, comm_type, *args):
        self.comm_type = comm_type
        super(CommunicationNode, self).__init__("communication", *args)
