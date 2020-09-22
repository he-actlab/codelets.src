
class Codelet(object):
    def __init__(self, name, dfg_node):
        self._name = name
        self._node = dfg_node
        self._capability_list = []

    def get_node(self):
        return self._node

    @property
    def name(self):
        return self._name

    def instantiate(self, hag):
        raise NotImplementedError

    def emit(self):
        raise NotImplementedError

