from codelets.adl.architecture_node import ArchitectureNode

class ComputeNode(ArchitectureNode):

    def __init__(self, name, capabilities=None, index=None):
        # capabilities: {name: {'delay': lambda function, 'encoding': TODO}}
        # NOTE maybe other information need to be added such as inputs and outputs... and
        # their sources and destinations
        super(ComputeNode, self).__init__(name, index=index)
        self._capabilities = {}
        if capabilities:
            assert isinstance(capabilities, dict)
            for cap_name, cap_attr in capabilities.items():
                self.set_capability(cap_name, cap_attr)

        self.set_attr("node_color", self.viz_color)

    @property
    def viz_color(self):
        return "#BFBFFF"

    def set_capability(self, name, args):
        assert 'delay' in args.keys(), 'delay is mandatory information'
        self._capabilities[name] = args

    def get_capabilities(self):
        return self._capabilities

    def is_compatible(self, op_name):
        return op_name in self._capabilities.keys()

    # inputs are dictionary
    def use_capability(self, name, inputs):

        # identify inputs and outputs?
        # get begin_cycle and end_cycle
        # check is_available
        # set_occupied
        pass

    def set_occupied(self, op_node, capability, begin_cycle, end_cycle):
        # check for overlaps, "o" is occupied and "n" is new
        n = (begin_cycle, end_cycle)
        overlaps = [o for o in self._occupied if o[2] > n[0] and o[2] < n[1] or o[3] > n[0] and o[3] < n[1]]
        assert len(overlaps) == 0, 'this op_node cannot be mapped here, check before using set_occupied'

        # append to _occupied
        self._occupied.append((op_node, capability, begin_cycle, end_cycle))