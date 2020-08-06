import numpy as np

from codelets.graph import Node 

class ArchitectureNode(Node):
    """
    Base class for Architecture Node
    Inherited from Node
    """

    def __init__(self):
        super().__init__()

        # type
        self._anode_type = type(self).__name__

        # capabilities: {name: {'delay': lambda function, 'encoding': TODO}}
        # NOTE maybe other information need to be added such as inputs and outputs... and 
        # their sources and destinations
        self._capabilities = {}

        # occupied: [(op_node, capability, begin_cycle, end_cycle)]
        # later consider changing to custom Heap because this needs to be accessed very frequently
        self._occupied = [] # NOTE state in TABLA compiler...
    
    def __str__(self):
        return f'op {self.index} ({self.get_type()}): \
                 preds={self.get_preds_indices()} ({self._attrs["in_degree"]}), \
                 succs={self.get_succs_indices()} ({self._attrs["out_degree"]})'
    
    
    # modifying anode type arbitrarily should not be permitted
    #def set_type(self, anode_type):
    #    self._anode_type = anode_type

    def get_type(self):
        return self._anode_type

    
    def set_capabilities(self, name, args):
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

    def get_occupied(self):
        
        return self._occupied

    def is_available(self, begin_cycle, end_cycle):
        
        # check for overlaps, "o" is occupied and "n" is new
        n = (begin_cycle, end_cycle)
        overlaps = [o for o in self._occupied if o[2] > n[0] and o[2] < n[1] or o[3] > n[0] and o[3] < n[1]]
        return len(overlaps) == 0


