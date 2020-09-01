import numpy as np

# TODO does different on-chip dataflow need to be considered?
#      if so, how should it be included?
#      current implementation considers a single type of on-chip dataflow!

class Capability(object):
    """
    Base class for capability
    """

    def __init__(self, name):
        self._name = name

        # list of inputs should be identical regardless of dataflows
        # 'input name': 'dims'
        self._inputs = {}
        self._outputs = {}
        self._subcapabilities = []

        # latency can be either a fixed value or a lambda function
        self._latency = 0

    def set_name(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def get_sub_capabilities(self):
        return self._subcapabilities

    def add_input(self, name, src, dims):
        self._inputs[name] = {'src': src, 'dims': dims}

    def get_inputs(self):
        return self._inputs.keys()
    
    def get_required_input_dims(self):
        # this one liner identifies unique dimensions that are required by the capability
        return set(sum([list(self._inputs[name]['dims'].keys()) for name in self._inputs.keys()], []))

    def get_required_input_ranges(self, dim):
        dim_ranges = [self._inputs[name]['dims'][dim] for name in self._inputs.keys()]
        equal = len(set(dim_ranges)) <= 1
        assert equal, 'ranges should be identically defined for same dim'
        return set(dim_ranges)[0]


    def add_output(self, name, dst, dims):
        self._outputs[name] = {'dst': dst, 'dims': dims}

    def get_outputs(self):
        return self._outputs.keys()

    def get_required_output_dims(self):
        # this one liner identifies unique dimensions that are required by the capability
        return set(sum([list(self._outputs[name]['dims'].keys()) for name in self._outputs.keys()], []))

    def get_required_output_ranges(self, dim):
        dim_ranges = [self._outputs[name]['dims'][dim] for name in self._outputs.keys()]
        equal = len(set(dim_ranges)) <= 1
        assert equal, 'ranges should be identically defined for same dim'
        return set(dim_ranges)[0]
    

    def set_latency(self, latency):
        self._latency = latency

    def get_latency(self):
        return self._latency

    def get_output(self, name):
        assert name in self._outputs
        return self._outputs[name]

    def get_input(self, name):
        assert name in self._inputs
        return self._inputs[name]