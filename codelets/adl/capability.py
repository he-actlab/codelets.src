import numpy as np

class Capability(object):
    """
    Base class for capability
    """

    def __init__(self, name):
        self._name = name

        # list of inputs should be identical regardless of dataflows
        # 'input name': 'dims'
        self._inputs = {}

        # basic instructions may have one 'default' dataflow, but some capabilities may have
        # multiple dataflows for same functionality.
        self._dataflows = {}


    def get_name(self):
        return self._name


    def add_input(self, name, src, dims):
        self._inputs[name] = {'src': src, 'dims': dims}

    def get_inputs(self):
        return self._inputs.keys()

    def _get_required_dims_from_inputs(self):
        # this one liner identifies unique dimensions that are required by the capability
        return set(sum(list(self._inputs['dims'].values()), []))


    # TODO 
    # list of things that determine delay... for example in communication nodes, it requires different things?
    # dependencies... for example, additional or less delay due to dependencies
    # encoding... for example, NCHW and NHWC may require different memory calculation
    def add_dataflow(self, name='default', delay=None, constraint=None):
        assert name not in self._dataflows.keys(), f'{name} dataflow is already defined'
        self._dataflows[name] = {'delay': delay, 'constraint': constraint}
        assert self._get_required_dims_from_inputs() == self._get_required_dims_from_dataflows(), 'required dims should be consistent'

    def get_dataflows(self):
        return self._dataflows.keys()

    def get_required_dims(self, dataflow='default'):
        return self._dataflows[dataflow]['constraint'].keys()

    # returns tuple (min, max) for the dimension, which denotes range: [min, max]
    def get_required_dim_range(self, dataflow='default', dim=None):
        return self._dataflows[dataflow]['constraint'][dim]

    def _get_required_dims_from_dataflows(self):
        return set(sum([list(self.get_required_dims(dataflow)) for dataflow in self.get_dataflows()], []))


    # TODO
    # dependencies... take as input a previous capability being executed, and see if they can be pipelined?
    #                 this may yield different delays...
    def get_delay(self, dataflow='default'):
        delay = self._dataflows[dataflow]['delay']
        assert delay, 'if delay is None, capability is invalid'
