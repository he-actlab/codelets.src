import json
import polymath as pm
from typing import List, Dict, Callable
from codelets.adl import ArchitectureNode, CodeletInstance
from pathlib import Path
import numpy as np
from collections import defaultdict

class CodeletProgram(object):

    def __init__(self, name, hag: ArchitectureNode):
        self._name = name
        self._hag = hag
        self._codelets = []
        self._relocatables = {}
        # TODO: Set datatype here
        self._relocatables['INSTR_MEM'] = {'total_length': 0,
                                            'bases': defaultdict(dict)
                                            }
        self._relocatables['INPUT'] = {'total_length': 0,
                                         'bases': defaultdict(dict)}
        self._relocatables['STATE'] = {'total_length': 0,
                                         'bases': defaultdict(dict)}
        self._relocatables['INTERMEDIATE'] = {'total_length': 0,
                                         'bases': defaultdict(dict)}
        self._relocatables['SCRATCH'] = {'total_length': 0,
                                         'bases': defaultdict(dict)}

    @property
    def name(self):
        return self._name

    @property
    def hag(self) -> ArchitectureNode:
        return self._hag

    @property
    def codelets(self) -> List[CodeletInstance]:
        return self._codelets

    @property
    def relocatables(self):
        return self._relocatables

    def add_codelet(self, cdlt: CodeletInstance):
        self._codelets.append(cdlt)

    def save(self, output_path=None, save_format="json"):
        if output_path:
            if output_path[-1] != "/":
                output_path = output_path + "/"
        else:
            output_path = str(Path.cwd()) + "/"

        if save_format == "json":
            full_path = f"{output_path}{self.name}.json"
            self.save_json(full_path)
        elif save_format == "text":
            full_path = f"{output_path}{self.name}.txt"
            self.save_text(full_path)
        else:
            raise ValueError(f"Invalid file output: {save_format}")

    def save_json(self, full_path):
        json_blob = [o.compiled_json() for o in self.codelets]
        with open(full_path, 'w') as outfile:
            json.dump(json_blob, outfile, indent=4)

    def save_text(self, full_path):
        instructions = []
        for c in self.codelets:
            instructions += c.get_text_instructions()
        instructions = "\n".join(instructions)
        with open(full_path, 'w') as outfile:
            outfile.write(instructions)

    def save_binary(self, full_path):
        raise NotImplementedError

    def update_relocation_offset(self, offset_type, offset_id, size):
        current_offset = self.relocatables[offset_type]['total_length']
        if offset_id not in self.relocatables[offset_type]['bases']:
            self.relocatables[offset_type]['bases'][offset_id]['start'] = current_offset
            self.relocatables[offset_type]['bases'][offset_id]['end'] = current_offset + size
            self.relocatables[offset_type]['total_length'] += size
        else:
            stored_size = self.relocatables[offset_type]['bases'][offset_id]['end'] - self.relocatables[offset_type]['bases'][offset_id]['start']
            assert stored_size == size

    def add_relocation(self, node: pm.Node, cdlt: CodeletInstance):
        instr_len = len(cdlt.capabilities)
        self.update_relocation_offset('INSTR_MEM', cdlt.codelet_id, instr_len)
        for i in node.inputs:
            data_size = np.prod(i.shape)
            if isinstance(i, pm.input):
                offset_type = 'INPUT'
            elif isinstance(i, pm.state):
                offset_type = 'STATE'
            else:
                offset_type = 'INTERMEDIATE'
            self.update_relocation_offset(offset_type, i.name, data_size)

        for o in node.outputs:
            data_size = np.prod(o.shape)
            offset_type = 'INTERMEDIATE'
            self.update_relocation_offset(offset_type, o.name, data_size)

    def instantiate_codelet(self, node):
        cdlt = self.hag.get_codelet(node.op_name)
        instance_params = {}
        cdlt.set_input_types(node)
        cdlt.set_output_types(node)
        cap_copy = [c.copy() for c in cdlt.capability_sequence]


        for k, v in cdlt.op_params.items():
            if isinstance(v, Callable):
                instance_params[k] = v(node)
        cdlt_instance = CodeletInstance(cap_copy, node.inputs, node.outputs, cdlt, op_params=instance_params)
            # TODO: Need to make sure all nodes have input/output defined
        self.add_relocation(node, cdlt_instance)
        return cdlt_instance


