import json
from typing import List
from codelets.adl.flex_template import FlexTemplate
from codelets.adl.graph import ArchitectureNode
from codelets.adl.operation import OperandTemplate, Loop, Compute, Transfer, Configure, Operation
from codelets.adl import Codelet
from pathlib import Path
import numpy as np
import polymath as pm
from .relocation_table import RelocationTable

EMIT_OPTIONS = ["decimal", "operations", "string_final", "string_placeholders", "decimal", "binary"]


class CodeletProgram(object):

    def __init__(self, name, hag: ArchitectureNode):
        self._name = name
        self._hag = hag
        self._codelets = []
        self._relocatables = RelocationTable()

    @property
    def name(self) -> str:
        return self._name

    @property
    def hag(self) -> ArchitectureNode:
        return self._hag

    @property
    def codelets(self) -> List[Codelet]:
        return self._codelets

    @property
    def relocatables(self) -> RelocationTable:
        return self._relocatables

    def add_codelet(self, cdlt: Codelet):
        self._codelets.append(cdlt)

    def get_codelet(self, cdlt_id: int):
        for cdlt in self.codelets:
            if cdlt.instance_id == cdlt_id:
                return cdlt
        raise KeyError(f"Unable to get codelet with id {cdlt_id} in codelet list")


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

    def save_binary(self, full_path):
        raise NotImplementedError

    def get_tiling_dims(self, inputs: List[OperandTemplate], outputs: List[OperandTemplate]):
        assert all([i.is_instantiated() for i in inputs])
        assert all([o.is_instantiated() for o in outputs])

    INSTR_FN_TEMPLATE = """def param_fn{FN_ID}(hag, op, cdlt, relocation_table, program, fixed_val=None): return {FN_BODY}"""

    def set_instruction_templates(self, cdlt: Codelet):
        for o in cdlt.ops:
            template = self.hag.get_operation_template(o)
            template_copy = [ft.template_copy() for ft in template]
            o.set_template(template_copy)

    def instantiate_instructions(self, cdlt: Codelet, fixed_val=None):
        # TODO: Replace this with evaluate
        cdlt._num_instr = 0
        for o in cdlt.ops:
            args = (self, self.hag, o.global_op_id, cdlt.instance_id)
            for flex_temp in o.instructions:
                flex_temp.set_instruction_length(*args)
                cdlt._num_instr += flex_temp.num_instructions

        self.relocatables.add_instr_relocation(cdlt)

        for o in cdlt.ops:
            args = (self, self.hag, o.global_op_id, cdlt.instance_id)

            for ft in o.instructions:
                ft.evaluate(*args)

    def instantiate_codelet(self, node):
        cdlt = self.hag.get_codelet_template(node.op_name)
        self.add_codelet(cdlt)
        return cdlt

    def emit(self, output_type):
        codelet_strings = []
        for c in self.codelets:
            codelet_strings.append(c.emit(output_type))
        if output_type != "json":
            return "\n".join(codelet_strings)
        else:
            return codelet_strings


    def instantiate_instructions_templates(self, node, cdlt):
        self.set_instruction_templates(cdlt)
        self.relocatables.add_data_relocation(node)
        self.instantiate_instructions(cdlt)


    # TODO: Fix these
    def save_json(self, full_path):
        json_blob = []
        with open(full_path, 'w') as outfile:
            json.dump(json_blob, outfile, indent=4)

    def save_text(self, full_path):
        instructions = []
        instructions = "\n".join(instructions)
        with open(full_path, 'w') as outfile:
            outfile.write(instructions)


def generate_possible_tilings(shape_dict, memory_paths):
    possible_tilings = {}

    for k, v in shape_dict.items():
        tile_permutations = []

def tiling_constraint(shapes, node_capacities, tile_sizes):

    for i, t in enumerate(tile_sizes):
        data_size = np.prod([t[s] for s in shapes])
        if data_size >= node_capacities[i]:
            return False
    return True
