from codelets.adl.codelet import Codelet
from typing import List
from codelets.adl.graph import ArchitectureNode
import json
from .program import CodeletProgram
import polymath as pm

def save_json(codelets: List[Codelet], full_path):
    json_blob = [o.compiled_json() for o in codelets]
    with open(full_path, 'w') as outfile:
        json.dump(json_blob, outfile, indent=4)

def save_text(codelets: List[Codelet], full_path):
    instructions = []
    for c in codelets:
        instructions += c.get_text_instructions()
    instructions = "\n".join(instructions)
    with open(full_path, 'w') as outfile:
        outfile.write(instructions)


# TODO: Implement this
def save_binary(codelets: List[Codelet], full_path, annotated=False):
    json_blob = [o.compiled_json() for o in codelets]
    with open(full_path, 'w') as outfile:
        json.dump(json_blob, outfile, indent=4)

def compile(program_graph, hag: ArchitectureNode, output_path, store_output=True, output_type="json"):
    # TODO: Add lowering pass using HAG
    program = CodeletProgram(program_graph.name, hag)
    node_sequence = sequence_nodes(program_graph, hag)
    program = map_tile_nodes(node_sequence, program)
    return program
    # if store_output:
    #     program.save(output_path, save_format=output_type)
    # return program

def sequence_nodes(program_graph: pm.Node, hag: ArchitectureNode, sequence_algorithm="default"):
    node_list = []
    if sequence_algorithm == "default":
        for name, node in program_graph.nodes.items():
            if hag.has_codelet(node.op_name):
                node_list.append(node)
    else:
        raise RuntimeError(f"{sequence_algorithm} is not a valid sequencing algorithm")

    return node_list

def map_tile_nodes(node_sequence, program: CodeletProgram) -> CodeletProgram:
    codelets = {}
    for n in node_sequence:
        cdlt = program.instantiate_codelet(n)
        codelets[n.name] = cdlt

    for n in node_sequence:
        codelets[n.name].instantiate_operations(n, program.hag)

    for n in node_sequence:
        program.instantiate_instructions_templates(codelets[n.name])

    return program

