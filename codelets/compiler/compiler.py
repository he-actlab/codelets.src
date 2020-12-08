from codelets.adl.codelet import CodeletInstance, Codelet
from typing import List
from codelets.adl import Capability, ArchitectureNode
import json
import polymath as pm

def save_json(codelets: List[CodeletInstance], full_path):
    json_blob = [o.compiled_json() for o in codelets]
    with open(full_path, 'w') as outfile:
        json.dump(json_blob, outfile, indent=4)

def save_text(codelets: List[CodeletInstance], full_path):
    instructions = []
    for c in codelets:
        instructions += c.get_text_instructions()
    instructions = "\n".join(instructions)
    with open(full_path, 'w') as outfile:
        outfile.write(instructions)


# TODO: Implement this
def save_binary(codelets: List[CodeletInstance], full_path, annotated=False):
    json_blob = [o.compiled_json() for o in codelets]
    with open(full_path, 'w') as outfile:
        json.dump(json_blob, outfile, indent=4)

def compile(program_graph, hag: ArchitectureNode, output_path, store_output=True, output_type="json"):
    # TODO: Add lowering pass using HAG

    node_sequence = sequence_nodes(program_graph, hag)
    codelets = map_tile_nodes(node_sequence, hag)
    # TODO: memory key-value pairs in tiling json
    # TODO: Input op names
    # op_type: input, output etc
    if store_output is not None:
        filename = program_graph.name
        if output_path[-1] != "/":
            output_path = output_path + "/"
        if output_type == "json":
            full_path = f"{output_path}compiled_{filename}.json"
            save_json(codelets, full_path)
        elif output_type == "text":
            full_path = f"{output_path}compiled_{filename}.txt"
            save_text(codelets, full_path)
        else:
            raise ValueError(f"Invalid file output: {output_type}")

    return codelets

def sequence_nodes(program_graph, hag: ArchitectureNode, sequence_algorithm="default"):
    node_list = []
    if sequence_algorithm == "default":
        for name, node in program_graph.nodes.items():
            if hag.has_codelet(node.op_name):
                node_list.append(node)
            # elif not isinstance(node, (pm.placeholder, pm.write)):
            #     print(node.op_name)
    else:
        raise RuntimeError(f"{sequence_algorithm} is not a valid sequencing algorithm")

    return node_list


def map_tile_nodes(node_sequence, hag: ArchitectureNode) -> List[CodeletInstance]:
    codelets = []
    for n in node_sequence:
        cdlt = hag.get_codelet(n.op_name)
        cdlt_instance = cdlt.instantiate_codelet(n, hag, codelets)
        # op = GENESYS_SA_CAPS[n.op_name](n, hag, codelets)
        codelets.append(cdlt_instance)
        assert isinstance(cdlt_instance, CodeletInstance)


    return codelets

