
from codelets.examples.genesys_capabilities import GENESYS_CAPS
from codelets.adl import Capability
import json
import polymath as pm

def save_ops(operations, full_path):
    json_blob = [o.json_op() for o in operations]
    with open(full_path, 'w') as outfile:
        json.dump(json_blob, outfile, indent=4)

def compile(program_graph, hag, output_path):
    # TODO: Add lowering pass using HAG
    filename = program_graph.name
    if output_path[-1] != "/":
        output_path = output_path + "/"
    full_path = f"{output_path}compiled_{filename}.json"
    node_sequence = sequence_nodes(program_graph, hag)
    operations = map_tile_nodes(node_sequence, hag)
    # TODO: memory key-value pairs in tiling json
    # TODO: Input op names
    # op_type: input, output etc

    save_ops(operations, full_path)
    return operations

def sequence_nodes(program_graph, hag, sequence_algorithm="default"):
    node_list = []
    if sequence_algorithm == "default":
        for name, node in program_graph.nodes.items():
            if node.op_name in GENESYS_CAPS:
                node_list.append(node)
            elif not isinstance(node, (pm.placeholder, pm.write)):
                print(node.op_name)
    else:
        raise RuntimeError(f"{sequence_algorithm} is not a valid sequencing algorithm")
    return node_list


def map_tile_nodes(node_sequence, hag):
    operations = []
    for n in node_sequence:
        op = GENESYS_CAPS[n.op_name](n, hag)
        assert isinstance(op, Capability)
        operations.append(op)

    return operations

