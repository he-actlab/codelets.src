from typing import Tuple, Dict
from codelets.adl.graph import ArchitectureNode
from .program import CodeletProgram
from .transformations import tile, lift_operations
import polymath as pm

TileConstraint = Dict[Tuple[str, str], Tuple[int, int]]


def compile(program_graph, hag: ArchitectureNode, output_path, store_output=True, output_type="json"):
    # TODO: Add lowering pass using HAG
    program = CodeletProgram(program_graph.name, hag)
    node_sequence = sequence_nodes(program_graph, hag)
    program = map_tile_nodes(node_sequence, program)
    # if store_output:
    #     program.save(output_path, save_format=output_type)
    # return program
    return program


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
    # This function performs breadth-first compilation, with coarsest abstractions first:
    # 1. Generate codelets from nodes
    # 2. Generate operands/operations within codelets
    # 3. Generate instruction templates within operations
    for n in node_sequence:
        cdlt = program.instantiate_codelet(n)
        codelets[n.name] = cdlt

    for n in node_sequence:
        cdlt = codelets[n.name]
        cdlt.instantiate_operations(n, program.hag)
        # TODO: Check if certain optimizations are necessary
        # cdlt = lift_operations(codelets[n.name])
        # # Sequence of transformations here
        cdlt = tile(cdlt, program.hag)
        codelets[n.name] = cdlt

    for n in node_sequence:
        program.instantiate_instructions_templates(n, codelets[n.name])

    return program

def naive_tiling_fn():
    pass




