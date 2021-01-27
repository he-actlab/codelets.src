from collections import defaultdict
from itertools import chain
from typing import List

from codelets.adl import Codelet, ComputeNode
from codelets.adl.graph import ArchitectureNode, StorageNode
from codelets.adl.operation import Compute, OperandTemplate, Operation, Transfer
from codelets.compiler.transformations.transformations import split_operation, \
    TileConstraint
from codelets.compiler.analysis import collect_operand_dependencies


def tile(cdlt: Codelet, hag: ArchitectureNode) -> Codelet:
    cdlt = set_tile_levels(cdlt)
    op_list_copy = cdlt.ops.copy()

    # TODO: Fix extract band to extract multiple bands
    # start_idx, end_idx = extract_band(cdlt)
    read_start_idx, read_end_idx, write_start_idx, write_end_idx = extract_band(cdlt)

    # TODO: Need to fix this logic, as it doesnt make sense with interleaved loop levels
    loop_levels = [o.loop_level for o in op_list_copy[read_start_idx:write_end_idx]]
    loop_level_factor = max(loop_levels) - min(loop_levels)
    cdlt = set_codelet_tiling(cdlt, hag)


    # TODO: THis currently only works for 2 transfers, need to fix to work for 2+
    for level in range(len(cdlt.tile_levels) - 2):
        split_level_factor = level + 1
        # First split reads
        for read_offset in range(read_start_idx, read_end_idx):
            outer_op_idx = read_offset + (read_end_idx - read_start_idx)*(split_level_factor - 1)
            inner_op_idx = read_offset + (read_end_idx - read_start_idx)*(split_level_factor)
            loop_level = loop_level_factor + cdlt.ops[outer_op_idx].loop_level
            inner_op = split_operation(cdlt, cdlt.ops[outer_op_idx], loop_level, level)
            cdlt.insert_op(inner_op, inner_op_idx)

        # Now update compute loop level
        compute_idx = read_end_idx + (read_end_idx - read_start_idx) * (split_level_factor)
        cdlt.ops[compute_idx].loop_level = cdlt.ops[compute_idx].loop_level + loop_level_factor

        write_start = write_start_idx + (read_end_idx - read_start_idx) * (split_level_factor) + 1
        write_end = write_end_idx + (read_end_idx - read_start_idx) * (split_level_factor) + 1

        # Now split writes
        for write_offset in range(write_start, write_end):
            inner_op_idx = write_offset + (write_end_idx - write_start_idx)*(split_level_factor - 1)
            outer_op_idx = write_offset + (write_end_idx - write_start_idx)*(split_level_factor)
            if level == 0:
                cdlt.ops[inner_op_idx].loop_level += loop_level_factor
            loop_level = cdlt.ops[inner_op_idx].loop_level - loop_level_factor
            outer_op = split_operation(cdlt, cdlt.ops[inner_op_idx], loop_level, level)
            cdlt.insert_op(outer_op, outer_op_idx)

    return cdlt

def shape_from_offset(offset):
    pass

def set_codelet_tiling(cdlt: Codelet, hag: ArchitectureNode):

    tile_constraints = get_tile_constraints(hag, cdlt.operands)
    # TESTING
    temp_tiling = {}
    static_shapes = cdlt.get_operand_shapes()
    for level, nodes in cdlt.tile_levels.items():
        for arch_node in nodes:
            temp_tiling[arch_node] = static_shapes

    # TESTING
    # TODO: Add another level of memory to test second level of tiling

    for level, nodes in list(cdlt.tile_levels.items()):
        for domain_key, size in static_shapes.items():
            # TODO: For now, no splits are applied
            if domain_key == "OC":
                cdlt.set_domain_tile(level, domain_key, 2)
            elif domain_key == "OH" and size % 2 == 0:
                cdlt.set_domain_tile(level, domain_key, 2)
            else:
                cdlt.set_domain_tile(level, domain_key, 1)

    # 1. go through loops and apply splits
    # 2. go through transfers and apply splits (computing size)
    # 3. while going through transfers, apply splits to operands
    # for o in cdlt.operands:
    #     for idx in range(len(o.evaluated_tiling)):
    #         tile_level, tiling = o.evaluated_tiling[idx]
    #         if len(tiling) != len(o.shape_list):
    #             assert idx != 0
    #
    #             o.evaluated_tiling[idx] = (tile_level, cdlt.domain_tiling[tile_level][])

    return cdlt

def get_tile_constraints(hag: ArchitectureNode, operands: List[OperandTemplate]) -> TileConstraint:
    path_constraints = {}
    for o in operands:
        for src_name, dst_name in o.tiling.keys():
            if (src_name, dst_name) in path_constraints:
                continue
            src_node = hag.get_subgraph_node(src_name)
            dst_node = hag.get_subgraph_node(dst_name)

            # TODO: Need a better way to characterize constraints here
            if isinstance(dst_node, ComputeNode):
                assert isinstance(src_node, StorageNode)
                max_size = src_node.read_bw
                min_size = max_size
            elif isinstance(dst_node, StorageNode):

                if isinstance(src_node, ComputeNode):
                    max_size = min_size = dst_node.write_bw
                else:
                    assert isinstance(src_node, StorageNode)
                    max_size = dst_node.size
                    min_size = 1
            else:
                raise TypeError(f"Unable to handle architecture node type {type(dst_node)}")
            path_constraints[(src_name, dst_name)] = (min_size, max_size)

    return path_constraints


def extract_band(cdlt: Codelet):
    read_start_idx = write_start_idx = len(cdlt.ops)
    read_end_idx = write_end_idx = -1

    output_dependencies = []

    for output in cdlt.outputs:
        output_dependencies += collect_operand_dependencies(output, cdlt)

    read_ops = True
    for idx, o in enumerate(cdlt.ops):
        if o.op_str in output_dependencies:
            if isinstance(o, Compute):
                read_ops = False
                read_end_idx = idx
                write_start_idx = idx
            elif read_ops and read_start_idx > idx:
                read_start_idx = idx
            else:
                write_end_idx = idx

    return (read_start_idx, read_end_idx, write_start_idx, write_end_idx)



def set_tile_levels(cdlt: Codelet):

    all_paths = [list(map(lambda et: et[0], o.evaluated_tiling)) for o in cdlt.operands]
    for p in all_paths:
        level_path = []
        for level, node in enumerate(p):
            if node not in level_path:
                level_path.append(node)
                if node not in cdlt.tile_levels[level]:
                    cdlt.set_tile_level(level, node)
            else:
                break
    return cdlt