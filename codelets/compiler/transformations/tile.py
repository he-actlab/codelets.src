from collections import defaultdict
from itertools import chain
from typing import List

from codelets.adl import Codelet, ComputeNode
from codelets.adl.graph import ArchitectureNode, StorageNode
from codelets.adl.operation import Compute, OperandTemplate, Operation, Transfer
from codelets.compiler.transformations.transformations import split_operation, \
    TileConstraint


def tile(cdlt: Codelet, hag: ArchitectureNode) -> Codelet:
    cdlt = set_tile_levels(cdlt)
    op_list_copy = cdlt.ops.copy()

    # TODO: Fix extract band to extract multiple bands
    start_idx, end_idx = extract_band(cdlt)
    loop_levels = [o.loop_level for o in op_list_copy[start_idx:end_idx]]
    loop_level_factor = max(loop_levels) - min(loop_levels)
    added_ops = []


    cdlt = set_codelet_tiling(cdlt, hag)


    for level in range(len(cdlt.tile_levels) - 2):
        for op_idx in range(start_idx, end_idx):

            loop_level = (level + 1)*loop_level_factor + op_list_copy[op_idx].loop_level
            inner_op = split_operation(cdlt, op_list_copy[op_idx], loop_level, level)
            op_list_copy[op_idx] = inner_op

            added_ops.append(inner_op)
            # if isinstance(op, Transfer):
            #     src_shape = op.operand.sha
            #     for i, (pt, ofs) in enumerate(zip(op.path[1:], op.offsets[1:])):
            #         src_node = op.path[i]
            #         dst_node = pt
            #         xfer_info = op.transfers[(src_node, dst_node)]
            #         src_shape = [cdlt.domain_tiling[src_node][dom_key]
                                 # xfer_info._src_dim_sizes =

    assert isinstance(cdlt.ops[end_idx], Compute)
    cdlt.ops[end_idx].loop_level += (len(cdlt.tile_levels) - 2)*loop_level_factor
    cdlt.ops[end_idx:end_idx] = added_ops
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
    for level, nodes in list(cdlt.tile_levels.items()):
        for n in nodes:
            for domain_key, size in static_shapes.items():
                # TODO: For now, no splits are applied
                cdlt.set_domain_tile(n, domain_key, 1)

    # 1. go through loops and apply splits
    # 2. go through transfers and apply splits (computing size)
    # 3. while going through transfers, apply splits to operands
    # for o in cdlt.operands:
    #     for idx in range(len(o.evaluated_tiling)):
    #         hag_node, tiling = o.evaluated_tiling[idx]
    #         if len(tiling) != len(o.shape_list):
    #             assert idx != 0
    #
    #             o.evaluated_tiling[idx] = (hag_node, cdlt.domain_tiling[hag_node][])

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
    start_idx = -1
    end_idx = -1
    for idx, o in enumerate(cdlt.ops):
        if isinstance(o, Compute):
            transfer_deps = [cdlt.op_map[d] for d in o.dependencies]
            loop_deps = list(set(chain.from_iterable([tdep.dependencies + [tdep.op_str] for tdep in transfer_deps])))
            start_idx = idx - 1
            end_idx = idx
            while start_idx > 0:
                if cdlt.ops[start_idx].op_str in loop_deps:
                    start_idx -= 1
                else:
                    break
            break
    return (start_idx + 1, end_idx)


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