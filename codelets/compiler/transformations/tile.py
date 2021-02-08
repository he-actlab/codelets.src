from collections import defaultdict
from itertools import chain, combinations, product
from typing import List

from codelets.adl import Codelet, ComputeNode
from codelets.adl.graph import ArchitectureNode, StorageNode
from codelets.adl.operation import Compute, OperandTemplate, Operation, Transfer, size_from_extent, size_from_offsets
from . import split_operation, factors
from codelets.compiler.analysis import collect_operand_dependencies
from sympy import Basic, Idx
import numpy as np


def default_tile_heuristic(hag: ArchitectureNode, cdlt: Codelet, tiling_splits):
    total_accesses = 0
    for l, splits in tiling_splits.items():
        for _, s in splits.items():
            total_accesses += s
    return total_accesses



def tile(cdlt: Codelet, hag: ArchitectureNode, heuristic_fn=None) -> Codelet:
    cdlt.set_tile_levels()
    heuristic_fn = heuristic_fn or default_tile_heuristic
    # Find amount of splits for each loop by looking at dependencies
    loop_splits = {}
    for i, o in enumerate(cdlt.operands):
        loops = [d for d in o.dependencies if "loop" in d]
        max_level = max(cdlt.get_tile_level(dp) for dp in o.data_path)
        for l in loops:
            if l in loop_splits and loop_splits[l] < max_level:
                loop_splits[l] = max_level
            else:
                loop_splits[l] = max_level


    bands = cdlt.extract_bands()
    cdlt = set_codelet_tiling(cdlt, hag, heuristic_fn)

    for start, end in bands:
        idx = start
        splits = loop_splits[cdlt.ops[idx].op_str] - 1
        dep_mapping = {}
        for split in range(splits):
            op_band = cdlt.ops[start: end + 1]
            offset = (end - start)
            num_splits = 0
            for op in op_band:
                i = cdlt.ops.index(op)
                target_idx = offset + i

                if cdlt.ops[target_idx].op_type == "loop":
                    inner_loop_level = cdlt.ops[target_idx].loop_level + 1
                else:
                    inner_loop_level = cdlt.ops[target_idx].loop_level

                if inner_loop_level < op.loop_level:
                    raise RuntimeError

                inner_deps = [dep_mapping[dp] for dp in op.dependencies]
                new_op_id, new_global_id = cdlt.get_new_op_ids(op)
                extra_kwargs = {}

                if op.op_type == "transfer":
                    if len(op.path) <= 2:
                        dep_mapping[op.op_str] = op.op_str
                        offset -= 1
                        if cdlt.get_tile_level(op.path[0]) > cdlt.get_tile_level(op.path[1]):
                            cdlt.insert_op(op, target_idx)
                        continue
                    elif cdlt.get_tile_level(op.path[0]) > cdlt.get_tile_level(op.path[1]):
                        inner_path, outer_path = op.path[split: split + 2], op.path[split + 1:]
                        op._path = outer_path
                        extra_kwargs["path"] = inner_path
                        inner_op = cdlt.ops[i].copy(cdlt, loop_level=inner_loop_level,
                                                    op_id=new_op_id,
                                                    global_op_id=new_global_id,
                                                    dependencies=inner_deps, **extra_kwargs)
                        inner_idx = target_idx
                        dep_mapping[op.op_str] = inner_op.op_str

                        # Update outer op
                        op._dependencies.append(inner_op.op_str)
                        cdlt.insert_op(op, target_idx)

                    else:
                        outer_path, inner_path = op.path[split: split + 2], op.path[split + 1:]
                        op._path = outer_path
                        extra_kwargs["path"] = inner_path
                        inner_deps.append(op.op_str)
                        inner_op = cdlt.ops[i].copy(cdlt, loop_level=inner_loop_level,
                                                    op_id=new_op_id,
                                                    global_op_id=new_global_id,
                                                    dependencies=inner_deps, **extra_kwargs)


                        inner_idx = target_idx + 1
                        dep_mapping[op.op_str] = inner_op.op_str
                    num_splits += 1
                elif op.op_type == "loop":

                    inner_op = cdlt.ops[i].copy(cdlt, loop_level=inner_loop_level,
                                                    op_id=new_op_id,
                                                    global_op_id=new_global_id,
                                                    dependencies=inner_deps, **extra_kwargs)

                    dep_mapping[op.op_str] = inner_op.op_str
                    inner_idx = target_idx + 1
                    num_splits += 1
                else:
                    dep_mapping[op.op_str] = op.op_str
                    op.dependencies = inner_deps
                    op.loop_level = inner_loop_level
                    inner_op = op
                    inner_idx = target_idx
                    num_splits += 1
                cdlt.insert_op(inner_op, inner_idx)



    return cdlt


def set_codelet_tiling(cdlt: Codelet, hag: ArchitectureNode, heuristic_fn):

    tile_constraints = get_tile_constraints(cdlt, hag)
    level_accesses = defaultdict(list)
    loop_dependencies = []
    level_pairs = defaultdict(list)
    # Collect accesses and loop dependencies
    for o in cdlt.operands:
        for i, move in enumerate(o.test_data_moves):
            print(f"{o.name}: {move.src_node} -> {move.dst_node}, {move.op_name}")
        for i, access in enumerate(o.data_moves):
            # print(f"{o.name}: {access.access_type}, {access.hag_node}, {access.op_name}")

            level_accesses[cdlt.get_tile_level(access.hag_node)].append(access)
        print()
        loop_dependencies += [dp for dp in o.dependencies if dp not in loop_dependencies and "loop" in dp]
    print()
    loop_factors = {}
    # Find all loop factors
    for l in loop_dependencies:
        loop = cdlt.op_map[l]
        loop_factors[loop.op_str] = factors(loop.iter_count)


    perms = product(*tuple(loop_factors.values()))
    min_heuristic = float("inf")
    # for lev in list(cdlt.tile_levels.keys())[1:]:
    #     level = lev + 1
    #     for p in perms:
    #         perm_map = {list(loop_factors.keys())[i]: v for i, v in enumerate(p)}
    #         for level_access in level_accesses[level]:
    #             if

    #         for o in cdlt.operands:

        # for write in level_accesses[level]["write"]:
        #     op = cdlt.op_map[write.op_name]
        #     if op.op_type == "transfer" and not write.is_set():
        #         operand = cdlt.get_operand(write.operand_name)
        #         idx = operand.data_moves.index(write)
        #         read = operand.data_moves[idx - 1]
        #         assert read.op_name == write.op_name
        #         # for  o in read.offset_map.items():
        #
        #
        #     elif op.op_type == "compute":
        #         pass


    #     pass
            # start_key = tuple(o.data_path[level:level+2])
            #
            # start_idx = o.data_path[1]
            # start_size = size_from_extent(cdlt, o.data_moves[start_key])
            # start_constraints = tile_constraints[start_key]
            # print
            # for n in o.data_path:

    #         print(o.data_moves)

    # for level in list(cdlt.tile_levels.keys())[:-1]:
    #     for t in transfers:
    #         transfer_type = "input" if t.operand.operand in cdlt.inputs else "output"
    #         if transfer_type == "input":
    #             path_key = tuple(t.path[level:2+level])
    #             xfer = t.transfers[path_key]
    #             start_size = size_from_extent(cdlt, xfer._src_offset)
    #             constraints = tile_constraints[path_key]
    #             hag.data_transfer_constraints(*path_key)
    #             loops = []
    #             # for i in range(len(start_size)):
    #             #     if np.prod(start_size[:-i])
    #             for o in xfer._src_offset:
    #                 if isinstance(o, Basic):
    #                     indices = list(o.atoms(Idx))
    #                     loops.append(indices)
    #
    #
    #                 # else:
    #                 #     raise RuntimeError(f"Cant handle scalar offsets currently")
    #
    #         else:
    #             path_key = tuple(t.path[len(t.path) - 2 - level:len(t.path) - level])
    #             xfer = t.transfers[path_key]
    #             start_size = size_from_extent(cdlt, xfer._dst_offset)
    #             constraints = tile_constraints[path_key]
    #
    #             loops = []
    #             for o in xfer._dst_offset:
    #                 if isinstance(o, Basic):
    #                     indices = list(o.atoms(Idx))
    #                     loops.append(indices)

    # TESTING
    # temp_tiling = {}
    # static_shapes = cdlt.get_operand_shapes()
    # for level, nodes in cdlt.tile_levels.items():
    #     for arch_node in nodes:
    #         temp_tiling[arch_node] = static_shapes
    # # TESTING
    # # TODO: Add another level of memory to test second level of tiling
    #
    # for level, nodes in list(cdlt.tile_levels.items()):
    #     for domain_key, size in static_shapes.items():
    #         # TODO: For now, no splits are applied
    #         if domain_key == "OC":
    #             cdlt.set_domain_tile(level, domain_key, 2)
    #         elif domain_key == "OH" and size % 2 == 0:
    #             cdlt.set_domain_tile(level, domain_key, 2)
    #         else:
    #             cdlt.set_domain_tile(level, domain_key, 1)

    # 1. go through loops and apply splits
    # 2. go through transfers and apply splits (computing size)
    # 3. while going through transfers, apply splits to operands
    # for expr in cdlt.operands:
    #     for idx in range(len(expr.evaluated_tiling)):
    #         tile_level, tiling = expr.evaluated_tiling[idx]
    #         if len(tiling) != len(expr.shape_list):
    #             assert idx != 0
    #
    #             expr.evaluated_tiling[idx] = (tile_level, cdlt.domain_tiling[tile_level][])

    return cdlt

def get_tile_constraints(cdlt: Codelet, hag: ArchitectureNode):
    path_constraints = {}
    transfers = [o for o in cdlt.ops if isinstance(o, Transfer)]
    for xfer in transfers:
        for path, xfer_info in xfer.transfers.items():
            src_name, dst_name = path
            if (src_name, dst_name) in path_constraints:
                continue
            src_node = hag.get_subgraph_node(src_name)
            dst_node = hag.get_subgraph_node(dst_name)
            edge = hag.get_subgraph_edge(src_name, dst_name)

            # TODO: Need a better way to characterize constraints here
            if isinstance(dst_node, ComputeNode):
                assert isinstance(src_node, StorageNode)
                max_size = edge.bandwidth
                min_size = edge.bandwidth
            elif isinstance(dst_node, StorageNode):
                if isinstance(src_node, ComputeNode):
                    max_size = min_size = edge.bandwidth
                else:
                    assert isinstance(src_node, StorageNode)
                    max_size = dst_node.size
                    min_size = edge.bandwidth
            else:
                raise TypeError(f"Unable to handle architecture node type {type(dst_node)}")
            path_constraints[(src_name, dst_name)] = (min_size, max_size)
    return path_constraints
