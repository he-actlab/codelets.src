from collections import defaultdict, deque
from itertools import chain, combinations, product
from typing import List
from pytools import memoize, memoize_on_first_arg

from codelets.adl import Codelet, ComputeNode
from codelets.adl.flex_param import FlexParam
from codelets.adl.graph import ArchitectureNode, StorageNode
from codelets.adl.operation import Compute, OperandTemplate, Operation, Transfer, Loop
from codelets.compiler.transformations import split_operation, factors
import numpy as np
import polymath as pm


def tile(program, node: pm.Node, cdlt: Codelet, heuristic_fn=None) -> Codelet:
    hag = program.hag
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

                if isinstance(cdlt.ops[target_idx], Loop):
                    inner_loop_level = cdlt.ops[target_idx].loop_level + 1
                else:
                    inner_loop_level = cdlt.ops[target_idx].loop_level

                if inner_loop_level < op.loop_level:
                    raise RuntimeError

                inner_deps = [dep_mapping[dp] for dp in op.dependencies]
                new_op_id, new_global_id = cdlt.get_new_op_ids(op)
                extra_kwargs = {}

                if isinstance(op, Transfer):
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
                        extra_kwargs["operand"] = op.operand
                        inner_op = cdlt.ops[i].copy(cdlt, loop_level=inner_loop_level,
                                                    op_id=new_op_id,
                                                    global_op_id=new_global_id,
                                                    dependencies=inner_deps, **extra_kwargs)
                        assert id(op.operand) == id(inner_op.operand)
                        op.operand.update_transfer_access(inner_op)

                        inner_idx = target_idx
                        dep_mapping[op.op_str] = inner_op.op_str

                        # Update outer op
                        op._dependencies.append(inner_op.op_str)
                        cdlt.insert_op(op, target_idx)

                    else:
                        outer_path, inner_path = op.path[split: split + 2], op.path[split + 1:]
                        op._path = outer_path
                        extra_kwargs["path"] = inner_path
                        extra_kwargs["operand"] = op.operand
                        inner_deps.append(op.op_str)
                        inner_op = op.copy(cdlt, loop_level=inner_loop_level,
                                                    op_id=new_op_id,
                                                    global_op_id=new_global_id,
                                                    dependencies=inner_deps, **extra_kwargs)
                        assert id(op.operand) == id(inner_op.operand)

                        op.operand.update_transfer_access(inner_op)
                        inner_idx = target_idx + 1
                        dep_mapping[op.op_str] = inner_op.op_str

                    num_splits += 1
                elif isinstance(op, Loop):

                    extra_kwargs['start'] = 0
                    extra_kwargs['end'] = cdlt.domain_loop_map[split + 1][op.op_str]
                    extra_kwargs['stride'] = 1

                    inner_op = op.copy(cdlt, loop_level=inner_loop_level,
                                                    op_id=new_op_id,
                                                    global_op_id=new_global_id,
                                                    dependencies=inner_deps, **extra_kwargs)

                    op.start = 0
                    op.stride = cdlt.domain_loop_map[split + 1][op.op_str]
                    op.end = cdlt.domain_loop_map[split][op.op_str]

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


def hoist(program, node: pm.Node, cdlt: Codelet) -> Codelet:
    for o in cdlt.ops:
        i = cdlt.ops.index(o)
        i_loop_level = o.loop_level
        idx = -1
        loop_level = -1
        for dep in o.dependencies:

            dep_idx = cdlt.ops.index(cdlt.op_map[dep])
            if cdlt.ops[dep_idx].op_type == "loop":
                dep_level = cdlt.ops[dep_idx].loop_level + 1
            else:
                dep_level = cdlt.ops[dep_idx].loop_level

            if dep_level > loop_level:
                loop_level = dep_level

            if dep_idx > idx:
                idx = dep_idx

        if idx < 0:
            idx = i

        if idx < i:
            cdlt.ops.insert(idx + 1, cdlt.ops.pop(i))
            idx += 1

        if loop_level < i_loop_level and loop_level > 0:
            cdlt.ops[idx].loop_level = loop_level

    return cdlt

def instantiate_codelet(program, node: pm.Node, cdlt: Codelet):
    cdlt = program.hag.get_codelet_template(node.op_name)
    program.add_codelet(cdlt)
    return cdlt

def get_level_tiling(loop_dependencies, shapes, splits):
    out_shapes = {}
    out_factors = {}
    for l in loop_dependencies:
        out_shapes[l] = shapes[l] // splits[l]
        out_factors[l] = factors(out_shapes[l])
    perms = product(*tuple(out_factors.values()))
    # Need to skip past the first tiling because its all 1's
    next(perms)
    return out_shapes, out_factors, perms

def default_tile_heuristic(hag: ArchitectureNode, cdlt: Codelet, tiling_splits):
    total_accesses = 0
    for l, splits in tiling_splits.items():
        for _, s in splits.items():
            total_accesses += s
    return total_accesses

def set_codelet_tiling(cdlt: Codelet, hag: ArchitectureNode, heuristic_fn):
    # TODO: Try to look ahead and see if all paths lead to node, in which case
    # we can add additional constraints to the first level
    tile_constraints = get_tile_constraints(cdlt, hag)
    level_accesses = defaultdict(list)
    loop_dependencies = []
    # Collect accesses and loop dependencies
    for o in cdlt.operands:
        for i, access in enumerate(o.data_moves):
            if access.src_node != access.dst_node:
                level_accesses[cdlt.get_tile_level(access.dst_node)].append(access)
        loop_dependencies += [dp for dp in o.dependencies if dp not in loop_dependencies and "loop" in dp]

    # Find all starting loop factors
    shapes = defaultdict(dict)
    level_factors = defaultdict(dict)
    selected_splits = defaultdict(dict)
    accumulated_splits = {}
    for l in loop_dependencies:
        loop = cdlt.op_map[l]
        level_factors[0][loop.op_str] = factors(loop.iter_count)
        shapes[0][loop.op_str] = loop.iter_count
        selected_splits[0][loop.op_str] = 1
        accumulated_splits[loop.op_str] = 1

    perm_stack = deque()
    perm_stack.append(product(*tuple(level_factors[0].values())))
    level = 1
    @memoize
    def find_valid_splits(p, lvl):
        valid_splits = p

        perm_map = {l: p[i]*accumulated_splits[l] for i, l in enumerate(loop_dependencies)}
        for level_access in level_accesses[lvl]:
            size = level_access.get_size_from_splits(cdlt, perm_map)

            dtype_size = cdlt.get_operand(level_access.operand_name).dtype.bytes()
            total_size = np.prod(list(size.values()))*dtype_size
            constraint_sat = tile_constraints[(level_access.src_node, level_access.dst_node)].evaluate_fn(total_size)
            # min_size, max_size = tile_constraints[(level_access.src_node, level_access.dst_node)]

            if not constraint_sat:
                valid_splits = None
                break
        return valid_splits

    while level <= list(cdlt.tile_levels.keys())[-1] and level > 0:
        prev_level = level - 1
        perms = perm_stack[prev_level]
        assert perms is not None
        valid_splits = None

        for p in perms:
            valid_splits = find_valid_splits(p, level)
            if valid_splits:
                valid_splits = {list(level_factors[level - 1].keys())[i]: v for i, v in enumerate(valid_splits)}
                break

        if not valid_splits:
            perm_stack.pop()
            shapes.pop(prev_level)
            level_factors.pop(prev_level)
            prev_splits = selected_splits.pop(prev_level)
            accumulated_splits = {k: v//prev_splits[k] for k, v in accumulated_splits.items()}
            level -= 1
        else:
            selected_splits[level] = valid_splits.copy()
            accumulated_splits = {k: v*selected_splits[level][k] for k, v in accumulated_splits.items()}
            shapes[level], level_factors[level], new_perms = get_level_tiling(loop_dependencies, shapes[prev_level], valid_splits)
            perm_stack.append(new_perms)
            level += 1

    if level == 0:
        raise RuntimeError(f"Unable to find adequate tiling for Codelet:"
                           f"Codelet Dimensions: {cdlt.operand_dim_mapping()}\n"
                           f"Op: {cdlt.op_name}"
                           f"constraints:{tile_constraints}\n")
    # Lastly, update operands
    for o in cdlt.operands:

        for idx, a in enumerate(o.data_moves):
            if all(a in [None, 0] for a in list(a.offset_map.values())):
                assert idx > 0
                a.offset_map = o.data_moves[idx - 1].offset_map.copy()
            if len(a.shape_map) == 0:
                a.set_size_from_splits(cdlt, selected_splits)
            a.set_offset_map(cdlt, shapes)
    # TODO: Store all information int he codelet

    cdlt._domain_tiling = selected_splits
    cdlt._domain_loop_map = shapes
    return cdlt


# TODO: THis needs to return a list of functions with the same function signature
def get_tile_constraints(cdlt: Codelet, hag: ArchitectureNode):
    path_constraints = {}
    for o in cdlt.operands:
        for access in o.data_moves:
            if (access.src_node, access.dst_node) in path_constraints or access.src_node == access.dst_node:
                continue
            src_node = hag.get_subgraph_node(access.src_node)
            dst_node = hag.get_subgraph_node(access.dst_node)
            edge = hag.get_subgraph_edge(access.src_node, access.dst_node)
            if isinstance(dst_node, ComputeNode):
                constraint = f"size <= {edge.bandwidth} and size >= 0"

                assert isinstance(src_node, StorageNode)
                max_size = edge.bandwidth
                # TODO: Need to add something which adds padding function here and uses a function constraint
                # min_size = edge.bandwidth
                min_size = 0
            elif isinstance(dst_node, StorageNode):
                if isinstance(src_node, ComputeNode):
                    constraint = f"size <= {edge.bandwidth} and size >= 0"

                    max_size = edge.bandwidth
                    # min_size = edge.bandwidth
                    min_size = 0
                else:
                    assert isinstance(src_node, StorageNode)
                    constraint = f"size <= {dst_node.size} and size >= 0"

                    max_size = dst_node.size
                    min_size = 0
                    # min_size = edge.bandwidth
            else:
                raise TypeError(f"Unable to handle architecture node type {type(dst_node)}")
            path_constraints[(access.src_node, access.dst_node)] = FlexParam(f"constraint_{(access.src_node)}_{(access.dst_node)}",
                                                                              ["size"], constraint)


    return path_constraints
