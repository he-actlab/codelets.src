from typing import TYPE_CHECKING
from collections import defaultdict, deque
from itertools import product
from pytools import memoize
from pathlib import Path
import json
if TYPE_CHECKING:
    from codelets.adl import ArchitectureNode, StorageNode, ComputeNode
    from codelets.codelet_impl import Codelet
from codelets.adl.flex_param import FlexParam

from . import TilingInfo
from codelets.compiler.transformations import factors, factors_rand_sort,\
    factors_reversed, get_sorted_perms
import numpy as np

FACTOR_FN_MAP = {'default': factors, 'random': factors_rand_sort, 'reversed': factors_reversed}

@memoize
def get_sizes_from_splits(loops, shapes, splits):
    out_shapes = []

    for i, l in enumerate(loops):
        out_shapes.append(shapes[i] // splits[i])

    return tuple(out_shapes)

def get_level_tiling(cdlt, loop_dependencies, shapes, splits, factor_fn):
    out_shapes = {}
    out_factors = {}

    if 'fixed_tile_dims' in cdlt.compilation_params:
        fixed_dims = cdlt.compilation_params['fixed_tile_dims']
    else:
        fixed_dims = []


    for l in loop_dependencies:
        out_shapes[l] = shapes[l] // splits[l]

        if cdlt.domain_loop_map[l] in fixed_dims:
            out_factors[l] = [1]
        else:
            out_factors[l] = factor_fn(out_shapes[l])

    perms = product(*tuple(out_factors.values()))
    # Need to skip past the first tiling because its all 1's
    # next(perms)
    return out_shapes, out_factors, perms

def default_tile_heuristic(hag: 'ArchitectureNode', cdlt: 'Codelet', tiling_splits):
    total_accesses = 0
    for l, splits in tiling_splits.items():
        for _, s in splits.items():
            total_accesses += s
    return total_accesses

def update_shape_from_arch(node, shaped_nodes, arch_constraint, dim_index,
                           layout_nhwc=False, force_reshape=False):

    if node.name not in shaped_nodes or force_reshape:
        if node.shape[dim_index] % arch_constraint != 0:
            dim_shape = node.shape[dim_index] + (arch_constraint - (node.shape[dim_index] % arch_constraint))
        else:
            dim_shape = node.shape[dim_index]
        new_shape = list(node.shape)
        new_shape[dim_index] = dim_shape
        if layout_nhwc:
            node.shape = (new_shape[0], new_shape[2], new_shape[3], new_shape[1])
            new_shape = list(node.shape)
        else:
            node.shape = tuple(new_shape)
        shaped_nodes[node.name] = node.shape
    elif shaped_nodes[node.name] != node.shape:
        new_shape = list(shaped_nodes[node.name])
        node.shape = tuple(new_shape)
    else:
        new_shape = list(node.shape)

    return tuple(new_shape)


def find_tiling(cdlt, level, perm_stack):
    if level > list(cdlt.tile_levels.keys())[-1] or level <= 0:
        return level

    prev_level = level - 1
    perms = perm_stack[prev_level]
    assert perms is not None
    valid_splits = None

def store_tile_checkpoint(cdlt, checkpoint_path):
    abs_path = Path(checkpoint_path).absolute()
    if abs_path.exists():
        with open(f'{abs_path}') as f:
            tiling = json.load(f)
    else:
        tiling = {}
    tile_key = f"{cdlt.op_name}{cdlt.instance_id}"
    tiling[tile_key] = cdlt.domain_tiling

    with open(f'{abs_path}', "w") as outfile:
        json.dump(tiling, outfile, indent=4)

def set_codelet_tiling(cdlt: 'Codelet', hag: 'ArchitectureNode', factor_fn_name):
    # TODO: Try to look ahead and see if all paths lead to node, in which case
    # we can add additional constraints to the first level
    if factor_fn_name is not None:
        factor_fn = FACTOR_FN_MAP[factor_fn_name]
    else:
        factor_fn = FACTOR_FN_MAP['default']

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

    if 'fixed_tile_dims' in cdlt.compilation_params:
        fixed_dims = cdlt.compilation_params['fixed_tile_dims']
    else:
        fixed_dims = []

    for l in loop_dependencies:
        loop = cdlt.op_map[l]
        if cdlt.domain_loop_map[l] in fixed_dims:
            level_factors[0][loop.op_str] = [1]
        else:
            level_factors[0][loop.op_str] = factor_fn(loop.iter_count)

        shapes[0][loop.op_str] = loop.iter_count
        selected_splits[0][loop.op_str] = 1
        accumulated_splits[loop.op_str] = 1
    perm_stack = deque()

    if 0 in cdlt.domain_tiling:
        first_perm = [tuple(cdlt.domain_tiling[0][ld] for ld in loop_dependencies)]
    else:
        first_perm = product(*tuple(level_factors[0].values()))

    perm_stack.append(first_perm)
    max_level = 1
    level = 1
    level_counter = defaultdict(int)

    def find_valid_splits(p, lvl, pperm):
        valid_splits = p

        perm_map = {l: p[i]*accumulated_splits[l] for i, l in enumerate(loop_dependencies)}
        size_map = {}


        for level_access in level_accesses[lvl]:

            size = level_access.get_size_from_splits(cdlt, perm_map)
            key = (level_access.src_node, level_access.dst_node)

            for k, v in size.items():
                if k in size_map and v != size_map[k]:
                    raise RuntimeError(f"Size is not equal to collected sizes for access:\n"
                                       f"Size from splits: {size}\n"
                                       f"Size map: {size_map}\n"
                                       f"Level: {lvl}\n"
                                       f"Key: {key}\n")

                else:
                    size_map[k] = v


            dtype_size = cdlt.get_operand(level_access.operand_name).dtype.bits()
            constraint_sat = tile_constraints.evaluate_constraint(key, size, dtype_size)
            # constraint_sat = tile_constraints[key].evaluate_constraint(size, dtype_size)

            if not constraint_sat:
                valid_splits = None
                break
        return valid_splits
    loop_deps_fixed = tuple(loop_dependencies)
    parent_perms = deque()
    prev_perm = None
    parent_perms.append(prev_perm)
    while level <= list(cdlt.tile_levels.keys())[-1] and level > 0:
        if level > max_level:
            max_level = level
        prev_level = level - 1
        if level in cdlt.domain_tiling:
            perms = [tuple(cdlt.domain_tiling[level][ld] for ld in loop_dependencies)]
        else:
            perms = perm_stack[prev_level]
        assert perms is not None
        valid_splits = None
        fixed_shapes = tuple([shapes[prev_level][l] for l in loop_dependencies])

        for p in perms:
            level_counter[level] += 1
            perm_shapes = get_sizes_from_splits(loop_deps_fixed, fixed_shapes, p)
            passes_hint = tile_constraints.check_tile_hints(level, loop_deps_fixed, perm_shapes, p)
            if not passes_hint:
                continue
            valid_splits = find_valid_splits(p, level, prev_perm)

            if valid_splits:
                prev_perm = p
                valid_splits = {list(level_factors[level - 1].keys())[i]: v for i, v in enumerate(valid_splits)}
                break

        if not valid_splits:
            prev_perm = parent_perms.pop()
            perm_stack.pop()
            shapes.pop(prev_level)
            level_factors.pop(prev_level)
            prev_splits = selected_splits.pop(prev_level)
            accumulated_splits = {k: v//prev_splits[k] for k, v in accumulated_splits.items()}
            level -= 1
        else:
            parent_perms.append(prev_perm)
            selected_splits[level] = valid_splits.copy()
            accumulated_splits = {k: v*selected_splits[level][k] for k, v in accumulated_splits.items()}
            shapes[level], level_factors[level], new_perms = get_level_tiling(cdlt, loop_dependencies, shapes[prev_level], valid_splits, factor_fn)
            perm_stack.append(new_perms)
            level += 1

    if level == 0:
        raise RuntimeError(f"Unable to find adequate tiling for Codelet:"
                       f"Codelet Dimensions: {cdlt.operand_dim_mapping()}\n"
                       f"Max level reached: {max_level}\n"
                           f"Times per level: {level_counter}\n"
                       f"Op: {cdlt.op_name}{cdlt.instance_id}\n"
                       f"constraints:{[(k, t.fn_body_str) for k, t in tile_constraints.constraint_fps.items()]}\n")
    # Lastly, update operands
    for o in cdlt.operands:

        for idx, a in enumerate(o.data_moves):
            if all(a in [None, 0] for a in list(a.offset_map.values())):
                assert idx > 0
                a.reinit_offset_map(o.data_moves[idx - 1].offset_map.copy())
                # a.offset_map = o.data_moves[idx - 1].offset_map.copy()

            if len(a.shape_map) == 0:
                a.set_size_from_splits(cdlt, selected_splits)

            a.set_offset_map(cdlt, shapes)

    # TODO: Store all information int he codelet
    cdlt._domain_tiling = selected_splits
    cdlt._domain_loop_map = shapes


    return cdlt



# TODO: THis needs to return a list of functions with the same function signature
def get_tile_constraints(cdlt: 'Codelet', hag: 'ArchitectureNode'):
    path_constraints = {}
    tile_info = TilingInfo(f"{cdlt.op_name}{cdlt.instance_id}_tile_info",
                           cdlt.domain_loop_map, len(list(cdlt.tile_levels.keys())))

    for o in cdlt.operands:
        for access in o.data_moves:
            if (access.src_node, access.dst_node) in path_constraints or access.src_node == access.dst_node:
                continue

            src_node = hag.get_subgraph_node(access.src_node)
            dst_node = hag.get_subgraph_node(access.dst_node)
            edge = hag.get_subgraph_edge(access.src_node, access.dst_node)
            if dst_node.node_type == 'compute':
                constraint = f"size == {edge.bandwidth}"
                assert src_node.node_type == 'storage'
                # TODO: Need to add something which adds padding function here and uses a function constraint
            elif dst_node.node_type == 'storage':
                if src_node.node_type == 'compute':
                    constraint = f"size == {edge.bandwidth}"
                else:
                    assert src_node.node_type == 'storage'
                    constraint = f"size <= {dst_node.size} and size >= 0"
            else:
                raise TypeError(f"Unable to handle architecture node type {type(dst_node)}")
            level = cdlt.get_tile_level(access.dst_node)
            tile_info.add_constraint(access.src_node, access.dst_node, level, constraint)

    for loop_name, dim_name in cdlt.domain_loop_map.items():
        for level in cdlt.tile_levels.keys():
            tile_hint_key = f"{dim_name}_hint{level}"
            if tile_hint_key in cdlt.compilation_params:
                tile_info.add_tile_hint(level, loop_name, cdlt.compilation_params[tile_hint_key])

    for level in cdlt.tile_levels.keys():
        level_hint_key = f"LEVEL{level}_hint"
        if level_hint_key in cdlt.compilation_params:
            tile_info.add_level_hint(level, cdlt.compilation_params[level_hint_key])

    return tile_info
