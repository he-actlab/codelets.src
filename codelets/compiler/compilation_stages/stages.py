
from codelets.adl import Codelet
from codelets.adl.operation import Transfer, Loop, Compute
from .stage_utils import default_tile_heuristic, set_codelet_tiling
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
                        op.operand.update_op_accesses(cdlt, op, dep_mapping)
                        op.operand.update_transfer_access(op)

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
                        op.operand.update_op_accesses(cdlt, inner_op, dep_mapping)
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
                        op.operand.update_op_accesses(cdlt, op, dep_mapping)

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
                    assert isinstance(op, Compute)
                    dep_mapping[op.op_str] = op.op_str
                    op.dependencies = inner_deps
                    op.loop_level = inner_loop_level
                    inner_op = op
                    for s in op.sources:
                        s.update_op_accesses(cdlt, inner_op, dep_mapping)
                        s.compute_tile(op, "source")

                    for d in op.dests:
                        d.update_op_accesses(cdlt, inner_op, dep_mapping)
                        d.compute_tile(op, "dest")

                    inner_idx = target_idx
                    num_splits += 1
                cdlt.insert_op(inner_op, inner_idx)

    cdlt = program.codelets[0]
    for o in cdlt.operands:
        if len(o.data_moves) > 0 and o.data_moves[-1].dst_node not in o.tiling:
            last_move = o.data_moves[-1]
            dest_name = last_move.dst_node
            level = cdlt.get_tile_level(dest_name)
            level_sizes = cdlt.domain_loop_map[level]
            o.tiling[dest_name] = last_move.get_size_from_loops(cdlt, level_sizes)

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


def insert_dtype_cast(program, n, cdlt):
    pass


