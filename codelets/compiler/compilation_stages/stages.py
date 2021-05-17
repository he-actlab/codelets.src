from typing import TYPE_CHECKING

from codelets.templates.operand_template import IndexOperandTemplate, OperandTemplate
from codelets.templates.operation_template import OperationTemplate
from codelets.templates.codelet_template import CodeletTemplate
from codelets.codelet_impl import Codelet
from codelets.compiler.program import CodeletProgram
from collections import defaultdict

from .tiling_utils import set_codelet_tiling
from .stage_utils import default_tile_heuristic, \
    update_shape_from_arch, store_tile_checkpoint, \
    find_node_key, insert_simd_typecast
import polymath as pm
import json

SYSTOLIC_ARRAY_CDLTS = ['conv_bias', 'conv', 'gemm', 'gemm_no_bias']

# TODO: Update SIMD_CDLTS for dtypes
SIMD_CDLTS = ['max_pool', 'elem_add', 'relu', 'global_avg_pool', 'batch_normalization',
              'sgd4', 'elem_add_grad', 'sgd4d', 'elem_tanh', 'avg_pool']
POOL_OPS = ['max_pool', 'global_avg_pool', 'avg_pool']
BINARY_SIMD = ['elem_add', 'sgd4d', 'elem_add_grad', 'global_average_pool_grad', 'relu_grad', 'elem_tanh_grad',
               'sgd4d', 'max_pool_grad', 'average_pool_grad']

UNARY_SIMD = ['relu', 'max_pool', 'global_avg_pool', 'elem_tanh', 'avg_pool', 'elem_tanh2d']
NOOPS = ['coarse_flatten']
STANDARD_SHAPE_OPS = ['elem_add', 'relu', 'global_avg_pool', 'batch_norm', 'sgd4d',
                      'max_pool_grad', 'global_average_pool_grad', 'relu_grad', 'elem_add_grad', 'elem_tanh_grad',
                      'average_pool_grad']
INFERENCE_OPS = [""]

INTERMEDIATE_INPUT_INDICES = {
    "conv": [0],
    "conv_bias": [0],
    "relu": [0],
    "elem_tanh": [0],
    "elem_tanh2d": [0],
    "max_pool": [0],
    "avg_pool": [0],
    "global_avg_pool": [0],
    "batch_normalization": [0],
    "elem_add": [0, 1],
    "cross_entropy_loss": [0, 1],
    "cross_entropy_loss_grad": [0, 1, 2],
    "gemm": [0]
}

SA_OPS = ["conv", "conv_bias", "gemm", "gemm_bias"]

TRANSPOSED_SHAPES = [['N', 'C', 'H', 'W'], ['N', 'IC', 'IH', 'IW'],
                     ['N', 'C', 'IH', 'IW'], ['N', 'OC', 'OH', 'OW'],
                     ['ON', 'OC', 'OH', 'OW'], ['N', 'C', 'OH', 'OW']]
TRANSPOSE_PERM = [0, 2, 3, 1]
TRANSPOSE_POS = [0, 3, 1, 2]
FLIP_SHAPE_PERM = [2, 3, 0, 1]
FLIP_SHAPES = [['OC', 'IC', 'KH', 'KW']]

def update_operand_dtypes(program: 'CodeletProgram', node: pm.Node, cdlt: 'Codelet', dtype_map=None) -> 'Codelet':
    if cdlt.op_name in SYSTOLIC_ARRAY_CDLTS:
        cdlt.inputs[0].set_dtype(dtype_map['SYSTOLIC_ARRAY']['inp_weight'])
        cdlt.inputs[1].set_dtype(dtype_map['SYSTOLIC_ARRAY']['inp_weight'])
        if len(cdlt.inputs) == 3:
            cdlt.inputs[2].set_dtype(dtype_map['SYSTOLIC_ARRAY']['bias_out'])
        cdlt.outputs[0].set_dtype(dtype_map['SYSTOLIC_ARRAY']['bias_out'])
    else:
        for o in cdlt.operands:
            o.set_dtype(dtype_map['SIMD'])
    return cdlt

def template_pad_pass(program, template: 'CodeletTemplate') -> 'CodeletTemplate':
    if not isinstance(template, CodeletTemplate):
        return template
    else:
        if template.op_name in ["avg_pool", 'global_avg_pool']:
            template.update_dummy_op('denom', template.node.inputs[0].shape[1]*template.node.inputs[0].shape[2])

        if template.op_name == "mean_var":
            template.update_dummy_op('denom', template.node.inputs[0].shape[0]*template.node.inputs[0].shape[1]*template.node.inputs[0].shape[2])

        if template.op_name in ["conv", "conv_bias"]:
            template.update_dummy_op('IH', template.node.inputs[0].shape[1] + 2*template.node.kwargs['pad'])
            template.update_dummy_op('IW', template.node.inputs[0].shape[2] + 2*template.node.kwargs['pad'])

        if template.op_name in SA_OPS:
            inp_constr = template.hag.all_subgraph_nodes['pe_array'].dimensions[0]
            out_constr = template.hag.all_subgraph_nodes['pe_array'].dimensions[1]

            inp_dim = template.inputs[0].shape_list[-1]
            out_dim = template.outputs[0].shape_list[-1]
            dummy_inp_dim = template.node.inputs[0].shape[-1]
            dummy_out_dim = template.node.outputs[0].shape[-1]
            template.update_dummy_op(inp_dim.name, dummy_inp_dim + (inp_constr - dummy_inp_dim) % inp_constr)
            template.update_dummy_op(out_dim.name, dummy_out_dim + (out_constr - dummy_out_dim) % out_constr)
        else:
            constr = template.hag.all_subgraph_nodes['SIMD'].dimensions[0]
            updated_dims = []
            for idx, i in enumerate(template.inputs):
                if i.shape_list_names[-1] not in updated_dims:
                    dim = i.shape_list[-1]
                    dummy_dim = template.node.inputs[idx].shape[-1]
                    template.update_dummy_op(dim.name, dummy_dim + (constr - dummy_dim) % constr)
                    updated_dims.append(dim.name)

            for idx, o in enumerate(template.outputs):
                if o.shape_list_names[-1] not in updated_dims:
                    dim = o.shape_list[-1]
                    dummy_dim = template.node.outputs[idx].shape[-1]
                    template.update_dummy_op(dim.name, dummy_dim + (constr - dummy_dim) % constr)
                    updated_dims.append(dim.name)

            # inp_dim = template.inputs[0].shape_list[-1]
            # out_dim = template.outputs[0].shape_list[-1]
            # dummy_inp_dim = template.node.inputs[0].shape[-1]
            # dummy_out_dim = template.node.outputs[0].shape[-1]
            # template.update_dummy_op(inp_dim.name, dummy_inp_dim + (constr - dummy_inp_dim) % constr)
            # template.update_dummy_op(out_dim.name, dummy_out_dim + (constr - dummy_out_dim) % constr)

        return template

def template_layout_pass(program, template: 'CodeletTemplate') -> 'CodeletTemplate':
    if not isinstance(template, CodeletTemplate):
        return template
    else:
        reordered_operands = {}
        updated_ops = []
        for idx, i in enumerate(template.inputs):
            if i.shape_list_names in TRANSPOSED_SHAPES:
                for sidx, s in enumerate(i.shape_list):
                    if s.name not in updated_ops:
                        template.update_dummy_op(s.name, template.node.inputs[idx].shape[TRANSPOSE_POS[sidx]])
                        updated_ops.append(s.name)
                i.reorder_shapes(TRANSPOSE_PERM)
                reordered_operands[i.name] = TRANSPOSE_PERM
            elif i.shape_list_names in FLIP_SHAPES:
                for sidx, s in enumerate(i.shape_list):
                    if s.name not in updated_ops:
                        template.update_dummy_op(s.name, template.node.inputs[idx].shape[FLIP_SHAPE_PERM[sidx]])
                        updated_ops.append(s.name)

                i.reorder_shapes(FLIP_SHAPE_PERM)
                reordered_operands[i.name] = FLIP_SHAPE_PERM

        for idx, o in enumerate(template.outputs):
            if o.shape_list_names in TRANSPOSED_SHAPES:
                for sidx, s in enumerate(o.shape_list):
                    if s.name not in updated_ops:
                        template.update_dummy_op(s.name, template.node.outputs[idx].shape[TRANSPOSE_POS[sidx]])
                        updated_ops.append(s.name)
                o.reorder_shapes(TRANSPOSE_PERM)
                reordered_operands[o.name] = TRANSPOSE_PERM
            elif o.shape_list_names in FLIP_SHAPES:
                for sidx, s in enumerate(o.shape_list):
                    if s.name not in updated_ops:
                        template.update_dummy_op(s.name, template.node.outputs[idx].shape[FLIP_SHAPE_PERM[sidx]])
                        updated_ops.append(s.name)
                o.reorder_shapes(FLIP_SHAPE_PERM)
                reordered_operands[o.name] = FLIP_SHAPE_PERM


        for o in template.ops:
            if o.op_type == 'transfer':
                operand = o.param_map['operand']
                if isinstance(operand, IndexOperandTemplate) and operand.name in reordered_operands:
                    operand.reorder_offsets(reordered_operands[operand.name])

            elif o.op_type == 'compute':
                for iop in o.param_map['sources']:
                    if isinstance(iop, IndexOperandTemplate) and iop.name in reordered_operands:
                        iop.reorder_offsets(reordered_operands[iop.name])

                for oop in o.param_map['dests']:
                    if isinstance(oop, IndexOperandTemplate) and oop.name in reordered_operands:
                        oop.reorder_offsets(reordered_operands[oop.name])

        return template

def add_simd_typecast(program: 'CodeletProgram', node: pm.Node, cdlt: 'Codelet', dtype_map=None, codelet_output_map=None) -> 'Codelet':
    if cdlt.is_noop():
        output_key = node.outputs[0].name
        input_key = node.inputs[0].name

        if input_key not in dtype_map:
            input_key = find_node_key(node.inputs[0], dtype_map)

        dtype_map[output_key] = dtype_map[input_key]
        codelet_output_map[output_key] = (cdlt.op_name, cdlt.instance_id)
        insert_simd_typecast(program, node, cdlt.inputs[0], cdlt, dtype_map, codelet_output_map, input_key)

    else:
        for idx, operand in enumerate(cdlt.inputs):
            i = node.inputs[idx]
            if not isinstance(i, (pm.input, pm.state)):
                i_key = i.name
                if i_key not in dtype_map:
                    i_key = find_node_key(i, dtype_map)
                    dtype_map[i.name] = dtype_map[i_key]
                    codelet_output_map[i.name] = codelet_output_map[i_key]
                insert_simd_typecast(program, node, operand, cdlt, dtype_map, codelet_output_map, i_key)
            else:
                dtype_map[i.name] = cdlt.get_operand_by_node_name(i.name).dtype
                codelet_output_map[i.name] = (cdlt.op_name, cdlt.instance_id)

        # for o in node.outputs:
        for idx in range(len(cdlt.outputs)):
            o = node.outputs[idx]
            dtype_map[o.name] = cdlt.get_operand_by_node_name(o.name).dtype
            codelet_output_map[o.name] = (cdlt.op_name, cdlt.instance_id)

    return cdlt


def pad_operands(program: 'CodeletProgram', node: pm.Node, cdlt: 'Codelet', shaped_nodes=None) -> 'Codelet':
    assert isinstance(shaped_nodes, dict)

    return cdlt

def tile(program: 'CodeletProgram', node: pm.Node, cdlt: 'Codelet', factor_fn_name='default', heuristic_fn=None, checkpoint_file=None) -> 'Codelet':
    hag = program.hag

    cdlt.set_tile_levels()
    heuristic_fn = heuristic_fn or default_tile_heuristic
    # Find amount of splits for each loop by looking at dependencies
    loop_splits = {}

    for i, o in enumerate(cdlt.operands):
        if len(o.dependencies) == 0 and len(o.data_path) == 0:
            continue
        loops = [d for d in o.dependencies if "loop" in d]
        max_level = max(cdlt.get_tile_level(dp) for dp in o.data_path)
        for l in loops:
            if l in loop_splits and loop_splits[l] < max_level:
                loop_splits[l] = max_level
            else:
                loop_splits[l] = max_level
    bands = cdlt.extract_bands()

    cdlt = set_codelet_tiling(cdlt, hag, factor_fn_name)
    outer_loop_map = {}
    loop_replacement_map = {}
    for start, end in bands:
        idx = start
        splits = loop_splits[cdlt.ops[idx].op_str] - 1
        llevels = [o.loop_level for o in cdlt.ops[start: end + 1]]
        max_level = max(llevels)
        min_level = min(llevels)
        dep_mapping = {}
        for split in range(splits):
            op_band = cdlt.ops[start: end + 1]

            offset = (end - start)
            num_splits = 0
            for op in op_band:
                i = cdlt.ops.index(op)
                target_idx = offset + i

                inner_loop_level = (max_level - min_level) + op.loop_level
                if inner_loop_level < op.loop_level:
                    raise RuntimeError

                inner_deps = [dep_mapping[dp] for dp in op.dependencies]
                new_op_id, new_global_id = cdlt.get_new_op_ids(op)
                extra_kwargs = {}

                if op.op_type == 'transfer':
                    if len(op.path) <= 2:
                        dep_mapping[op.op_str] = op.op_str
                        outgoing = False

                        offset -= 1
                        if cdlt.get_tile_level(op.path[0]) > cdlt.get_tile_level(op.path[1]):
                            offset += 1
                            outgoing = True
                            cdlt.insert_op(op, target_idx)
                        op.operand.update_transfer_access(op, outgoing=outgoing)
                        continue
                    elif cdlt.get_tile_level(op.path[0]) > cdlt.get_tile_level(op.path[1]):
                        outgoing = True
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
                        op.operand.update_transfer_access(inner_op, outgoing=outgoing)

                        inner_idx = target_idx
                        dep_mapping[op.op_str] = inner_op.op_str

                        # Update outer op
                        op._dependencies.append(inner_op.op_str)
                        cdlt.insert_op(op, target_idx)
                    else:
                        outgoing = False
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

                        op.operand.update_transfer_access(inner_op, outgoing)

                        inner_idx = target_idx + 1
                        dep_mapping[op.op_str] = inner_op.op_str

                    num_splits += 1

                elif op.op_type == 'loop':


                    extra_kwargs['start'] = 0
                    extra_kwargs['end'] = cdlt.domain_loop_map[split + 1][op.op_str]
                    extra_kwargs['stride'] = 1

                    inner_op = op.copy(cdlt, loop_level=inner_loop_level,
                                                    op_id=new_op_id,
                                                    loop_id=new_op_id,
                                                    global_op_id=new_global_id,
                                                    dependencies=inner_deps, **extra_kwargs)
                    cdlt._domain_loop_map[split+1][inner_op.op_str] = cdlt.domain_loop_map[split + 1][op.op_str]
                    op.start = 0
                    op.stride = cdlt.domain_loop_map[split + 1][op.op_str]
                    op.end = cdlt.domain_loop_map[split][op.op_str]
                    cdlt._domain_loop_map[split+1].pop(op.op_str)

                    dep_mapping[op.op_str] = inner_op.op_str
                    inner_idx = target_idx + 1
                    num_splits += 1
                    cdlt.loop_param_map[inner_op.op_str] = cdlt.loop_param_map[op.op_str]

                    if cdlt.loop_param_map[op.op_str] not in outer_loop_map:
                        outer_loop_map[cdlt.loop_param_map[op.op_str]] = op.op_str
                else:
                    assert op.op_type == 'compute', f"Invalid op type: {op.op_type}"
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
                if op.op_type == "loop" and outer_loop_map[cdlt.loop_param_map[op.op_str]] != op.op_str:
                    old_op = cdlt.ops.pop(cdlt.ops.index(op))
                    loop_replacement_map[old_op.op_str] = outer_loop_map[cdlt.loop_param_map[op.op_str]]

    for op in cdlt.ops:
        new_deps = []
        for d in op.dependencies:
            if d in loop_replacement_map:
                new_deps.append(loop_replacement_map[d])
            else:
                new_deps.append(d)
        op._dependencies = new_deps

    for o in cdlt.operands:
        if len(o.data_moves) > 0 and o.data_moves[-1].dst_node not in o.tiling:

            last_move = o.data_moves[-1]
            dest_name = last_move.dst_node
            level = cdlt.get_tile_level(dest_name)
            level_sizes = cdlt.domain_loop_map[level]
            o.tiling[dest_name] = last_move.get_size_from_loops(cdlt, level_sizes)

        if o in cdlt.outputs and not o.is_tiled():
            missing_tiles = [l for l in o.unique_data_locations() if l not in list(o.tiling.keys())]
            for m in missing_tiles:
                level = cdlt.get_tile_level(m)
                level_sizes = cdlt.domain_loop_map[level]
                mmove = None
                for a in o.data_moves:
                    if a.src_node == m:
                        mmove = a
                        break
                prev_level = level
                if mmove is None:
                    raise RuntimeError(f"UNable to find movement for missing tile {m}\n"
                                       f"Moves: {o.movement_keys()}")
                o.tiling[m] = mmove.get_size_from_loops(cdlt, level_sizes)

        if not o.is_tiled():
            raise RuntimeError(f"Number of tilings does not match the data path size for {o.name}:\n"
                               f"Tiling keys: {list(o.tiling.keys())}\n"
                               f"Unique data path locations: {o.unique_data_locations()}\n"
                               f"Data path: {o.data_path}")
    if checkpoint_file is not None:
        store_tile_checkpoint(cdlt, checkpoint_file)

    return cdlt


def hoist(program, node: pm.Node, cdlt: 'Codelet') -> 'Codelet':
    for o in cdlt.ops:
        if o.op_type == "loop":
            continue
        i = cdlt.ops.index(o)
        all_deps = o.dependencies
        loop_name = cdlt.get_max_loop_dep(o)

        if loop_name is None:
            idx = i
            loop_level = 0
        elif len(all_deps) == 0:
            idx = cdlt.ops.index(cdlt.op_map[loop_name]) + 1
            loop_level = cdlt.op_map[loop_name].loop_level + 1
        else:
            idx = max([cdlt.ops.index(cdlt.op_map[dep]) for dep in all_deps])
            loop_level = cdlt.op_map[loop_name].loop_level + 1


        if (idx + 1) < i and cdlt.ops[idx + 1].loop_level <= loop_level:
            idx += 1
            cdlt.ops.insert(idx, cdlt.ops.pop(i))
            cdlt.ops[idx].loop_level = loop_level

    return cdlt


