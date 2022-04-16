from pathlib import Path
from typing import Iterable
from benchmarks.model_generator import create_custom_conv, create_custom_gemm,\
    create_custom_matmul, create_custom_layer, create_custom_multi_layer, is_dw_conv,\
    collect_value_info
from benchmarks.load_onnx_model import store_unique_model_layers, convert_model_to_polymath
from examples.genesys.datagen_functions import check_conv_params, compute_im2col_dims
from examples.genesys import compile_genesys_layer, GENESYS_CFG, compile_genesys
from examples.genesys.codelets import FUSION_OP_INFO
from examples.genesys.genesys_network_sim import compile_full_model
from tools.neuroweaver.compile_model import get_onnx_weights
import numpy as np
import os
import json
from collections import defaultdict
import onnx

from compile_layer import store_outputs
OUT_DIR = Path(f"{Path(__file__).parent}/compilation_output")
BENCH_DIR = Path(f"{Path(__file__).parent}/../benchmarks")
MODEL_DIR = Path(f"{Path(__file__).parent}/../benchmarks/models")
LAYER_DIR = Path(f"{Path(__file__).parent}/../benchmarks/layers")
NAME_MAPPING = {
    "conv_bias": "conv",
    "relu2d": "relu",
    "elem_sigmoid": "sigmoid",
    "gemm_no_bias": "gemm",
    "elem_add": "add",
    "elem_ceil2d": "ceil",
    "elem_div": "div",
    "elem_less": "less",
    "elem_equal": "equal",
    "elem_pow2d": "pow",
    "reduce_min2d": "reducemin",
    "reduce_mean2d": "reducemean",
    "elem_mul": "mul",
    "elem_tanh": "tanh",
    "elem_clip": "clip",
    "elem_exp": "exp",
    "elem_tanh2d": "tanh",
    "tensor_transpose2d": "transpose",
    "elem_sub": "sub",
    "leaky_relu": "leakyrelu",
    "avg_pool": "averagepool",
    "max_pool": "maxpool",
    "global_avg_pool": "globalaveragepool",
    "depthwise_conv": "depthwise_conv"
}
BENCH_BASE_ADDR = {"INSTR": 0, "OBUF": 0, "BBUF": 4096, "WBUF": 24576, "IBUF": 4259840}

def compile_custom_multi_layer(model_name, layer_sequence, params, store_compile=False, dir_ext=None,
                              partials=False, added_constr=None, verbose=False, fuse_layers=False):
    create_custom_multi_layer(layer_sequence, params, True, True, False, True, fname=model_name)

    program = compile_full_model(model_name,
                                store_compile=store_compile,
                                dir_ext=dir_ext,
                                 added_constr=added_constr,
                                 train_mode=False,
                                 verbose=verbose,
                                 model_data=None,
                                 fuse_layers=fuse_layers
                              )

    return program

def compile_fusion_model(model_name, layer_sequence, params, store_compile=False, dir_ext=None,
                              partials=False, added_constr=None, verbose=False, fuse_layers=False):
    program = compile_full_model(model_name,
                                 store_compile=store_compile,
                                 dir_ext=dir_ext,
                                 added_constr=added_constr,
                                 train_mode=False,
                                 verbose=verbose,
                                 model_data=None,
                                 fuse_layers=fuse_layers
                                 )

    return program



def compile_custom_layer(model_name, layer_name, params, store_compile=False, dir_ext=None,
                              partials=False, added_constr=None):
    create_custom_layer(layer_name, params, True, True, False, False, fname=model_name)
    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    if layer_name in NAME_MAPPING:
        store_unique_model_layers(model_name, store_as_polymath=True,
                                  name_mapping={NAME_MAPPING[layer_name]: layer_name})
    else:
        store_unique_model_layers(model_name, store_as_polymath=True)
    batch_size = 1
    tile_method = "min_tiles"

    update_cfg_dtypes = False
    tiling_path = None
    store_tiling = False
    store_json_output = False
    json_output_filename = None
    full_layer_name = f"{model_name}_{layer_name}"
    # This function returns
    program = compile_genesys_layer(full_layer_name,
                              update_cfg_dtypes=update_cfg_dtypes,
                              tiling_path=tiling_path,
                              store_tiling=store_tiling,
                              store_checkpoint=False,
                              store_json_output=store_json_output,
                              json_output_filename=json_output_filename,
                              verbose=False,
                              benchmark_path=BENCH_DIR,
                              factor_fn='default',
                            batch_size=batch_size,
                            do_hoist_stage=True,
                            do_tile_stage=True,
                            print_config=False,
                            tiling_search_algorithm=tile_method,
                                    do_compile=False
                                    # relocation_offsets=reloc_offsets
                              )

    if store_compile:
        if added_constr:
            program = update_tile_constraints(program, added_constr, layer_name)
        dir_ext = dir_ext or ''
        # program.compile(verbose=False, finalize_instructions=True)

        store_outputs("cc_layer1", layer_name, False,
                      1,
                      False,
                      None,
                      use_random=True,
                      dir_ext=f"{dir_ext}",
                      actual_data=False,
                      store_partials=partials, program=program)
    return program





def compile_custom_gemm_layer(m, n, p, model_name, store_compile=False, dir_ext=None,
                              partials=False, added_constr=None):

    create_custom_gemm(True, True, False, False, m, n, p, fname=model_name)
    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    store_unique_model_layers(model_name, store_as_polymath=True)

    batch_size = 1
    tile_method = "min_tiles"
    # tile_method = "valid_split"

    update_cfg_dtypes = False
    tiling_path = None
    store_tiling = False
    store_json_output = False
    json_output_filename = None
    layer_name = f"{model_name}_gemm"

    # This function returns
    program = compile_genesys_layer(layer_name,
                              update_cfg_dtypes=update_cfg_dtypes,
                              tiling_path=tiling_path,
                              store_tiling=store_tiling,
                              store_checkpoint=False,
                              store_json_output=store_json_output,
                              json_output_filename=json_output_filename,
                              verbose=False,
                              benchmark_path=BENCH_DIR,
                              factor_fn='default',
                            batch_size=batch_size,
                            do_hoist_stage=True,
                            do_tile_stage=True,
                            print_config=False,
                            tiling_search_algorithm=tile_method,
                                    do_compile=False
                                    # relocation_offsets=reloc_offsets
                              )
    if store_compile:
        if added_constr:
            program = update_tile_constraints(program, added_constr, "gemm")
        dir_ext = dir_ext or ''
        program.compile(verbose=False, finalize_instructions=True)

        store_outputs("cc_layer1", "gemm", False,
                      1,
                      False,
                      None,
                      use_random=True,
                      dir_ext=f"{dir_ext}",
                      actual_data=False,
                      store_partials=partials, program=program)
    return program

def compile_custom_conv_layer(n, ic, oc, ih, iw, k, stride, pad, model_name, store_compile=False, dir_ext=None,
                              partials=False, added_constr=None):
    check_conv_params(n, ic, oc, ih, iw, k, stride, pad)
    input_shape = (n, ic, ih, iw)
    # model_name = f"resnet50_{name_postfix}"

    create_custom_conv(True, True, False, False, input_shape, oc, k, stride, pad, name=model_name)
    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    store_unique_model_layers(model_name, store_as_polymath=True)

    batch_size = 1
    tile_method = "min_tiles"
    # tile_method = "valid_split"

    update_cfg_dtypes = False
    tiling_path = None
    store_tiling = False
    store_json_output = False
    json_output_filename = None
    layer_name = f"{model_name}_conv"

    # This function returns
    program = compile_genesys_layer(layer_name,
                              update_cfg_dtypes=update_cfg_dtypes,
                              tiling_path=tiling_path,
                              store_tiling=store_tiling,
                              store_checkpoint=False,
                              store_json_output=store_json_output,
                              json_output_filename=json_output_filename,
                              verbose=False,
                              benchmark_path=BENCH_DIR,
                              factor_fn='default',
                            batch_size=batch_size,
                            do_hoist_stage=True,
                            do_tile_stage=True,
                            print_config=False,
                            tiling_search_algorithm=tile_method,
                                    do_compile=False
                                    # relocation_offsets=reloc_offsets
                              )
    if store_compile:
        if added_constr:
            program = update_tile_constraints(program, added_constr, 'conv_bias')

        dir_ext = dir_ext or ''
        # program.compile(verbose=False, finalize_instructions=True)

        store_outputs("cc_layer1", "conv", False,
                      1,
                      False,
                      None,
                      use_random=True,
                      dir_ext=f"{dir_ext}",
                      actual_data=False,
                      store_partials=partials, program=program)

    return program


def get_onnx_shape(tensor_dict, val_name):
    assert val_name in tensor_dict, f"Unable to find shape for {val_name}"
    value = tensor_dict[val_name]
    return value
    # if isinstance(value, onnx.ValueInfoProto):
    #     shape = [d.dim_value for d in value.type.tensor_type.shape.dim]
    # else:
    #     assert isinstance(value, onnx.TensorProto)
    #     shape = value.dims
    # return tuple(shape)



def get_all_unique_layer_params(model_name, layer_name, input_shape_params, out_shape_params, param_names):
    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    model = onnx.load_model(model_path)
    tensor_dict = collect_value_info(model.graph)
    ic_bw_constr = GENESYS_CFG['IBUF_CHANNEL_BW'] // 8
    oc_bw_constr = GENESYS_CFG['PARAM_BUF_CHANNEL_BW'] // 8

    layer_params = []
    for n in model.graph.node:

        if n.op_type == layer_name and not is_dw_conv(n, tensor_dict):
            outputs = n.output
            inputs = n.input
            kv_map = {}

            for a in n.attribute:
                if a.name in param_names:
                    attr_val = onnx.helper.get_attribute_value(a)
                    assert a.name not in kv_map
                    if isinstance(attr_val, Iterable):
                        kv_map[a.name] = attr_val[0]
                    else:
                        kv_map[a.name] = attr_val

            for i, v in enumerate(inputs):
                shape = get_onnx_shape(tensor_dict, v)
                assert len(shape) == len(input_shape_params[i])
                for pidx, p in enumerate(input_shape_params[i]):
                    if p in kv_map:
                        assert kv_map[p] == shape[pidx] or p in ['IC', 'OC'], f"Mismatched values for input key {p}:\n" \
                                                         f"Input val: {shape[pidx]}, Previous: {kv_map[p]} "
                    else:
                        kv_map[p] = shape[pidx]
                    if p == 'IC':
                        kv_map[p] = kv_map[p] + (ic_bw_constr - kv_map[p]) % ic_bw_constr
                    elif p == 'OC':
                        kv_map[p] = kv_map[p] + (oc_bw_constr - kv_map[p]) % oc_bw_constr

            for o, v in enumerate(outputs):
                shape = get_onnx_shape(tensor_dict, v)
                assert len(shape) == len(out_shape_params[o])
                for pidx, p in enumerate(out_shape_params[o]):
                    if p in kv_map:
                        assert kv_map[p] == shape[pidx] or p in ['IC', 'OC'], f"Mismatched values for output key {p}:\n" \
                                                         f"Output val: {shape[pidx]}, Previous: {kv_map[p]} "
                    else:
                        kv_map[p] = shape[pidx]
                    if p == 'IC':
                        kv_map[p] = kv_map[p] + (ic_bw_constr - kv_map[p]) % ic_bw_constr
                    elif p == 'OC':
                        kv_map[p] = kv_map[p] + (oc_bw_constr - kv_map[p]) % oc_bw_constr

            if kv_map not in layer_params:
                layer_params.append(kv_map)
    return layer_params

def ceildiv(a, b):
    return -(-a // b)

def validate_base_addresses(conv_params, instr_len, layer_type="conv_bias"):
    BENCH_BASE_ADDR = {"INSTR": 0, "OBUF": 0, "BBUF": 4096, "WBUF": 24576, "IBUF": 4259840}
    if instr_len * 32 // 8 > BENCH_BASE_ADDR['BBUF']:
        return False
    if layer_type == "conv_bias":

        buf_size = conv_params['OC']*32 // 8
        buf_end = buf_size + BENCH_BASE_ADDR['BBUF']
        if buf_end > BENCH_BASE_ADDR['WBUF']:
            return False
        wgt_size = conv_params['IC']*conv_params['OC']*conv_params['KW']*conv_params['KH']
        wgt_end = wgt_size + BENCH_BASE_ADDR['WBUF']
        if wgt_end > BENCH_BASE_ADDR['IBUF']:
            return False
        ipt_size = conv_params['IC']*conv_params['N']*conv_params['IH']*conv_params['IW']
    else:
        buf_size = conv_params['P'] * 32 // 8
        buf_end = buf_size + BENCH_BASE_ADDR['BBUF']
        if buf_end > BENCH_BASE_ADDR['WBUF']:
            return False
        wgt_size = conv_params['M'] * conv_params['N']
        wgt_end = wgt_size + BENCH_BASE_ADDR['WBUF']
        if wgt_end > BENCH_BASE_ADDR['IBUF']:
            return False
    return True

def splits_from_params(params):
    splits = {}
    for k, v in params.items():
        if "_tile" in k:
            idx = k.split("_")[0]
            splits[idx] = v
    return splits


def invalid_tiling(params):
    splits = splits_from_params(params)
    if splits['IC'] == 1 or any([splits['KH'] > 1, splits['KW'] > 1, splits['OH'] > 1, splits['OW'] > 1]):
        return False
    else:
        return True

def pixel_skip(params):
    return params['KH'] < params['stride'] or params['KW'] < params['stride']

def tiled_oc(params):
    splits = splits_from_params(params)
    return splits['OC'] > 1

def layer_key_from_params(params):
    p = {k: v for k,v in params.items() if 'tile' not in k}
    return tuple(sorted(list(p.items()), key=lambda x: x[0]))

def programs_with_params(base_test_name, check_params):
    valid_programs = []
    for dir in os.scandir(f"{OUT_DIR}/all_resnet50_tests"):
        if dir.name.startswith(base_test_name) and "case" in dir.name and dir.is_dir():
            params = collect_program_params(f"{dir.path}/{base_test_name}_json.json")
            is_valid = True
            for k, v in check_params.items():
                assert k in params
                if isinstance(v, int):
                    if v != params[k]:
                        is_valid = False
                        break
                elif not v(params[k]):
                    is_valid = False
                    break
            if is_valid:
                valid_programs.append(dir.name)
    print(valid_programs)

def find_duplicates(dname, base_test_name):
    pixel_tests = []
    invalid_tiling_tests = []
    valid_tests = []
    tiled_oc_tests = []
    all_params = []
    duplicates = []
    for d in os.scandir(f"{OUT_DIR}/{dname}"):
        if d.name.startswith(base_test_name) and d.is_dir():
            params = collect_program_params(f"{d.path}/{base_test_name}_json.json")
            if check_dictionary_presence(params, all_params):
                duplicates.append(d.name)
            else:
                all_params.append(params)
    print(f"Duplicates: {len(duplicates)}")


def check_programs():
    with open("tgt_tests.txt", "r") as f:
        files = f.readlines()
    files = [f.strip() for f in files]
    base_test_name = "resnet50_1_conv"
    pixel_tests = []
    invalid_tiling_tests = []
    valid_tests = []
    tiled_oc_tests = []
    all_params = []
    duplicates = []
    constraints = defaultdict(list)
    for dir in os.scandir(f"{OUT_DIR}/all_resnet50_tests"):

        if dir.name.startswith(base_test_name) and "case" in dir.name and dir.is_dir() and dir.name not in files:
            params = collect_program_params(f"{dir.path}/{base_test_name}_json.json")
            if check_dictionary_presence(params, all_params):
                duplicates.append(dir.name)
            else:
                all_params.append(params)
            if invalid_tiling(params):
                invalid_tiling_tests.append(dir.name)
            if pixel_skip(params):
                pixel_tests.append(dir.name)
            if not pixel_skip(params) and not invalid_tiling(params):
                valid_tests.append(dir.name)
            if tiled_oc(params):
                tiled_oc_tests.append(dir.name)

    print(f"Pixel tests: {pixel_tests}, {len(pixel_tests)}")
    print(f"Invalid tiling tests: {invalid_tiling_tests}, {len(invalid_tiling_tests)}")
    print(f"Valid tests: {valid_tests}, {len(valid_tests)}")
    print(f"Tiled OC tests: {tiled_oc_tests}, {len(tiled_oc_tests)}")
    print(f"Duplicates: {len(duplicates)}")
    # for i in invalid_tiling_tests:
    #     if i in pixel_tests:
            # print(i)
    for dir in os.scandir(f"{OUT_DIR}/all_resnet50_tests"):
        # if dir.name in invalid_tiling_tests or dir.name in pixel_tests:
        if dir.name in invalid_tiling_tests:
            params = collect_program_params(f"{dir.path}/{base_test_name}_json.json")
            key = layer_key_from_params(params)
            added_constrs = []
            if key in constraints:
                added_constrs.append(unique_splits_constraints(constraints[key]))

            if dir.name in invalid_tiling_tests:
                added_constrs.append("splits['IC'] > 1")

            if len(added_constrs) == 0:
                cstrt = None
            else:
                cstrt = " and ".join(added_constrs)
            ext = dir.name.split(base_test_name)[-1][1:]
            program = compile_from_existing_program(base_test_name, f"{ext}", base_test_name, params=params, added_constraint=cstrt)
            nparams = get_program_params(program, "conv_bias")
            splits = splits_from_params(nparams)
            constraints[key].append(splits)

def collect_program_params(fpath):
    params = {}
    with open(fpath) as f:
        program = json.load(f)
        layer = program['program'][0]

        params['N'] = layer['iterable_dimensions']['N']
        params['IC'] = layer['iterable_dimensions']['IC']
        params['OC'] = layer['iterable_dimensions']['OC']
        params['KH'] = layer['iterable_dimensions']['KH']
        params['KW'] = layer['iterable_dimensions']['KW']
        params['OH'] = layer['iterable_dimensions']['OH']
        params['OW'] = layer['iterable_dimensions']['OW']
        params['IH'] = layer['inputs'][0]["tiling"]['DRAM']['IH']
        params['IW'] = layer['inputs'][0]["tiling"]['DRAM']['IW']
        params['stride'] = layer['operation_parameters']['stride']
        params['pad'] = layer['operation_parameters']['pad']
        params['N_tile'] = layer['iterable_dimensions']['N'] // layer['inputs'][0]["tiling"]['IBUF']['N']
        params['IC_tile'] = layer['iterable_dimensions']['IC'] // layer['inputs'][0]["tiling"]['IBUF']['IC']
        params['KH_tile'] = layer['iterable_dimensions']['KH'] // layer['inputs'][1]["tiling"]['WBUF']['KH']
        params['KW_tile'] = layer['iterable_dimensions']['KW'] // layer['inputs'][1]["tiling"]['WBUF']['KW']
        params['OH_tile'] = layer['iterable_dimensions']['OH'] // layer['outputs'][0]["tiling"]['OBUF']['OH']
        params['OW_tile'] = layer['iterable_dimensions']['OW'] // layer['outputs'][0]["tiling"]['OBUF']['OW']
        params['OC_tile'] = layer['iterable_dimensions']['OC'] // layer['outputs'][0]["tiling"]['OBUF']['OC']
    return params

def compile_from_existing_program(dirname, dir_ext, json_name, added_constraint=None, params=None, preserve_tiling=False):
    if not params:
        params = collect_program_params(f"{OUT_DIR}/{dirname}/{json_name}_json.json")
    program = compile_custom_conv_layer(params['N'], params['IC'], params['OC'], params['IH'],
                                        params['IW'], params['KH'],
                                        params['stride'], params['pad'],
                                        json_name)
    program._name = json_name
    if added_constraint:
        program = update_tile_constraints(program, added_constraint, "conv_bias")
    program.compile(verbose=False, finalize_instructions=True)

    store_outputs(dirname, None, False,
                  1,
                  False,
                  None,
                  use_random=True,
                  dir_ext=dir_ext,
                  actual_data=False,
                  store_partials=False, program=program)
    return program

def update_tile_constraints(program, constraint, layer_type, orig_constraint=None):
    # if not orig_constraint:
    #     program.hag.codelets['conv_bias'].compilation_params['LEVEL1_hint'] = constraint
    # else:
    #     program.hag.codelets['conv_bias'].compilation_params['LEVEL1_hint'] = f"{orig_constraint} and {constraint}"
    if 'LEVEL1_hint' not in program.hag.codelets[layer_type].compilation_params.keys():
        program.hag.codelets[layer_type].compilation_params['LEVEL1_hint'] = constraint
    elif constraint not in program.hag.codelets[layer_type].compilation_params['LEVEL1_hint']:
        orig = program.hag.codelets[layer_type].compilation_params['LEVEL1_hint']
        new_constraint = f"{orig} and {constraint}"
        program.hag.codelets[layer_type].compilation_params['LEVEL1_hint'] = new_constraint
    # program.hag.codelets[layer_id].compilation_params['LEVEL2_hint'] = "splits['H'] == 1 and splits['W'] == 16 " \
    #                                                                      "and splits['N'] == 2 and splits['C'] == 32"
    return program

def collect_existing_params(base_dir_name, base_test_name):
    all_compilation_configs = []
    duplicate_configs = []
    dir_count = 0
    for dir in os.scandir(OUT_DIR):

        if base_dir_name in dir.name and dir.is_dir():
            params = collect_program_params(f"{dir.path}/{base_test_name}_json.json")
            dir_count += 1


            if check_dictionary_presence(params, all_compilation_configs):
                duplicate_configs.append(dir.name)
            else:
                all_compilation_configs.append(params)

    print(list(sorted(duplicate_configs)))

    print(len(list(sorted(duplicate_configs))))

    return all_compilation_configs, dir_count

def unique_splits_constraints(splits):
    assert len(splits) > 0
    constraints = []
    for s in splits:
        c = []
        for k, v in s.items():
            c.append(f"'{k}': {v}")
        c_str = "(splits != {" + ",".join(c) + "})"
        constraints.append(c_str)
    return " and ".join(constraints)

def check_dictionary_presence(d, dlist):
    for v in dlist:
        if v == d:
            return True
    return False

def get_program_params(program, cdlt_type):

    cparams = {}
    if cdlt_type == "gemm":
        cparams['M'], cparams['N'] = program.codelets[0].inputs[0].shape
        assert program.codelets[0].inputs[1].shape[0] == cparams['N']
        cparams['N'], cparams['P'] = program.codelets[0].inputs[1].shape
    else:
        cparams['N'], cparams['IH'], cparams['IW'], cparams['IC'] = program.codelets[0].inputs[0].shape

        cparams['N'], cparams['OH'], cparams['OW'], cparams['OC'] = program.codelets[0].outputs[0].shape

        cparams['KH'], cparams['KW'], cparams['IC'], cparams['OC'] = program.codelets[0].inputs[1].shape
        cparams['stride'] = program.codelets[0].required_params['stride'].value
        cparams['pad'] = program.codelets[0].required_params['pad'].value

    for k, v in program.codelets[0].param_splits[1].items():
        cparams[f'{k}_tile'] = v
    return cparams

def scale_and_compile_layers(model_name,
                             dir_ext,
                             layer_params,
                             updated_layer_params,
                             nunique,
                             do_scaling=True,
                             layers=None,
                             verbose=False,
                             im2col_layers=None, added_constraint=None, debug_output=False):
    im2col_layers = [] if not im2col_layers else im2col_layers
    # layer_idx = idx_start if idx_start is not None else len(updated_layer_params)
    layers = layers if layers is not None else list(range(len(layer_params)))
    orig_conv_constraint = None
    orig_gemm_constraint = None
    all_shapes = []
    param_constraints = defaultdict(list)
    for idx in range(len(layer_params)):

        if idx not in layers:
            continue
        layer = layer_params[idx]
        nlayer_perm = 0
        scale_val = 1
        if verbose:
            layer_param_str = ", ".join([f"{k} = {v}" for k, v in layer.items()])
            print(f"Generating permutations for layer {idx}:\n"
                  f"{layer_param_str}")

        print(f"Start for {nlayer_perm}")
        while nlayer_perm < nunique:
            new_layer_params = {}
            if any([v <= scale_val for k, v in layer.items() if k not in ['N', 'strides', 'pads', 'IC', 'OC', 'KH', 'KW']]) and \
                    scale_val > 1:
                layer_param_str = "\n".join([f"{k} = {v}" for k, v in layer.items()])
                raise RuntimeError(f"Invalid scaling value for layer:\n"
                                   f"Scale value: {scale_val}\n"
                                   f"Layer sizes:\n{layer_param_str}")

            n = ceildiv(layer['N'], scale_val)
            new_layer_params['N'] = n
            if layer['IC'] > GENESYS_CFG['ARRAY_M'] and scale_val > 1:
                ic = ceildiv(layer['IC'], scale_val)
            else:
                ic = layer['IC']
            new_layer_params['IC'] = ic

            if layer['OC'] > GENESYS_CFG['ARRAY_N'] and scale_val > 1:
                oc = ceildiv(layer['OC'], scale_val)
            else:
                oc = layer['OC']
            ic += (GENESYS_CFG['ARRAY_M'] - ic) % GENESYS_CFG['ARRAY_M']
            oc += (GENESYS_CFG['ARRAY_M'] - oc) % GENESYS_CFG['ARRAY_M']
            new_layer_params['OC'] = oc
            h = ceildiv(layer['IH'], scale_val)
            new_layer_params['IH'] = h
            w = ceildiv(layer['IW'], scale_val)
            new_layer_params['IW'] = w
            ksize = layer['KH']
            stride = layer['strides']
            pad = layer['pads']
            new_layer_params['KH'], new_layer_params['KW'] = ksize, ksize

            if check_dictionary_presence(new_layer_params, all_shapes):
                scale_val += 1
                continue
            else:
                all_shapes.append(new_layer_params.copy())
            new_layer_params['pads'] = pad
            new_layer_params['strides'] = stride
            if idx in im2col_layers:

                oh = int((h + 2 * pad - ksize) / stride) + 1
                ow = int((w + 2 * pad - ksize) / stride) + 1
                M, N, P = compute_im2col_dims(new_layer_params, oh, ow)
                program = compile_custom_gemm_layer(M, N, P, f"{model_name}_custom",
                                                    partials=debug_output)
            else:
                program = compile_custom_conv_layer(n, ic, oc, h, w, ksize, stride, pad, f"{model_name}_custom",
                                                    partials=debug_output, store_compile=False)

            if orig_conv_constraint is None:
                orig_conv_constraint = program.hag.codelets['conv_bias'].compilation_params['LEVEL1_hint']
                orig_gemm_constraint = program.hag.codelets['gemm'].compilation_params['LEVEL1_hint']
                if added_constraint is not None:
                    orig_conv_constraint = f"{orig_conv_constraint} and {added_constraint}"
                    orig_gemm_constraint = f"{orig_gemm_constraint} and {added_constraint}"

            if idx in im2col_layers:
                orig_constraint = orig_gemm_constraint
                cdlt_type = "gemm"
            else:
                orig_constraint = orig_conv_constraint
                cdlt_type = "conv_bias"
            key = layer_key_from_params(new_layer_params)

            if key in param_constraints:
                assert orig_conv_constraint is not None
                constraint = unique_splits_constraints(param_constraints[key])
                program.hag.codelets[cdlt_type].compilation_params[
                    'LEVEL1_hint'] = f"{orig_constraint} and {constraint}"

            curr_params = program.hag.codelets[cdlt_type].compilation_params['LEVEL1_hint']
            if added_constraint not in curr_params:
                if len(curr_params) == 0:
                    program.hag.codelets[cdlt_type].compilation_params[
                        'LEVEL1_hint'] = orig_constraint
                else:
                    program.hag.codelets[cdlt_type].compilation_params[
                        'LEVEL1_hint'] += f" and {orig_constraint}"

            if verbose:
                layer_param_str = ", ".join([f"{k} = {v}" for k, v in new_layer_params.items()])
                constraint_str = program.hag.codelets[cdlt_type].compilation_params['LEVEL1_hint']
                print(f"Generating permutation {nlayer_perm} for layer {idx}:\n"
                      f"layer params: {layer_param_str}\n"
                      f"Constraints: {constraint_str}")
            try:
                print(f"Compiling with split {scale_val}")
                program.compile(verbose=False, finalize=True)
            except Exception as e:
                print(f"Unable to compile layer {e}")
                if do_scaling:
                    scale_val += 1
                    continue
                else:
                    raise RuntimeError(f"Unable to compile layer:\n"
                                       f"{e}")
            cparams = {}

            if idx in im2col_layers:
                cparams['M'], cparams['N'] = program.codelets[0].inputs[0].shape
                assert program.codelets[0].inputs[1].shape[0] == cparams['N']
                cparams['N'], cparams['P'] = program.codelets[0].inputs[1].shape
            else:
                cparams['N'], cparams['IH'], cparams['IW'], cparams['IC'] = program.codelets[0].inputs[0].shape

                cparams['N'], cparams['OH'], cparams['OW'], cparams['OC'] = program.codelets[0].outputs[0].shape

                cparams['KH'], cparams['KW'], cparams['IC'], cparams['OC'] = program.codelets[0].inputs[1].shape
                cparams['stride'] = stride
                cparams['pad'] = pad

            for k, v in program.codelets[0].param_splits[1].items():
                cparams[f'{k}_tile'] = v
            nparams = get_program_params(program, cdlt_type)
            splits = splits_from_params(nparams)
            param_constraints[key].append(splits)
            # splits = splits_from_params(cparams)
            # layer_splits.append(splits)

            if not check_dictionary_presence(cparams, updated_layer_params):
                updated_layer_params.append(cparams)
                if cdlt_type != "gemm" and pixel_skip(cparams):
                    print(f"Test case wit pixel")
                instr_len = program.codelets[0].num_instr

                if not validate_base_addresses(cparams, instr_len, layer_type=cdlt_type):
                    raise RuntimeError

                if verbose:
                    print(f"Storing outputs for layer")
                    print(program.emit("operations_idx"))

                store_outputs("cc_layer1", cdlt_type, False,
                              1,
                              False,
                              None,
                              use_random=True,
                              dir_ext=f"{dir_ext}{idx*nunique + nlayer_perm}",
                              actual_data=False,
                              store_partials=debug_output,
                              program=program)
                nlayer_perm += 1
            else:
                raise RuntimeError(f"Found duplicate layer somehow:\n"
                                   f"Splits: {splits}\n"
                                   f"Updated Layer params: {updated_layer_params}\n"
                                   f"Cparams: {cparams}\n"
                                   f"Layer: {idx}\n"
                                   f"Layer perm: {nlayer_perm}")

    return updated_layer_params

def sys_array_bench_old():
    inp_params = []
    inp_params.append(["N", "IC", "IH", "IW"])
    inp_params.append(["OC", "IC", "KH", "KW"])
    inp_params.append(["OC"])

    out_params = []
    out_params.append(["N", "OC", "OH", "OW"])

    attr_params = []
    attr_params.append("pads")
    attr_params.append("strides")

    n = 1
    oc = 64
    ic = 128
    ih = 12
    iw = 12
    stride = 1
    pad = 0
    k = 3
    model_name = "fpga_8x8_tile2_ic_oh_ow"
    # tcstr = f"splits['OC'] > 1 and splits['OH'] > 1 and splits['IC'] == 1"


    # tcstr = "np.prod(list(splits.values())) <= 20 " \
    #                   "and splits['IC'] > 1 " \
    #                   "and splits['OC'] > 1 " \
    #                   "and splits['OH'] > 1 " \
    #                   "and splits['OW'] > 1"
    tcstr = "np.prod(list(splits.values())) < 16 " \
                      "and splits['IC'] > 1 " \
                      "and splits['OH'] > 1 " \
                      "and splits['OW'] > 1"
    # # # # # #
    program = compile_custom_conv_layer(n, ic, oc, ih, iw, k, stride, pad, model_name,
                                        store_compile=True,
                                        partials=True,
                                        added_constr=tcstr
                                        )


def model_benches(model_name,
                  nunique=1,
                  sa_size=64,
                  layers=None,
                  layer_start=0,
                  layer_end=None,
                  debug_output=False,
                  ext=None,
                   verbose=False,
                  do_scaling=True):
    assert sa_size == GENESYS_CFG['ARRAY_M'] and sa_size == GENESYS_CFG['ARRAY_N']
    inp_params = []
    inp_params.append(["N", "IC", "IH", "IW"])
    inp_params.append(["OC", "IC", "KH", "KW"])
    inp_params.append(["OC"])

    out_params = []
    out_params.append(["N", "OC", "OH", "OW"])

    attr_params = []
    attr_params.append("pads")
    attr_params.append("strides")
    tile_constraint = "True"
    # tile_constraint = "np.prod(list(splits.values())) < 64"
    # tile_constraint = "np.prod(list(splits.values())) < 50 " \
    #                   "and splits['IC'] > 1 " \
    #                   "and splits['OC'] > 1 " \
    #                   "and splits['OH'] > 1 " \
    #                   "and splits['OW'] > 1"
    # tile_constraint = "np.prod(list(splits.values())) < 16 " \
    #                   "and splits['IC'] > 1 " \
    #                   "and splits['OH'] > 1 " \
    #                   "and splits['OW'] > 1"
    ext = ext or ""
    import pprint
    layer_params = get_all_unique_layer_params(model_name, "Conv", inp_params, out_params, attr_params)
    layer_end = layer_end or len(layer_params)
    scale_and_compile_layers(model_name, f"{sa_size}x{sa_size}_{ext}", layer_params[layer_start:layer_end], [], nunique,
                             added_constraint=tile_constraint,
                             layers=layers,
                             debug_output=debug_output,
                             verbose=verbose,
                             do_scaling=do_scaling)


def simd_benchmarks1(tests=None, layers=None, num=0):
    base_name = f"fpga{num}"

    configs = {
        "t0":  {"constraint": "True", "scale_factor": 1},
        "t1" : {"constraint": "splits['H'] > 1", "scale_factor": 3},
        "t2" : {"constraint": "splits['N'] > 1 and splits['H'] > 1 and splits['W'] == 1", "scale_factor": 2},
        "t3" : {"constraint": "splits['N'] > 1 and splits['H'] > 1 and splits['W'] > 1", "scale_factor": 2},
    }
    ops = ["relu", "elem_add", "elem_mul", "max_pool", "elem_sigmoid", "elem_sub"]
    if layers is None:
        layers = ops
    if tests is None:
        tests = list(configs.keys())

    for o in layers:
        for n in tests:
            assert n in configs
            cfg = configs[n]
            off = cfg['scale_factor']
            model_name = f"{base_name}_{n}"
            if o == "max_pool":
                constraint = cfg['constraint'].replace("'H'", "'OH'").replace("'W'", "'OW'")
                params = {"N": 2, "C": 128, "IH": 112//(2**off), "IW": 112//(2**off), "OH": 28, "OW": 28, "KH": 3, "KW": 3, "pad": 1, "stride": 2}
            else:
                constraint = cfg['constraint']
                # params = {"N": 2, "C": 128, "H": 256//(2**off), "W": 256//(2**off)}
                params = {"N": 1, "C": 128, "H": 10, "W": 10}
            program = compile_custom_layer(model_name, o, params, store_compile=True, added_constr=constraint)



def simd_benchmarks2(tests=None, layers=None, num=12):
    base_name = f"fpga{num}"

    configs = {
        "t0": {"constraint": "True", "scale_factor": 1},
        # "t1" : {"constraint": "splits['H'] > 1", "scale_factor": 3},
        "t1" : {"constraint": "splits['C'] == 2 and splits['H'] == 2 and splits['W'] > 1", "scale_factor": 2},
        # "t1" : {"constraint": "splits['H'] == 2 and splits['W'] == 2 and splits['C'] == 1", "scale_factor": 3},
        "t2" : {"constraint": "splits['N'] > 1 and splits['H'] > 1", "scale_factor": 2},
        "t3" : {"constraint": "splits['C'] > 1 and splits['H'] > 1 and splits['W'] > 1", "scale_factor": 2},
    }
    ops = ["global_avg_pool", "elem_clip", "depthwise_conv", "leaky_relu", "elem_sub"]
    if layers is None:
        layers = ops
    if tests is None:
        configs.pop("t0")
        tests = list(configs.keys())

    for o in layers:
        for n in tests:
            assert n in configs
            cfg = configs[n]
            off = cfg['scale_factor']
            model_name = f"{base_name}_{n}"
            if o == "global_avg_pool":
                # constraint = cfg['constraint'].replace("'H'", "'OH'").replace("'W'", "'OW'")
                constraint = "True" if n != "t3" else "splits['C'] > 1"
                params = {"N": 1, "C": 1024//off, "IH": 7, "IW": 7, "OH": 1, "OW": 1}
            elif o == "depthwise_conv":
                # constraint = cfg['constraint'].replace("'H'", "'OH'").replace("'W'", "'OW'")
                constraint = "True"
                if n == "t1":
                    stride = 2
                    kh, kw = 3, 3
                elif n == "t2":
                    stride = 1
                    kh, kw = 3, 3
                elif n == "t3":
                    stride = 1
                    kh, kw = 5, 5
                else:
                    raise RuntimeError
                params = {"N": 1, "C": 512//off, "IH": 28, "IW": 28, "OH": 14, "OW": 14, "KH": kh, "KW": kw, "stride": stride, "pad": 1}
            else:
                constraint = cfg['constraint']
                params = {"N": 2, "C": 128, "H": 256//(2**off), "W": 256//(2**off)}

            if o == "elem_clip":
                params['minval'] = 0
                params['maxval'] = 6
            program = compile_custom_layer(model_name, o, params, store_compile=True, added_constr=constraint)


def simd_benchmarks3(tests=None, layers=None, num=12):
    base_name = f"fpga{num}"

    configs = {
        "t0": {"constraint": "True", "scale_factor": 1, "reduce": "True"},
        "t1" : {"constraint": "splits['N'] > 1", "scale_factor": 3, "reduce": "splits['C'] > 1", "exp": np.float64(3.0)},
        "t2" : {"constraint": "splits['N'] > 1 and splits['C'] > 1", "reduce": "splits['C'] > 1 and splits['C'] > 1", "scale_factor": 2,
                "exp": np.float64(4.0)},
        "t3" : {"constraint": "splits['C'] != 0", "scale_factor": 1, "reduce": "splits['C'] != 0",
                "exp": np.float64(5.0)},
    }
    ops = ["elem_tanh2d", "elem_ceil2d", "elem_pow2d", "reduce_min2d", "reduce_mean2d"]
    if layers is None:
        layers = ops
    if tests is None:
        configs.pop("t0")
        tests = list(configs.keys())
    assert all([o in ops for o in layers])
    for o in layers:
        for n in tests:
            assert n in configs
            cfg = configs[n]
            off = cfg['scale_factor']
            model_name = f"{base_name}_{n}"
            params = {"N": 256, "C": 3072//off}
            constraint = cfg['constraint']

            if o == "elem_pow2d":
                params['exp'] = np.float64(3.0)
            elif o in ["reduce_mean2d", "reduce_min2d"]:
                params['keepdim'] = True
                params['axis'] = 0
                constraint = cfg['reduce']

            program = compile_custom_layer(model_name, o, params, store_compile=True, added_constr=constraint)


def simd_benchmarks4(tests=None, layers=None, num=0):
    base_name = f"fpga{num}"

    configs = {
        "t1" : {"constraint": "True", "scale_factor": 3},
        "t2" : {"constraint": "True", "scale_factor": 2},
        "t3" : {"constraint": "True", "scale_factor": 1},
    }
    ops = ["tensor_transpose2d"]
    if layers is None:
        layers = ops
    if tests is None:
        tests = list(configs.keys())

    for o in layers:
        for n in tests:
            assert n in configs
            cfg = configs[n]
            off = cfg['scale_factor']
            model_name = f"{base_name}_{n}"
            assert o == "tensor_transpose2d"
            constraint = cfg['constraint']
            # params = {"N": 256, "C": 128, 'axis': (1,0)}
            params = {"N": 256, "C": 3072//(2**off), 'axis': (1,0)}
            # params = {"N": 2, "C": 128, "H": 256 // (2 ** off), "W": 256 // (2 ** off)}

            program = compile_custom_layer(model_name, o, params, store_compile=True, added_constr=constraint)


def systolic_array_gemm_bench(num=2):
    base_test_name = f"fpga_8x8_tile{num}"

    layer_configs = []
    layer_configs.append({"m": 1, "n": 128, "p": 128})

    constraints = {}
    # constraints['ic_oc_oh_ow'] = "np.prod(list(splits.values())) <= 20 " \
    #                   "and splits['IC'] > 1 " \
    #                   "and splits['OC'] > 1 " \
    #                   "and splits['OH'] > 1 " \
    #                   "and splits['OW'] > 1"
    # constraints['ic_oh_ow'] = "np.prod(list(splits.values())) <= 20 and splits['OC'] == 1" \
    #                   "and splits['IC'] > 1 " \
    #                   "and splits['OH'] > 1 " \
    #                   "and splits['OW'] > 1"
    # constraints['oc_oh_ow'] = "np.prod(list(splits.values())) <= 20 and splits['IC'] == 1" \
    #                   "and splits['OC'] > 1 " \
    #                   "and splits['OH'] > 1 " \
    #                   "and splits['OW'] > 1"

    for test_id, lc in enumerate(layer_configs):
        # for name, cstr in constraints.items():
        test_name = f"{base_test_name}_case{test_id}"
        program = compile_custom_gemm_layer(lc['m'], lc['n'], lc['p'], test_name,
                                            store_compile=True,
                                            partials=True)

def systolic_array_conv_bench(sys_array_size=8, num=2,
                              constr_names=None,
                              layers=None):
    assert GENESYS_CFG['ARRAY_N'] == sys_array_size and GENESYS_CFG['ARRAY_M'] == sys_array_size
    base_test_name = f"fpga_{sys_array_size}x{sys_array_size}_tile{num}"
    scale_factor = sys_array_size//8
    inp_params = []
    inp_params.append(["N", "IC", "IH", "IW"])
    inp_params.append(["OC", "IC", "KH", "KW"])
    inp_params.append(["OC"])

    out_params = []
    out_params.append(["N", "OC", "OH", "OW"])

    attr_params = []
    attr_params.append("pads")
    attr_params.append("strides")
    layer_configs = []
    layer_configs.append({"n": 1, "oc": 128, "ic": 128, "ih": 10, "iw": 10, "k": 3, "stride": 1, "pad": 1}) # Case 0
    layer_configs.append({"n": 1, "oc": 128, "ic": 128, "ih": 12, "iw": 12, "k": 3, "stride": 2, "pad": 1})  # Case 1
    layer_configs.append({"n": 1, "oc": 128, "ic": 128, "ih": 14, "iw": 14, "k": 5, "stride": 1, "pad": 1}) # Case  2
    layer_configs.append({"n": 1, "oc": 128, "ic": 128, "ih": 9, "iw": 9, "k": 5, "stride": 2, "pad": 1}) # Case    3
    layer_configs.append({"n": 1, "oc": 128, "ic": 128, "ih": 12, "iw": 12, "k": 5, "stride": 2, "pad": 0}) # Case  4
    layer_configs.append({"n": 1, "oc": 128, "ic": 128, "ih": 10, "iw": 10, "k": 1, "stride": 1, "pad": 0}) # Case  5
    layer_configs.append({"n": 4, "oc": 128, "ic": 128, "ih": 10, "iw": 10, "k": 3, "stride": 1, "pad": 0}) # Case  6
    layer_configs.append({"n": 4, "oc": 128, "ic": 128, "ih": 10, "iw": 10, "k": 3, "stride": 2, "pad": 0}) # Case  7
    layer_configs.append({"n": 4, "oc": 128, "ic": 128, "ih": 11, "iw": 11, "k": 3, "stride": 2, "pad": 1}) # Case  8
    layer_configs.append({"n": 4, "oc": 128, "ic": 128, "ih": 11, "iw": 11, "k": 3, "stride": 3, "pad": 1}) # Case  9
    constraints = {}
    constraints['default'] = "True"
    constraints['ic_oc_oh_ow'] = "np.prod(list(splits.values())) <= 50 " \
                      "and splits['IC'] > 1 " \
                      "and splits['OC'] > 1 " \
                      "and splits['OH'] > 1 " \
                      "and splits['OW'] > 1"
    constraints['ic_oh_ow'] = "np.prod(list(splits.values())) <= 50 and splits['OC'] == 1" \
                      "and splits['IC'] > 1 " \
                      "and splits['OH'] > 1 " \
                      "and splits['OW'] > 1"
    constraints['oc_oh_ow'] = "np.prod(list(splits.values())) <= 50 and splits['IC'] == 1" \
                      "and splits['OC'] > 1 " \
                      "and splits['OH'] > 1 " \
                      "and splits['OW'] > 1"
    if layers is None:
        layers = list(range(len(layer_configs)))
    else:
        assert all([isinstance(i, int) for i in layers])


    if constr_names is None:
        constraints.pop('default')
        constr_names = list(constraints.keys())
    else:
        assert all([i in constraints for i in constr_names])

    for test_id in layers:
        lc = layer_configs[test_id]
        for name in constr_names:
            cstr = constraints[name]
            test_name = f"{base_test_name}_case{test_id}_{name}"
            program = compile_custom_conv_layer(lc['n'], lc['ic'], lc['oc'], lc['ih'], lc['iw'], lc['k'], lc['stride'], lc['pad'], test_name,
                                                store_compile=True,
                                                partials=False,
                                                added_constr=cstr)


def conv_out_size(dims):
    if 'iw' in dims:
        ow = (dims['iw'] - dims['k'] + 2*dims['pad']) // dims['stride'] + 1
        oh = (dims['ih'] - dims['k'] + 2*dims['pad']) // dims['stride'] + 1

        return (dims['n'], dims['oc'], oh, ow)
    else:
        assert 'IH' in dims
        ow = (dims['IW'] - dims['KH'] + 2*dims['pad']) // dims['stride'] + 1
        oh = (dims['IH'] - dims['KW'] + 2*dims['pad']) // dims['stride'] + 1

        return (dims['N'], dims['OC'], oh, ow)


def from_target_out_size(dims, ohw):
    ihw = ((ohw - 1) * dims['stride']) - 2*dims['pad'] + dims['k']
    return ihw

def systolic_array_conv_scaled(layer_id=None, num=2, case_num=0):
    if layer_id is not None:
        assert isinstance(layer_id, int)
        assert layer_id == case_num
    base_test_name = f"fpga_8x8_tile{num}"
    inp_params = []
    inp_params.append(["N", "IC", "IH", "IW"])
    inp_params.append(["OC", "IC", "KH", "KW"])
    inp_params.append(["OC"])

    out_params = []
    out_params.append(["N", "OC", "OH", "OW"])

    attr_params = []
    attr_params.append("pads")
    attr_params.append("strides")
    scales = [2, 3, 4, 5]
    layer_configs = []
    layer_configs.append({"n": 1, "oc": 128, "ic": 128, "ih": 10, "iw": 10, "k": 3, "stride": 1, "pad": 1}) # Case 0
    layer_configs.append({"n": 1, "oc": 128, "ic": 128, "ih": 12, "iw": 12, "k": 3, "stride": 2, "pad": 1})  # Case 1
    layer_configs.append({"n": 1, "oc": 128, "ic": 128, "ih": 14, "iw": 14, "k": 5, "stride": 1, "pad": 1}) # Case  2
    layer_configs.append({"n": 1, "oc": 128, "ic": 128, "ih": 9, "iw": 9, "k": 5, "stride": 2, "pad": 1}) # Case    3
    layer_configs.append({"n": 1, "oc": 128, "ic": 128, "ih": 12, "iw": 12, "k": 5, "stride": 2, "pad": 0}) # Case  4
    layer_configs.append({"n": 1, "oc": 128, "ic": 128, "ih": 10, "iw": 10, "k": 1, "stride": 1, "pad": 0}) # Case  5
    layer_configs.append({"n": 4, "oc": 128, "ic": 128, "ih": 10, "iw": 10, "k": 3, "stride": 1, "pad": 0}) # Case  6
    layer_configs.append({"n": 4, "oc": 128, "ic": 128, "ih": 10, "iw": 10, "k": 3, "stride": 2, "pad": 0}) # Case  7
    layer_configs.append({"n": 4, "oc": 128, "ic": 128, "ih": 11, "iw": 11, "k": 3, "stride": 2, "pad": 1}) # Case  8
    layer_configs.append({"n": 4, "oc": 128, "ic": 128, "ih": 11, "iw": 11, "k": 3, "stride": 3, "pad": 1}) # Case  9
    constraints = {}

    tile_cases = []
    tile_cases.append({'N': 1, 'IC': 128, 'OC': 64, 'OH': 5, 'OW': 5}) # Case 0
    tile_cases.append({'N': 1, 'IC': 128, 'OC': 64, 'OH': 3, 'OW': 3}) # Case 1
    tile_cases.append({'N': 1, 'IC': 128, 'OC': 64, 'OH': 6, 'OW': 6}) # Case 2
    tile_cases.append({'N': 1, 'IC': 128, 'OC': 64, 'OH': 2, 'OW': 2}) # Case 3
    tile_cases.append({'N': 1, 'IC': 128, 'OC': 64, 'OH': 2, 'OW': 2}) # Case 4
    tile_cases.append({'N': 1, 'IC': 128, 'OC': 64, 'OH': 5, 'OW': 5}) # Case 5
    tile_cases.append({'N': 4, 'IC': 128, 'OC': 64, 'OH': 4, 'OW': 4}) # Case 6
    tile_cases.append({'N': 4, 'IC': 128, 'OC': 64, 'OH': 2, 'OW': 2}) # Case 7
    tile_cases.append({'N': 4, 'IC': 128, 'OC': 64, 'OH': 3, 'OW': 3}) # Case 8
    tile_cases.append({'N': 4, 'IC': 128, 'OC': 64, 'OH': 2, 'OW': 2}) # Case 9

    tile_sizes = tile_cases[case_num]




    tile_size_constraint = " and ".join([f"sizes['{k}'] == {v}" for k, v in tile_sizes.items()])
    # tile_size_constraint = "True"

    constraints['oc_oh_ow'] = "splits['IC'] == 1 " \
                      "and splits['OC'] > 1 " \
                      "and splits['OH'] > 1 " \
                      f"and splits['OW'] > 1 and {tile_size_constraint}"

    for test_id, lc in enumerate(layer_configs):
        if layer_id is not None and layer_id != test_id:
            continue
        # init_ohw = conv_out_size(lc)[2]
        for s in scales:
            test_name = f"{base_test_name}_case{test_id}_{s}x_scale"
            # scaled_ihw = from_target_out_size(lc, init_ohw*s)
            # scaled_ihw = init_ohw
            if s % 2 == 0:
                program = compile_custom_conv_layer(lc['n'], lc['ic'], lc['oc']*s, lc['ih'], lc['iw'], lc['k'], lc['stride'], lc['pad'], test_name,
                                                    store_compile=True,
                                                    partials=False,
                                                    added_constr=constraints['oc_oh_ow'])
            else:
                program = compile_custom_conv_layer(lc['n']*s, lc['ic'], lc['oc']*s, lc['ih'], lc['iw'], lc['k'], lc['stride'], lc['pad'], test_name,
                                                    store_compile=True,
                                                    partials=False,
                                                    added_constr=constraints['oc_oh_ow'])


def simd_benchmarks5(tests=None, layers=None, num=12):
    base_name = f"fpga{num}"

    # configs = {
    #     "t0": {"constraint": "True", "scale_factor": 1},
    #     "t1" : {"constraint": "splits['H'] > 1", "scale_factor": 3},
    #     "t2" : {"constraint": "splits['N'] > 1 and splits['H'] > 1", "scale_factor": 2},
    #     "t3" : {"constraint": "splits['C'] > 1 and splits['H'] > 1 and splits['W'] > 1", "scale_factor": 2},
    # }
    configs = {
        "t0": {"constraint": "True", "scale_factor": 1},
        "t1" : {"constraint": "splits['H'] > 1", "scale_factor": 3},
        "t2" : {"constraint": "splits['N'] > 1 and splits['H'] > 1 and splits['W'] == 1", "scale_factor": 2},
        "t3" : {"constraint": "splits['N'] > 1 and splits['H'] > 1 and splits['W'] > 1", "scale_factor": 2},
    }
    ops = ["elem_div", "elem_less", "elem_equal", "elem_exp"]
    if layers is None:
        layers = ops
    if tests is None:
        configs.pop("t0")
        tests = list(configs.keys())

    for o in layers:
        for n in tests:
            assert n in configs
            cfg = configs[n]
            off = cfg['scale_factor']
            model_name = f"{base_name}_{n}"
            constraint = cfg['constraint']
            # params = {"N": 2, "C": 128, "H": 256 // (2 ** off), "W": 256 // (2 ** off)}

            params = {"N": 2, "C": 128, "H": 256//(2**off), "W": 256//(2**off)}

            program = compile_custom_layer(model_name, o, params, store_compile=True, added_constr=constraint)


def multi_layer_cases1(tests=None, layers=None, num=12):
    base_name = f"fpga{num}"

    configs = {
        "t0": {"constraint": "True", "scale_factor": 1},
        "t1": {"constraint": "splits['H'] > 1", "scale_factor": 3},
        "t2": {"constraint": "splits['N'] > 1 and splits['H'] > 1 and splits['W'] == 1", "scale_factor": 2},
        "t3": {"constraint": "splits['N'] > 1 and splits['H'] > 1 and splits['W'] > 1", "scale_factor": 2},
    }
    ops = [["conv_bias", "relu", "conv_bias"], ["conv_bias", "relu"]]
    if layers is None:
        layers = ops
    if tests is None:
        configs.pop("t0")
        tests = list(configs.keys())
    layer_params = {
        "conv_bias": {"N": 1, "IC": 512, "OC": 256, "IH": 28, "IW": 28, "OH": 14, "OW": 14, "pad": 0},
        "gemm": {"M": 1, "N": 512, "P": 256},
        "relu": {"N": 1, "C": 256, "H": 28, "W": 28},
        "elem_add": {"N": 1, "C": 256, "H": 28, "W": 28},
    }
    for o in layers:
        for n in tests:
            assert n in configs
            cfg = configs[n]
            off = cfg['scale_factor']
            seq_name = "_".join(o)
            model_name = f"{seq_name}_{base_name}_{n}"
            constraint = {l:"True" for l in o}

            input_params = {}

            input_shape = None
            layer_names = defaultdict(int)
            for lname in o:
                l = f"{lname}{layer_names[lname]}"
                layer_names[lname] += 1
                input_params[l] = layer_params[lname].copy()

                if lname == "conv_bias":
                    if input_shape is not None:
                        coff = off + 1
                    else:
                        coff = off
                    if n == "t1":
                        input_params[l]['stride'] = 2
                        input_params[l]['KH'], input_params[l]['KW'] = 3, 3
                    elif n == "t2":
                        input_params[l]['stride'] = 1
                        input_params[l]['KH'], input_params[l]['KW'] = 3, 3
                    elif n == "t3":
                        input_params[l]['stride'] = 1
                        input_params[l]['KH'], input_params[l]['KW'] = 5, 5
                    else:
                        raise RuntimeError

                    if input_shape is None:
                        input_params[l]['IC'] = input_params[l]['IC'] // coff
                        input_params[l]['OC'] = input_params[l]['OC'] // coff
                    else:
                        input_params[l]['IC'] = input_shape[1]
                        input_params[l]['OC'] = input_params[l]['OC'] // coff

                    input_shape = conv_out_size(input_params[l])

                elif lname == "relu":
                    assert input_shape is not None
                    input_keys = list(layer_params[lname].keys())
                    for i, v in enumerate(input_shape):
                        input_params[l][input_keys[i]] = v
                elif lname == "elem_add":
                    assert input_shape is not None
                    input_keys = list(layer_params[lname].keys())
                    for i, v in enumerate(input_shape):
                        input_params[l][input_keys[i]] = v
                elif lname == "gemm":
                    assert input_shape is not None
                    assert len(input_shape) == 4
                    input_params[l]['M'] = input_shape[0]
                    input_params[l]['N'] = input_shape[1]*input_shape[2]*input_shape[3]
                    input_params[l]['P'] = input_params[l]['P'] // off
                else:
                    raise RuntimeError
            program = compile_custom_multi_layer(model_name, o, input_params, store_compile=True, added_constr=constraint)


def fusion_testing(fusion_model, layer_sequence, num, testnum=12, store_compile=True, verbose=False,
                   added_constr=None, generate_data=True):
    seq_name = "_".join(layer_sequence)
    filename = f"{fusion_model}_{seq_name}{num}"

    dir_ext = f"test{testnum}"

    opname = None
    for name, fop in FUSION_OP_INFO.items():
        if 'seq' in fop and fop['seq'] == layer_sequence:
            opname = name
            break
    if opname is None:
        raise RuntimeError(f"INvalid fusion sequence: {layer_sequence}\n")
    fuse_layers = True
    if not os.path.exists(f"{MODEL_DIR}/srdfg/{filename}.srdfg"):
        convert_model_to_polymath(f"{MODEL_DIR}/{filename}.onnx")
        assert os.path.exists(f"{MODEL_DIR}/srdfg/{filename}.srdfg")

    if added_constr is not None:
        assert isinstance(added_constr, str)
        added_constr = {opname: added_constr}

    program = compile_full_model(filename,
                                 store_compile=store_compile,
                                 dir_ext=dir_ext,
                                 added_constr=added_constr,
                                 train_mode=False,
                                 verbose=verbose,
                                 model_data=None,
                                 fuse_layers=fuse_layers,
                                 generate_data=generate_data
                                 )


if __name__ == "__main__":
    # fusion_testing("efficientnet-lite4-11-opt",
    #                # ['Conv', 'Relu'],
    #                # ['Conv', 'Add', 'Relu'],
    #                # ['Conv', 'Relu', 'MaxPool'],
    #                # ['Conv', 'Clip', 'DepthwiseConv'],
    #                ['Conv', 'Clip', 'DepthwiseConv', 'Clip'],
    #                1,
    #                # added_constr="splits['OW'] == 2",
    #                testnum=0,
    #                generate_data=False)

    # systolic_array_conv_bench(32, num=40)
    # simd_benchmarks1(tests=["t0"], layers=["max_pool"], num=0)
    simd_benchmarks3(tests=["t0"], layers=["reduce_mean2d"], num=1)
    # simd_benchmarks1(tests=["t1"], layers=["elem_sigmoid"], num=1)
    # simd_benchmarks2(tests=["t1"], layers=["elem_clip"], num=0)
    # simd_benchmarks2(tests=["t3"], layers=["global_avg_pool"], num=0)
    # systolic_array_conv_bench(sys_array_size=64,
    #                           num=0,
    #                           constr_names=["default"],
    #                           layers=[0])
    # multi_layer_cases1(num=1)
    # simd_benchmarks5(num=68, layers=["elem_less", "elem_equal", "elem_exp"])
    # simd_benchmarks3(layers=["elem_pow2d"], num=59)
    # simd_benchmarks2(layers=["leaky_relu"], tests=["t1"], num=1)
    # simd_benchmarks3(layers=["elem_tanh2d", "elem_ceil2d"], num=50)
    # simd_benchmarks1(layers=["relu"], tests=["t1"], num=40)
    # model = "resnet18"

    # model = "efficientnet-lite4-11-opt"
    # model = "resnet18"
    # model_benches(model,
    #               sa_size=64,
    #               debug_output=False,
    #               layer_start=0,
    #               layer_end=1,
    #               ext="t14_",
    #               verbose=False,
    #               do_scaling=False)

    # model_benches("resnet18", sa_size=32, debug_output=False, ext="t12_", verbose=False)
    # resnet_benches(sa_size=32, debug_output=False, ext="t11_", verbose=False)

    # systolic_array_conv_scaled(layer_id=0, num=5, case_num=0)
    # systolic_array_gemm_bench(4)


