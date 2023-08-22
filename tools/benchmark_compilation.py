import argparse
import os
import numpy as np

import sys
from collections import defaultdict
from pathlib import Path
import multiprocessing as mp
from functools import partial
from pprint import pprint

import polymath as pm

from codelets.examples.genesys import load_fusion_op_info
from codelets.examples.genesys import load_config
from codelets.examples.genesys import compile_full_model
from codelets.examples.genesys import DataGen
from tools.compile_layer import store_program_codelets

import onnxruntime as ort
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
import onnx
CWD = Path(f"{Path(__file__).parent}")
MODEL_DIR = Path(f"{Path(__file__).parent}/../benchmarks/models")
CFG_PATH = f"{CWD}/../codelets/examples/genesys/configs"
FUSION_NAME_MAPPING = {
    'conv': 'conv_bias',
    'relu': 'relu',
    'leakyrelu': 'leaky_relu',
    'add': 'elem_add',
    'depthwiseconv': 'depthwise_conv_bias',
    'maxpool': 'max_pool',
    'globalaveragepool': 'global_avg_pool',
    'clip': 'elem_clip',
    'averagepool': 'avg_pool',
    'sub': 'elem_sub'
}
BENCHMARK_INFO = {
    "resnet18" : {
        "num_layers_unfused": 49,
        "num_layers_fused": 24,
        "fused_skipped": [18, 19, 20]
    },
    "resnet50": {
        "num_layers_unfused": 122,
        "num_layers_fused": 57,
        "fused_skipped": [45, 49, 52]
    },
    "efficientnet-lite4-opt-no-softmax": {
        "num_layers_unfused": 179,
        "num_layers_fused": 68,
        "fused_skipped": []
    },
    "mobilenetv2-opt": {
        "num_layers_unfused": 100,
        "num_layers_fused": 42,
        "fused_skipped": []
    },
    "bert-base-cased-transpose-opt-trimmed-ort": {
        "num_layers_unfused": 0,
        "num_layers_fused": 0,
    },
    "yolov3-opt-static": {
        "num_layers_unfused": 172,
        "num_layers_fused": 77,
        "fused_skipped": [35, 37, 39, 41, 43, 45, 47, 76]
    },
    "lenet-opt-trimmed" : {
        "num_layers_unfused": 8,
        "num_layers_fused": 5,
        "fused_skipped": []
    }
}

NOOP_LAYERS = []
BENCHMARK_NAMES = list(BENCHMARK_INFO.keys())


def check_fused_layer_count(model_path, program):
    model = onnx.load(model_path)
    onnx_layer_count = len(model.graph.node)
    layer_count = 0
    onnx_layers = defaultdict(int)
    cdlt_layers = defaultdict(int)
    FUSION_OP_INFO = load_fusion_op_info(program.hag.meta_cfg)

    for n in model.graph.node:
        if n.op_type not in NOOP_LAYERS:
            onnx_layers[n.op_type] += 1
        else:
            onnx_layer_count -= 1
    unmapped = []
    for c in program.codelets:
        if c.op_name in FUSION_OP_INFO:
            layer_count += len(FUSION_OP_INFO[c.op_name]['seq'])
            for o in FUSION_OP_INFO[c.op_name]['seq']:
                if o.lower() not in FUSION_NAME_MAPPING:
                    unmapped.append(o.lower())
                else:
                    cdlt_layers[FUSION_NAME_MAPPING[o.lower()]] += 1
                cdlt_layers[o.lower()] += 1
        else:
            cdlt_layers[c.op_name] += 1
            layer_count += 1

    if layer_count != onnx_layer_count:
        print(f"INconsistent layers after fusion compared to onnx:\n"
                           f"Onnx: {onnx_layer_count}\n"
                           f"Codelets: {layer_count}\n"
                           f"Onnx layers: {onnx_layers}\n"
                           f"Codlet layers: {cdlt_layers}")

def count_compute_ops(program):
    per_layer = defaultdict(int)
    per_compute_op = defaultdict(int)
    num_layers = defaultdict(int)
    compute_per_layer = {}
    total = 0
    for c in program.codelets:
        count_per_layer = False
        if c.op_name not in compute_per_layer:
            compute_per_layer[c.op_name] = defaultdict(int)
            count_per_layer = True
        num_layers[c.op_name] += 1
        for o in c.get_ops_by_type("compute"):
            per_layer[c.op_name] += 1
            per_compute_op[o.op_name] += 1
            if count_per_layer:
                compute_per_layer[c.op_name][o.op_name] += 1
            total += 1

    print(f"Total: {total}")
    print(f"Counts by layer:")
    pprint(per_layer)
    print(f"Counts by op type:")
    pprint(per_compute_op)

    print(f"Op Counts per layer:")
    pprint(compute_per_layer)

    print(f"Num Layers:")
    pprint(num_layers)

def compile_benchmark(model_name,
                      cfg_name,
                      identifier=0,
                      verbose=False,
                      filtered_layers=None,
                      stop_stage=None,
                      skip_layers=None,
                      skip_broken_layers=False,
                      only_systolic=False,
                      filter_op_types=None,
                      skip_op_types=None,
                      store_results=True,
                      count_compute=False,
                      check_layer_count=False,
                      conv_tile_constraints=None,
                      dir_ext=None
                      ):
    arch_config = load_config(f"{CWD}/../codelets/examples/genesys/configs/{cfg_name}")
    if dir_ext is None:
        dir_ext = ""
    else:
        assert isinstance(dir_ext, str)

    if model_name in BENCHMARK_NAMES:
        if arch_config['FUSE_LAYERS']:
            assert not only_systolic
            num_layers = BENCHMARK_INFO[model_name]['num_layers_fused']
        else:
            num_layers = BENCHMARK_INFO[model_name]['num_layers_unfused']
    else:
        num_layers = 0


    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    cfg_path = f"{CFG_PATH}/{cfg_name}"
    graph = pm.from_onnx(model_path)
    program, _ = compile_full_model(model_name, cfg_path,
                                 store_compile=False,
                                 dir_ext=None,
                                 added_constr=None,
                                 verbose=verbose,
                                 model_data=None,
                                 fuse_layers=arch_config['FUSE_LAYERS'],
                                 generate_data=False,
                                    graph=graph,
                                    batch_size=arch_config['BATCH_SIZE'])

    if conv_tile_constraints is not None:
        conv_layers = ["conv_bias", "conv_bias_add_relu", "conv_bias_relu"]
        for l in conv_layers:
            if "LEVEL1_hint" not in program.hag.codelets[l].compilation_params.keys():
                program.hag.codelets[l].compilation_params[f'LEVEL1_hint'] = conv_tile_constraints
            else:
                orig = program.hag.codelets[l].compilation_params[f'LEVEL1_hint']
                new_constraint = f"{orig} and {conv_tile_constraints}"
                program.hag.codelets[l].compilation_params[f'LEVEL1_hint'] = new_constraint

    if only_systolic:
        if verbose:
            print(f"Compiling {model_name} without quantization, only systolic layers.")
        assert not arch_config['USE_QUANTIZATION']
        systolic_layers = ["conv_bias", "gemm", "gemm_no_bias", "conv"]
        program.filtered_compile(verbose=verbose, finalize=True, filter_op_types=systolic_layers)
    elif skip_broken_layers:
        if verbose:
            print(f"Compiling {model_name} without broken layers.")
        assert 'fused_skipped' in BENCHMARK_INFO[model_name] and arch_config['FUSE_LAYERS']
        all_layers = [i for i in range(num_layers) if i not in BENCHMARK_INFO[model_name]['fused_skipped']]
        program.filtered_compile(all_layers, verbose=verbose, finalize=True, filter_op_types=filter_op_types)
    elif filtered_layers:
        assert skip_layers is None
        assert isinstance(filtered_layers, list)
        program.filtered_compile(filtered_layers, verbose=verbose, finalize=True, filter_op_types=filter_op_types)
    elif skip_layers:
        assert filtered_layers is None
        all_layers = [i for i in range(num_layers) if i not in skip_layers]
        program.filtered_compile(all_layers, verbose=verbose, finalize=True, filter_op_types=filter_op_types)
    elif filter_op_types:
        if verbose:
            print(f"Performing full compilation of {model_name} for layers {filter_op_types}.")
        program.filtered_compile(verbose=verbose, finalize=True, filter_op_types=filter_op_types)
    elif skip_op_types:
        assert isinstance(skip_op_types, list)
        if verbose:
            print(f"Performing full compilation of {model_name}, skipping layers {skip_op_types}.")
        program.filtered_compile(verbose=verbose, finalize=True, skip_op_types=skip_op_types)
    else:
        if verbose:
            print(f"Performing full compilation of {model_name}.")
        program.compile(verbose=verbose, finalize=True, stop_stage=stop_stage)
        if check_layer_count:
            check_fused_layer_count(model_path, program)

    # cdlt = program.codelets[0]
    # for o in cdlt.all_operands:
    #     print(f"{o.name} - {o.data_moves[0].src_node} - {o.data_moves[0].op_name}")
    # print(f"{cdlt.cdlt_uid}")

    if stop_stage is None and store_results:
        sys_array_size = arch_config['ARRAY_M']
        dgen = DataGen(program,
                       single_codelets=not arch_config['SHARED_DATAGEN'],
                       shared_datagen=arch_config['SHARED_DATAGEN'],
                       dir_ext=f"{dir_ext}benchmark{sys_array_size}x{sys_array_size}",
                       identifier=identifier,
                       generate_data=arch_config['DATAGEN'],
                       verbose=verbose,
                       out_path=f"{CWD}/compilation_output",
                        store_whole_program=arch_config['SINGLE_PROGRAM_COMPILATION'])
        dgen.generate()
    def print_attr(name):
        print(f"{name} depth: {program.hag.get_subgraph_node(name).depth}")
        print(f"{name} width: {program.hag.get_subgraph_node(name).width}")
        print(f"{name} banks: {program.hag.get_subgraph_node(name).banks}")
        print(f"{name} access type: {program.hag.get_subgraph_node(name).access_type}")
        print(f"{name} buffering: {program.hag.get_subgraph_node(name).buffering_scheme}")
        print(f"{name} indirection: {program.hag.get_subgraph_node(name).indirection}")
        print(f"")


    if count_compute:
        count_compute_ops(program)

def run_benchmarks(benchmarks,
                   cfg,
                   parallel=True,
                   **kwargs
                   ):

    if parallel:
        kwargs['verbose'] = False
        bench_pool = mp.Pool()
        bench_pool.map(partial(compile_benchmark, cfg, **kwargs), benchmarks)
    else:
        for b in benchmarks:
            print(f"Compiling {b}")
            compile_benchmark(b, cfg, **kwargs)

def neuro_benchmarks():
    t = "oc_ic_oh_ow"
    tilings = {
        "oc": "splits['OC'] > 1 and splits['IC'] == 1 and splits['OH'] == 1 and splits['OW'] == 1",
        "oc_oh": "splits['OC'] > 1 and splits['IC'] == 1 and splits['OH'] > 1 and splits['OW'] == 1",
        "oc_ow": "splits['OC'] > 1 and splits['IC'] == 1 and splits['OH'] == 1 and splits['OW'] > 1",
        "oc_oh_ow": "splits['OC'] > 1 and splits['IC'] == 1 and splits['OH'] > 1 and splits['OW'] > 1",
        "ic_oh_ow": "splits['OC'] == 1 and splits['IC'] > 1 and splits['OH'] > 1 and splits['OW'] > 1",
        "oc_ic_oh_ow": "splits['OC'] > 1 and splits['IC'] > 1 and splits['OH'] > 1 and splits['OW'] > 1",
    }

    # Model selection
    m = 0

    onnx_models = ['conv_only_k3-v1', 'conv_only_k1-v1', 'conv_benchmark_v1', 'conv_lrelu_add_v1-opt', 'conv_add_relu_v0-opt']

    layer = []
    # layer = [35]

    config = "neuro_cfg_8x8.json"
    dir_ext = f"neuro_unfused_{t}"

    compile_benchmark(onnx_models[m],
                      config,
                      only_systolic=False,
                      verbose=True,
                      conv_tile_constraints=tilings[t],
                      skip_broken_layers=False,
                      dir_ext=dir_ext,
                      # filtered_layers=layer,
                      identifier=0)

if __name__ == "__main__":
    if sys.stdin and sys.stdin.isatty():
    # if False:
        argparser = argparse.ArgumentParser(description='ONNX Benchmark Generator')
        argparser.add_argument('-m', '--model', required=True,
                               help='Name of the onnx model to create.')

        argparser.add_argument('-c', '--config', required=True,
                               help='Name of the architecture config file to use.')

        argparser.add_argument('-v', '--verbose', action='store_true', help='Use verbose compilation output')

        argparser.add_argument('-e', '--extension', type=str, default="0", help="Apply an extension to the compilation output directory name.")
        args = argparser.parse_args()

        fname = args.model
        if ".onnx" in fname:
            fname = fname.replace(".onnx", "")

        extension = args.extension
        verbose = args.verbose
        arch_config = args.config

        # compile_benchmark(fname,
        #                   arch_config,
        #                   only_systolic=False,
        #                   sw_pipeline_test=False,
        #                   addr_gen_test=False,
        #                   custom_config=False,
        #                   verbose=verbose,
        #                   skip_broken_layers=False,
        #                   identifier=extension)
        compile_benchmark(fname,
                          arch_config,
                          only_systolic=False,
                          verbose=verbose,
                          skip_broken_layers=False,
                          # filtered_layers=[0],
                          dir_ext=extension,
                          identifier=8)

    else:
        # config = "simd_paper32x32.json"

        # config = "simd_paper8x8_dse.json"
        # config = "simd_paper16x16_dse.json"
        # config = "simd_paper32x32_dse.json"
        # config = "simd_paper64x64_dse.json"
        # config = "simd_paper128x128_dse.json"
        # config = "paper_fpga16x16.json"
        # config = "fpga16x16.json"
        config = "fpga4x4.json"
        # config = "neuro_cfg_8x8.json"

        # dir_ext = "_default"
        # dir_ext = "_tpu_test"
        # dir_ext = "_ld_st"
        # dir_ext = "_addr_gen"
        # dir_ext = "_loop_overhead"
        # dir_ext = "_sw_pipeline"
        # dir_ext = "unfused_"
        dir_ext = ""

        benchmarks = ['resnet18', # 0
                      'resnet50', # 1
                      'efficientnet-lite4-opt-no-softmax', # 2
                      'mobilenetv2-opt', # 3
                      'yolov3-opt-static', # 4
                      'bert-base-cased-transpose-opt-trimmed-ort', # 5
                      "vgg16", # 6
                      'gpt2-trimmed-opt', # 7
                      "vit-pad256-transpose-ort", # 8
                      "custom_fft", # 9
                      "custom_small_fft", # 10
                      "resnet50_train", # 11
                      "resnet18_train", # 12
                      'conv_clip_depthwise_c32_w112_kw1', # 13
                      'conv_lrelu_add_oc64_v3-opt', # 14
                      'conv_lrelu_oc64', # 15
                      'conv_clip_depthwise_v1-opt', # 16
                      'fcn-resnet101-trimmed-opt', # 17
                      'mel_scale',# # 18,
                      'normalize-200-opt', # 19
                      'linear_reg-opt', # 20
                      'logistic_reg-opt', # 21
                      'linear_reg_test', # 22
                      'ppo_model', # 23,
                      'conv_benchmark_v1', # 24
                      'div_test', # 25,
                      'conv_lrelu_add_v1-opt', # 26
                      'conv_add_relu_v0-opt', # 27
                      "conv_benchmark_v1", # 28
                      'ddpg_model-opt',  # 29
                      'sac_model',  # 30
                      'ppo_model',  # 31,
                      'ddpg', # 32,
                      'short-gpt2-trimmed-opt', # 33
                      ]
        # neuro_benchmarks()
        simd_paper_benches = [1, 3, 4, 2, 6, 5]
        compile_benchmark(benchmarks[31],
                          config,
                          only_systolic=False,
                          verbose=True,
                          skip_broken_layers=False,
                          # filtered_layers=[0],
                          dir_ext=dir_ext,
                          identifier=8)
