import argparse
import os
import numpy as np
import json
import subprocess

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
        "num_layers_unfused": 655,
        "num_layers_fused": 0,
        "fuse_skipped": []
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
        skip_layers = [skip_layer if skip_layer >= 0 else num_layers + skip_layer for skip_layer in skip_layers]
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


def run_micro_tutorial_2023_systolic_size_sweep_variable_memory_depth_benchmarks() -> None:
    models = [
            "resnet50",
            "bert-base-cased-transpose-opt-trimmed-ort"
        ]
    default_config_values = {
        "PARAM_BUF_CHANNEL_BW": 512,
        "DRAM_DEPTH": 50000000,
        "SA_TILE_CONSTR": True,
        "USE_QUANTIZATION": False,
        "ALL_QUANT_OFF": True,
        "FUSION_CONSTRAINTS": False,
	    "FUSE_LAYERS": False,
        "SINGLE_PROGRAM_COMPILATION": False,
        "DATAGEN": False
    }

    systolic_array_widths = [4, 8, 16, 32, 64, 128, 256]
    ibuf_depths = [32768, 16384, 8192, 4096, 2048, 1024, 512]
    wbuf_depths = [32768, 16384, 4096, 1024, 256, 128, 512]
    obuf_depths = [8192, 4096, 2048, 1024, 1024, 1024, 1024]
    bbuf_depths = [4096, 2048, 1024, 512, 256, 512, 64]

    configs = []
    for i in range(len(systolic_array_widths)):
        config = default_config_values.copy()
        config["ARRAY_M"] = systolic_array_widths[i]
        config["ARRAY_N"] = systolic_array_widths[i]
        config["IBUF_DEPTH"] = ibuf_depths[i]
        config["WBUF_DEPTH"] = wbuf_depths[i]
        config["OBUF_DEPTH"] = obuf_depths[i]
        config["BBUF_DEPTH"] = bbuf_depths[i]
        configs.append(config)
    
    for config in configs[0:]:
        for model in models:
            config_file_path = "../codelets/examples/genesys/configs/micro_tutorial_2023_systolic_size_sweep_variable_memory_depth.json"
            if os.path.exists(config_file_path):
                os.remove(config_file_path)
            with open(config_file_path, "w") as f:
                json.dump(config, f, indent=4)
            skip_layers = [-1] if model == "bert-base-cased-transpose-opt-trimmed-ort" else None
            compile_benchmark(model, "micro_tutorial_2023_systolic_size_sweep_variable_memory_depth.json", verbose=False, only_systolic=False, skip_layers=skip_layers, identifier="micro_tutorial_2023_layer_by_layer_systolic_size_sweep_variable_memory_depth")
            output_directory = f"compilation_output/{model}_benchmark{config['ARRAY_M']}x{config['ARRAY_N']}_micro_tutorial_2023_layer_by_layer_systolic_size_sweep_variable_memory_depth" 
            assert os.path.exists(output_directory), f"Output directory {output_directory} does not exist."
            subprocess.run(["zip", "-r", f"{model}_{config['ARRAY_M']}x{config['ARRAY_N']}_micro_tutorial_2023_layer_by_layer_systolic_size_sweep_variable_memory_depth.zip", output_directory])
            os.remove(config_file_path)
            print(f"Successfully created {model}_{config['ARRAY_M']}x{config['ARRAY_N']}_micro_tutorial_2023_layer_by_layer_systolic_size_sweep_variable_memory_depth.zip")


def run_micro_tutorial_2023_systolic_array_size_sweep_fusion_on_off() -> None:
    models = [
            "resnet50",
            "bert-base-cased-transpose-opt-trimmed-ort"
        ]
    default_config_values = {
        "IBUF_DEPTH": 32768,
        "WBUF_DEPTH": 32768,
        "OBUF_DEPTH": 8192,
        "BBUF_DEPTH": 4096,
        "PARAM_BUF_CHANNEL_BW": 512,
        "DRAM_DEPTH": 50000000,
        "SA_TILE_CONSTR": True,
        "USE_QUANTIZATION": False,
        "ALL_QUANT_OFF": True, 
        "SINGLE_PROGRAM_COMPILATION": False,
        "DATAGEN": False
    }

    systolic_array_widths = [4, 8, 16, 32, 64, 128, 256]
    configs = []
    for i in range(len(systolic_array_widths)):
        config = default_config_values.copy()
        config["ARRAY_M"] = systolic_array_widths[i]
        config["ARRAY_N"] = systolic_array_widths[i]
        config["FUSION_CONSTRAINTS"] = False
        config["FUSE_LAYERS"] = False
        configs.append(config)
        config = configs[-1].copy()
        config["FUSION_CONSTRAINTS"] = True
        config["FUSE_LAYERS"] = True
        configs.append(config)
    
    for config in configs[0:]:
        for model in models:
            config_file_path = "../codelets/examples/genesys/configs/micro_tutorial_2023_systolic_size_sweep_fusion_on_off.json"
            if os.path.exists(config_file_path):
                os.remove(config_file_path)
            with open(config_file_path, "w") as f:
                json.dump(config, f, indent=4)
            skip_layers = [-1] if model == "bert-base-cased-transpose-opt-trimmed-ort" and not config["FUSE_LAYERS"] else None
            compile_benchmark(model, "micro_tutorial_2023_systolic_size_sweep_fusion_on_off.json", verbose=False, only_systolic=False, skip_layers=skip_layers, identifier="micro_tutorial_2023_layer_by_layer_systolic_size_sweep_fusion_on_off")
            output_directory = f"compilation_output/{model}_benchmark{config['ARRAY_M']}x{config['ARRAY_N']}_micro_tutorial_2023_layer_by_layer_systolic_size_sweep_fusion_on_off" 
            assert os.path.exists(output_directory), f"Output directory {output_directory} does not exist."
            subprocess.run(["zip", "-r", f"{model}_{config['ARRAY_M']}x{config['ARRAY_N']}_micro_tutorial_2023_layer_by_layer_systolic_size_sweep_fusion_on_off.zip", output_directory])
            os.remove(config_file_path)
            print(f"Successfully created {model}_{config['ARRAY_M']}x{config['ARRAY_N']}_micro_tutorial_2023_layer_by_layer_systolic_size_sweep_fusion_on_off.zip")


def run_micro_tutorial_2023_systolic_array_size_sweep_quantization_on_off() -> None:
    models = [
            "resnet50",
            "bert-base-cased-transpose-opt-trimmed-ort"
        ]
    default_config_values = {
        "IBUF_DEPTH": 32768,
        "WBUF_DEPTH": 32768,
        "OBUF_DEPTH": 8192,
        "BBUF_DEPTH": 4096,
        "PARAM_BUF_CHANNEL_BW": 512,
        "DRAM_DEPTH": 50000000,
        "FUSION_CONSTRAINTS": False,
	    "FUSE_LAYERS": False,
        "SA_TILE_CONSTR": True,
        "SINGLE_PROGRAM_COMPILATION": False,
        "DATAGEN": False
    }

    systolic_array_widths = [4, 8, 16, 32, 64, 128, 256]
    configs = []
    for i in range(len(systolic_array_widths)):
        config = default_config_values.copy()
        config["ARRAY_M"] = systolic_array_widths[i]
        config["ARRAY_N"] = systolic_array_widths[i]
        config["USE_QUANTIZATION"] = False
        config["ALL_QUANT_OFF"] = True
        configs.append(config)
        config = configs[-1].copy()
        config["USE_QUANTIZATION"] = True
        config["ALL_QUANT_OFF"] = False
        configs.append(config)
    
    for config in configs[0:]:
        for model in models:
            config_file_path = "../codelets/examples/genesys/configs/micro_tutorial_2023_systolic_size_sweep_quantization_on_off.json"
            if os.path.exists(config_file_path):
                os.remove(config_file_path)
            with open(config_file_path, "w") as f:
                json.dump(config, f, indent=4)
            skip_layers = [-1] if model == "bert-base-cased-transpose-opt-trimmed-ort" and not config["FUSE_LAYERS"] else None
            compile_benchmark(model, "micro_tutorial_2023_systolic_size_sweep_quantization_on_off.json", verbose=False, only_systolic=False, skip_layers=skip_layers, identifier="micro_tutorial_2023_layer_by_layer_systolic_size_sweep_quantization_on_off")
            output_directory = f"compilation_output/{model}_benchmark{config['ARRAY_M']}x{config['ARRAY_N']}_micro_tutorial_2023_layer_by_layer_systolic_size_sweep_quantization_on_off" 
            assert os.path.exists(output_directory), f"Output directory {output_directory} does not exist."
            subprocess.run(["zip", "-r", f"{model}_{config['ARRAY_M']}x{config['ARRAY_N']}_micro_tutorial_2023_layer_by_layer_systolic_size_sweep_quantization_on_off.zip", output_directory])
            os.remove(config_file_path)
            print(f"Successfully created {model}_{config['ARRAY_M']}x{config['ARRAY_N']}_micro_tutorial_2023_layer_by_layer_systolic_size_sweep_quantization_on_off.zip")


if __name__ == "__main__":
    run_specific_benchmarks: bool = True
    # if sys.stdin and sys.stdin.isatty():
    if not run_specific_benchmarks:
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

        compile_benchmark(fname,
                          arch_config,
                          only_systolic=False,
                        #   sw_pipeline_test=False,
                        #   addr_gen_test=False,
                        #   custom_config=False,
                          verbose=verbose,
                          skip_broken_layers=False,
                          identifier=extension)
    else:
        # run_micro_tutorial_2023_systolic_size_sweep_variable_memory_depth_benchmarks()
        run_micro_tutorial_2023_systolic_array_size_sweep_fusion_on_off()
