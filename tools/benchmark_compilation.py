import json
import polymath as pm
from examples.genesys import GENESYS_DTYPES, GENESYS_CFG, USE_QUANTIZATION, \
    SW_PIPELINE_TEST, ADDR_GEN_TEST, PAPER_CFG2, PAPER_CFG1, CUSTOM_CFG
from examples.genesys.codelets import FUSION_OP_INFO
from examples.genesys.genesys_network_sim import compile_full_model
from tools.compile_layer import store_program_codelets
from collections import defaultdict
from pathlib import Path
import os
import onnxruntime as ort
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
import onnx

MODEL_DIR = Path(f"{Path(__file__).parent}/../benchmarks/models")
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
}

NOOP_LAYERS = []
BENCHMARK_NAMES = list(BENCHMARK_INFO.keys())


def check_fused_layer_count(model_path, program):
    model = onnx.load(model_path)
    onnx_layer_count = len(model.graph.node)
    layer_count = 0
    onnx_layers = defaultdict(int)
    cdlt_layers = defaultdict(int)

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

    print(f"Unmapped: {set(unmapped)}")
    if layer_count != onnx_layer_count:
        print(f"INconsistent layers after fusion compared to onnx:\n"
                           f"Onnx: {onnx_layer_count}\n"
                           f"Codelets: {layer_count}\n"
                           f"Onnx layers: {onnx_layers}\n"
                           f"Codlet layers: {cdlt_layers}")



def compile_benchmark(model_name,
                      fuse_layers=False,
                      identifier=0,
                      custom_config=False,
                      verbose=False,
                      filtered_layers=None,
                      stop_stage=None,
                      skip_layers=None,
                      skip_broken_layers=False,
                      only_systolic=False,
                      filter_op_types=None,
                      sw_pipeline_test=False,
                      addr_gen_test=False
                      ):
    dir_ext = ""
    if fuse_layers:
        assert not only_systolic
        num_layers = BENCHMARK_INFO[model_name]['num_layers_fused']
    else:
        num_layers = BENCHMARK_INFO[model_name]['num_layers_unfused']

    if custom_config:
        assert CUSTOM_CFG

        assert USE_QUANTIZATION
        assert not SW_PIPELINE_TEST
        assert not ADDR_GEN_TEST
        assert not PAPER_CFG1
        assert not PAPER_CFG2
        dir_ext = "dse_"
    elif sw_pipeline_test:
        assert not USE_QUANTIZATION
        assert SW_PIPELINE_TEST
        assert not ADDR_GEN_TEST
        assert PAPER_CFG2
        dir_ext = "sw_pipeline_"
    elif addr_gen_test:
        assert ADDR_GEN_TEST
        assert USE_QUANTIZATION
        assert PAPER_CFG2
        assert not SW_PIPELINE_TEST
        dir_ext = "addr_gen_"
    else:
        assert not SW_PIPELINE_TEST
        assert not ADDR_GEN_TEST
    assert model_name in BENCHMARK_NAMES
    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    graph = pm.from_onnx(model_path)
    program = compile_full_model(model_name,
                                 store_compile=False,
                                 dir_ext=None,
                                 added_constr=None,
                                 train_mode=False,
                                 verbose=verbose,
                                 model_data=None,
                                 fuse_layers=fuse_layers,
                                 generate_data=False,
                                    graph=graph
                                     )

    if only_systolic:
        if verbose:
            print(f"Compiling {model_name} without quantization, only systolic layers.")
        assert not USE_QUANTIZATION
        systolic_layers = ["conv_bias", "gemm", "gemm_no_bias", "conv"]
        program.filtered_compile(verbose=verbose, finalize=True, filter_op_types=systolic_layers)
    elif skip_broken_layers:
        if verbose:
            print(f"Compiling {model_name} without broken layers.")
        assert 'fused_skipped' in BENCHMARK_INFO[model_name] and fuse_layers
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
    else:
        if verbose:
            print(f"Performing full compilation of {model_name}.")
        program.compile(verbose=verbose, finalize=True, stop_stage=stop_stage)
        check_fused_layer_count(model_path, program)


    if stop_stage is None:
        sys_array_size = GENESYS_CFG['ARRAY_M']
        store_program_codelets(program, identifier, dir_ext=f"{dir_ext}benchmark{sys_array_size}x{sys_array_size}")

if __name__ == "__main__":
    benchmarks = ['resnet18', 'resnet50',
                  'efficientnet-lite4-opt-no-softmax',
                  'mobilenetv2-opt',
                  'yolov3-opt-static',
                  'bert-base-cased-transpose-opt-trimmed-ort']
    #
    # compile_benchmark(benchmarks[-1],
    #                   fuse_layers=True,
    #                   only_systolic=False,
    #                   sw_pipeline_test=False,
    #                   addr_gen_test=False,
    #                   custom_config=True,
    #                   verbose=True,
    #                   # filtered_layers=[3],
    #                   # filter_op_types=['conv_bias_clip_depthwise_conv_bias_clip'],
    #                   skip_broken_layers=False,
    #                   identifier=24)

    compile_benchmark(benchmarks[-1],
                      fuse_layers=True,
                      only_systolic=False,
                      verbose=True,
                      addr_gen_test=True,
                      custom_config=False,
                      sw_pipeline_test=False,
                      skip_broken_layers=False,
                      identifier=25)

    #
    # for b in benchmarks[1:]:
    #     compile_benchmark(b,
    #                       fuse_layers=True,
    #                       only_systolic=False,
    #                       verbose=True,
    #                       addr_gen_test=True,
    #                       custom_config=False,
    #                       sw_pipeline_test=False,
    #                       skip_broken_layers=False,
    #                       identifier=25)