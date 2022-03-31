import json
import polymath as pm
from examples.genesys import GENESYS_DTYPES, GENESYS_CFG
from examples.genesys.codelets import FUSION_OP_INFO
from examples.genesys.genesys_network_sim import compile_full_model
from tools.compile_layer import store_program_codelets

from pathlib import Path
import os

import onnx

MODEL_DIR = Path(f"{Path(__file__).parent}/../benchmarks/models")


BENCHMARK_NAMES = ["resnet18",
                   "resnet50",
                   "efficientnet-lite4-11-opt-no-softmax",
                   "mobilenet27-opt",
                   "yolov3-opt-static",
                   "bertsquad-12-opt1"]

def check_fused_layer_count(model_path, program):
    model = onnx.load(model_path)
    onnx_layers = len(model.graph.node)
    layer_count = 0
    for c in program.codelets:
        if c.op_name in FUSION_OP_INFO:
            layer_count += len(FUSION_OP_INFO[c.op_name]['seq'])
        else:
            layer_count +=1

    if layer_count != onnx_layers:
        raise RuntimeError(f"INconsistent layers after fusion compared to onnx:\n"
                           f"Onnx: {onnx_layers}\n"
                           f"Codelets: {layer_count}\n")



def compile_benchmark(model_name,
                      fuse_layers=False,
                      generate_data=False,
                      identifier=0,
                      verbose=False,
                      filtered_layers=None):
    skip_layers = ["flatten"]
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
    if filtered_layers:
        assert isinstance(filtered_layers, list)
        program.filtered_compile(filtered_layers, verbose=verbose, finalize=True)
    else:
        program.compile(verbose=verbose, finalize=True)
    if fuse_layers and not filtered_layers:
        check_fused_layer_count(model_path, program)
    store_program_codelets(program, identifier, dir_ext="benchmark")



if __name__ == "__main__":
    # compile_benchmark('efficientnet-lite4-11-opt-no-softmax', fuse_layers=True, verbose=True)
    # compile_benchmark('mobilenet27-opt', fuse_layers=True, verbose=True)
    compile_benchmark('resnet18', fuse_layers=True,
                      verbose=False,
                      filtered_layers=[0],
                      identifier=4)