from pathlib import Path
import shutil
from layer_generator import update_tile_constraints
from benchmarks.load_onnx_model import store_unique_model_layers, convert_model_to_polymath
from codelets.examples import compile_genesys_layer
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

SUPPORTED_MODELS = ["resnet50", "resnet18"]

def compile_layer(model_name, layer_name, layer_id, identifier, batch_size = 1,
                  tile_method="min_tiles",
                  store_compile=True,
                  dir_ext=None,
                  added_constr=None,
                  fuse_layers=False,
                  generate_data=False):
    update_cfg_dtypes = False
    tiling_path = None
    store_tiling = False
    store_json_output = False
    json_output_filename = None
    full_layer_name = f"{model_name}_test{identifier}_{layer_name}{layer_id}"
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
                                    do_compile=False,
                                    fuse_layers=fuse_layers
                                    # relocation_offsets=reloc_offsets
                              )

    if store_compile:
        if added_constr:
            program = update_tile_constraints(program, added_constr, layer_name)
        dir_ext = dir_ext or ''
        # program.compile(verbose=False, finalize_instructions=True)

        store_outputs("cc_layer1", full_layer_name, False,
                      1,
                      False,
                      None,
                      use_random=True,
                      dir_ext=f"{dir_ext}",
                      actual_data=False,
                      store_partials=False,
                      program=program,
                      generate_data=generate_data)
    return program

def compile_network(model_name, identifier=0, generate_data=False, fuse_layers=False):
    skip_layers = ["flatten"]
    assert model_name in SUPPORTED_MODELS
    model_location = f"{MODEL_DIR}/{model_name}.onnx"
    output_location = f"{OUT_DIR}/{model_name}_{identifier}"
    if not Path(output_location).exists():
        try:
            os.makedirs(output_location)
        except OSError as e:
            print(f"Creation of directory {output_location} failed:\n {e}")
        else:
            print(f"Successfully created of directory {output_location}")
    else:
        raise RuntimeError(f"Directory {output_location} already exists.")

    model = onnx.load_model(model_location)
    layer_info = defaultdict(int)
    for n in model.graph.node:
        layer_id = n.op_type.lower()
        if layer_id in skip_layers:
            continue
        layer_num = layer_info[layer_id]
        layer_path_name = f"{model_name}_test{identifier}_{layer_id}{layer_num}"
        layer_path = f"{LAYER_DIR}/{layer_path_name}.onnx"
        layer_info[layer_id] += 1
        inputs = n.input
        outputs = n.output
        onnx.utils.extract_model(model_location, layer_path, inputs, outputs)
        convert_model_to_polymath(layer_path)
        program = compile_layer(model_name, layer_id, layer_num, identifier)
        assert Path(f"{OUT_DIR}/{layer_path_name}").exists()
        assert not Path(f"{output_location}/{layer_path_name}").exists()
        shutil.move(f"{OUT_DIR}/{layer_path_name}", f"{output_location}/")


if __name__ == "__main__":
    compile_network("resnet50")