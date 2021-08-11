import argparse
import os
from pathlib import Path
import json
from codelets.compiler.program import CodeletProgram
from examples.genesys.genesys_qmodels import generate_random_values
from examples.genesys import compile_genesys_layer, compile_genesys, get_arch
import pprint

ALL_LAYER_NAMES = ["resnet18_relu", "resnet18_add", "resnet18_conv", "resnet18_conv_bias", "resnet18_gemm", "resnet18_globalaveragepool",
                   "resnet18_train_batchnormalization", "lenet_averagepool", "lenet_conv", "lenet_gemm", "lenet_bn_conv", "custom_conv_conv"]
ALL_MODEL_NAMES = ["resnet18", "resnet50", "lenet", "lenet_bn", "custom_conv"]
ALL_MODEL_TRAIN_NAMES = ["resnet18_train", "resnet50_train", "lenet_train"]


CWD = Path(f"{__file__}").parent
BENCH_DIR = f"{CWD}/input_files"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_dirs(fpath, dir_ext):
    cwd = Path(f"{__file__}").parent
    base_path = Path(f"{cwd}/compilation_output/{Path(fpath).stem}{dir_ext}")

    if not Path(f"{base_path}").exists():
        try:
            os.makedirs(base_path)
        except OSError as e:
            print(f"Creation of directory {base_path} failed:\n {e}")
        else:
            print(f"Successfully created of directory {base_path}")
    else:
        print(f"Directory {base_path} already exists.")
    return base_path

def store_values(program, model_name, base_path, load_path=None, use_random=True, actual_data=False):

    cdlt = program.codelets[0]
    if load_path:
        fixed_values = {"folder_path": f"{CWD}/compilation_output/{load_path}"}
    else:
        fixed_values = None
    generate_random_values(cdlt, model_name, cdlt.op_name,
                           base_path=base_path,
                           use_random=use_random,
                           fixed_values=fixed_values,
                           actual_data=actual_data)


def store_outputs(model_name,
                  layer_name,
                  training_mode,
                  batch_size=1,
                  verbose=False,
                  emit_to_stdout=None,
                  load_path=None,
                  dir_ext=None,
                  actual_data=False,
                  use_random=False):
    name = model_name

    if layer_name is not None:
        name = f"{name}_{args.layer_name}"
    elif training_mode:
        name = f"{name}_train"
    tiling_path = None
    store_tiling = False
    store_json_output = False
    json_output_filename = None
    update_cfg_dtypes = False

    BENCH_DIR = Path(f"{CWD}/../benchmarks").absolute()
    print(f"Creating compilation output for {name}\n")

    if name in ALL_LAYER_NAMES:
        program = compile_genesys_layer(name,
                                  update_cfg_dtypes=update_cfg_dtypes,
                                  tiling_path=tiling_path,
                                  store_tiling=store_tiling,
                                  store_checkpoint=False,
                                  store_json_output=store_json_output,
                                  json_output_filename=json_output_filename,
                                  verbose=verbose,
                                  benchmark_path=BENCH_DIR,
                                  factor_fn='default',
                                batch_size=batch_size,
                                do_hoist_stage=True,
                                do_tile_stage=True,
                                print_config=False
                                  )
    elif name in ALL_MODEL_NAMES:
        program = compile_genesys(name,
                                  train=False,
                                  update_cfg_dtypes=update_cfg_dtypes,
                                  tiling_path=tiling_path,
                                  batch_size=batch_size,
                                  store_tiling=store_tiling,
                                  store_json_output=store_json_output,
                                  json_output_filename=json_output_filename,
                                  verbose=verbose,
                                  benchmark_path=BENCH_DIR,
                                  factor_fn='default',
                                  print_config=False
                                  )
    elif name in ALL_MODEL_TRAIN_NAMES:
        name = name.split("_")[0]
        program = compile_genesys(name,
                                  train=True,
                                  update_cfg_dtypes=update_cfg_dtypes,
                                  tiling_path=tiling_path,
                                  batch_size=batch_size,
                                  store_tiling=store_tiling,
                                  store_json_output=store_json_output,
                                  json_output_filename=json_output_filename,
                                  verbose=verbose,
                                  benchmark_path=BENCH_DIR,
                                  factor_fn='default',
                                  print_config=False
                                  )
    else:
        raise RuntimeError(f"Invalid layer name for compilation : {name}")

    arch_cfg = get_arch(None, None, update_cfg_dtypes)
    print(f"Configuration for program:")
    pprint.pprint(arch_cfg)
    if emit_to_stdout is not None:
        assert isinstance(emit_to_stdout, str)
        if "json" in emit_to_stdout:
            pprint.pprint(program.emit(emit_to_stdout))
        else:
            print(program.emit(emit_to_stdout))

    base_path = store_compilation_output(program, "arch_cfg", extension="json", arch_cfg=arch_cfg, dir_ext=dir_ext)
    store_compilation_output(program, "operations_idx", extension="txt", dir_ext=dir_ext)
    store_compilation_output(program, "json", extension="json", dir_ext=dir_ext)
    store_compilation_output(program, "string_final", extension="txt", dir_ext=dir_ext)
    store_compilation_output(program, "decimal", extension="txt", dir_ext=dir_ext)
    store_compilation_output(program, "binary", extension="txt", dir_ext=dir_ext)
    store_values(program, model_name, base_path, use_random=use_random, load_path=load_path, actual_data=actual_data)

def store_compilation_output(program: CodeletProgram, output_type, extension="txt", dir_ext=None, arch_cfg=None):
    if dir_ext:
        dir_ext = f"_{dir_ext}"
    else:
        dir_ext = ""
    out_path = create_dirs(program.name, dir_ext)
    if output_type == "arch_cfg":
        result = arch_cfg
    else:
        result = program.emit(output_type)
    if not isinstance(result, str):
        assert isinstance(result, dict)
        result = json.dumps(result, indent=2)
    with open(f"{out_path}/{program.name}_{output_type}.{extension}", "w") as outfile:
        outfile.write(result)
    return out_path

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='ONNX Benchmark Generator')
    argparser.add_argument('-m', '--model_name', required=True,
                           help='Name of the benchmark to compile.')
    argparser.add_argument('-de', '--dir_ext', required=False, default=None,
                           help='Storage name for directory')
    argparser.add_argument('-lp', '--load_path', required=False, default=None,
                           help='Load previous values for compilation test values.')
    argparser.add_argument('-l', '--layer_name', required=False, default=None,
                           help='Type fo output format')
    argparser.add_argument('-t', '--training_mode', type=str2bool, nargs='?', default=False,
                           const=True, help='Whether or not the model is in training mode')
    argparser.add_argument('-e', '--emit_to_stdout', required=False, default=None,
                           help='If unset, does not emit the compiled program.'
                                'Otherwise, emits using the specified output type.')
    argparser.add_argument('-r', '--use_random',type=str2bool, nargs='?', default=True,
                           const=True, help='Compile layer with randomized output')
    argparser.add_argument('-a', '--actual_data',type=str2bool, nargs='?', default=True,
                           const=True, help='Compile layer with actual data from layer')
    argparser.add_argument('-v', '--verbose',type=str2bool, nargs='?', default=False,
                           const=True, help='Compiel with verbose output')
    argparser.add_argument('-bs', '--batch_size', default=1, type=int)
    args = argparser.parse_args()

    store_outputs(args.model_name, args.layer_name, args.training_mode,
                  args.batch_size,
                  args.verbose,
                  args.emit_to_stdout,
                  use_random=args.use_random,
                  dir_ext=args.dir_ext,
                  actual_data=args.actual_data)
