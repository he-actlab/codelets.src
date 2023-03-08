from codelets.examples import compile_genesys, compile_genesys_layer
from pprint import pprint
import argparse
from pathlib import Path

CWD = Path(f"{__file__}").parent
TEST_DIR = f"{CWD}/input_files"
BENCH_DIR = f"{CWD}/../benchmarks"
MODEL_DIR = f"{BENCH_DIR}/models/srdfg"
LAYER_DIR = f"{BENCH_DIR}/layers/srdfg"
TILING_DIR = f"{BENCH_DIR}/tiling_info"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def run_model(model_name, train, batch_size, factor_fn):
    MODEL_NAMES = ['resnet18', 'resnet50']
    # GENESYS_DTYPES['SIMD'] = 'FXP16'
    # GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight'] = 'FXP4'
    # GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out'] = 'FXP16'
    # If you update the genesys datatypes above, set 'update_cfg_dtypes' to 'True'
    update_cfg_dtypes = False

    # If there is an existing tiling file for this particular model, set the tiling path here
    # If this is set to None, then it will re-tile.
    # NOTE: if you are compiling a training program, the filename should be f"{model_name}_train_tiling_info.json"

    # tiling_path = f"{model_name}_train_training_tiling_info_checkpoint0.json"
    tiling_path = None


    # If you had previously never stored tiling for this program, store it
    store_tiling = False

    # Whether or not to store the compiler output as json.
    # If you want to specify the filename, set 'json_output_filename' to a string name
    store_json_output = False
    json_output_filename = None
    BENCH_DIR = Path(f"{CWD}/../benchmarks").absolute()
    if model_name in MODEL_NAMES:
        # This function returns
        program = compile_genesys(model_name,
                                  train=train,
                                  update_cfg_dtypes=update_cfg_dtypes,
                                  tiling_path=tiling_path,
                                  batch_size=batch_size,
                                  store_tiling=store_tiling,
                                  store_json_output=store_json_output,
                                  json_output_filename=json_output_filename,
                                  verbose=True,
                                  benchmark_path=BENCH_DIR,
                                  factor_fn=factor_fn
                                  )
        res = program.emit("json_no_ops")
        pprint(res)
    else:
        program = compile_genesys_layer(model_name,
                                        update_cfg_dtypes=update_cfg_dtypes,
                                        tiling_path=tiling_path,
                                        store_tiling=store_tiling,
                                        store_checkpoint=False,
                                        store_json_output=store_json_output,
                                        json_output_filename=json_output_filename,
                                        verbose=True,
                                        benchmark_path=BENCH_DIR,
                                        factor_fn='default',
                                        batch_size=batch_size
                                        )

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='ONNX Benchmark Generator')
    argparser.add_argument('-m', '--model_name', required=True,
                           help='Name of the benchmark to create. One of "resnet18", "lenet')

    argparser.add_argument('-t', '--training_mode', type=str2bool, nargs='?', default=False,
                           const=True, help='Whether or not the model is in training mode')
    argparser.add_argument('-ff', '--factor_function', type=str, default='default')

    argparser.add_argument('-bs', '--batch_size', default=1, type=int)
    args = argparser.parse_args()
    run_model(args.model_name, args.training_mode, args.batch_size, args.factor_function)