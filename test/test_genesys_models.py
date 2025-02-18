from codelets.examples import compile_genesys, get_transformed_srdfg
import pytest
import polymath as pm
from collections import namedtuple
import json
from pathlib import Path
try:
    from .util import validate_program
except:
    from util import validate_program

CWD = Path(f"{__file__}").parent
TEST_DIR = f"{CWD}/input_files"
BENCH_DIR = f"{CWD}/../benchmarks"
MODEL_DIR = f"{BENCH_DIR}/models/srdfg"
LAYER_DIR = f"{BENCH_DIR}/layers/srdfg"
TILING_DIR = f"{BENCH_DIR}/tiling_info"

TestDfgNode = namedtuple('TestDfgNode', ['input_components', 'input_shapes', 'attrs'])
GENESYS_CFG_PATH = f"{CWD}/scratch/genesys_cfg.json"


def parse_cfg():
    with open(GENESYS_CFG_PATH) as f:
        genesys = json.load(f)
    return genesys


def test_srdfg_creation(cfg):
    model_name = 'lenet'

    # Determines whether to compile a training model or not
    train = True

    # If this is changed, the batch size will updated for the model
    batch_size = 1

    BENCH_DIR = Path(f"{CWD}/../benchmarks").absolute()
    graph = get_transformed_srdfg(model_name,
                                  cfg,
                              train=train,
                              batch_size=batch_size,
                              verbose=False,
                              benchmark_path=BENCH_DIR)

    for name, node in graph.nodes.items():
        if not isinstance(node, (pm.placeholder, pm.write)):
            print(f"{name}: {node.op_name}")

@pytest.mark.parametrize('model_name',[
    # "lenetbn",
    # "lenetbn_train",
    # "resnet50_train",
    "resnet18",
    # "resnet50",
    # "resnet18_train",
    # "lenet",
    # "lenet_train"
])
def test_genesys_model(model_name):
    train = False

    if "train" in model_name:
        model_name = model_name.split("_")[0]
        train = True
    # Determines whether to compile a training model or not

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

    # If this is changed, the batch size will updated for the model
    batch_size = 1

    # If you had previously never stored tiling for this program, store it
    store_tiling = False

    # Whether or not to store the compiler output as json.
    # If you want to specify the filename, set 'json_output_filename' to a string name
    store_json_output = False
    json_output_filename = None
    BENCH_DIR = Path(f"{CWD}/../benchmarks").absolute()

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
                              factor_fn='default',
                              print_config=False
                              )
    import pprint
    pprint.pprint(program.emit("json_no_ops"))
    # validate_program(program, print_difference=True)
