from codelets.examples.genesys import define_genesys, GENESYS_CFG, compile_genesys, GENESYS_DTYPES
import polymath as pm
from codelets import initialize_program, tile, hoist, pad_operands
from collections import namedtuple
import json
from pprint import pprint
from pathlib import Path

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

def test_genesys_resnet18():
    graph = pm.pb_load(f"{MODEL_DIR}/resnet18.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': {}})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile(tiling_path=f"{TILING_DIR}/resnet18_tiling_info.json")

    # program.compile()
    # program.store_tiling(f"{TILING_DIR}")

    res = program.emit("json_no_ops")
    pprint(res)

def test_genesys_resnet50():
    graph = pm.pb_load(f"{MODEL_DIR}/resnet50.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': {}})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    # program.compile(tiling_path=f"{TILING_DIR}/resnet18_tiling_info.json")

    program.compile()
    program.store_tiling(f"{TILING_DIR}")

    res = program.emit("json_no_ops")
    pprint(res)

def test_genesys_resnet18_train():
    graph = pm.pb_load(f"{MODEL_DIR}/resnet18_train.srdfg")

    train_graph = pm.create_training_graph(graph)
    layout_pass = pm.UpdateLayout('nchw', 'nhwc')
    multi_dim_pass = pm.RenameMultiDimOps()

    train_graph = multi_dim_pass(train_graph)
    train_graph = layout_pass(train_graph)

    genesys = define_genesys(GENESYS_CFG)
    #
    program = initialize_program(train_graph, genesys, mode="training")
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': {}})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    # program.compile(tiling_path=f"{TILING_DIR}/resnet18_tiling_info.json")
    #
    program.compile(verbose=True)
    program.store_tiling(f"{TILING_DIR}")
    #
    res = program.emit("json_no_ops")
    # pprint(res)

def test_genesys_model():
    model_name = 'resnet18'

    # Determines whether to compile a training model or not
    train = True

    # GENESYS_DTYPES['SIMD'] = 'FXP16'
    # GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight'] = 'FXP4'
    # GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out'] = 'FXP16'
    # If you update the genesys datatypes above, set 'update_cfg_dtypes' to 'True'
    update_cfg_dtypes = False

    # If there is an existing tiling file for this particular model, set the tiling path here
    # If this is set to None, then it will re-tile.
    # NOTE: if you are compiling a training program, the filename should be f"{model_name}_train_tiling_info.json"

    tiling_path = f"{model_name}_train_training_tiling_info_checkpoint0.json"
    # tiling_path = None

    # If this is changed, the batch size will updated for the model
    batch_size = 1

    # If you had previously never stored tiling for this program, store it
    store_tiling = True

    # Whether or not to store the compiler output as json.
    # If you want to specify the filename, set 'json_output_filename' to a string name
    store_json_output = True
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
                              benchmark_path=BENCH_DIR
                              )
