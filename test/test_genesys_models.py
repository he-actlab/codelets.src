from codelets.examples.genesys import genesys_instructions, define_genesys, GENESYS_CFG
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
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': []})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile(tiling_path=f"{TILING_DIR}/resnet18_tiling_info.json")

    # program.compile()
    # program.store_tiling(f"{TILING_DIR}")

    res = program.emit("json_no_ops")
    pprint(res)


def test_genesys_resnet18_train():
    train_graph = pm.pb_load(f"{MODEL_DIR}/resnet18_train.srdfg")
    # train_graph = pm.create_training_graph(graph)
    genesys = define_genesys(GENESYS_CFG)
    #
    program = initialize_program(train_graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': []})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    # program.compile(tiling_path=f"{TILING_DIR}/resnet18_tiling_info.json")
    #
    program.compile()
    # program.store_tiling(f"{TILING_DIR}")
    #
    res = program.emit("json_no_ops")
    pprint(res)