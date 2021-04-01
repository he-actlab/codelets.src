from codelets.examples.genesys import genesys_instructions, define_genesys, GENESYS_CFG, GENESYS_DTYPES
import polymath as pm
from pprint import pprint
from codelets import initialize_program, tile, hoist, pad_operands, update_operand_dtypes
from collections import namedtuple
import json
from pathlib import Path

CWD = Path(f"{__file__}").parent
TEST_DIR = f"{CWD}/input_files"
BENCH_DIR = f"{CWD}/../benchmarks"
MODEL_DIR = f"{BENCH_DIR}/models/srdfg"
LAYER_DIR = f"{BENCH_DIR}/layers/srdfg"

TestDfgNode = namedtuple('TestDfgNode', ['input_components', 'input_shapes', 'attrs'])
GENESYS_CFG_PATH = f"{CWD}/scratch/genesys_cfg.json"

def test_genesys_add():
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_add.srdfg")

    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)

    program.add_compilation_step("update_operand_dtypes", update_operand_dtypes, preproc=True, stage_kwargs={'dtype_map': GENESYS_DTYPES})
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': []})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("json_no_ops")
    pprint(res)

def test_genesys_relu():
    from pprint import pprint
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_relu.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': []})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("json_no_ops")
    pprint(res)

def test_genesys_max_pool():
    from pprint import pprint
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_maxpool.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': []})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("json_no_ops")
    pprint(res)

def test_genesys_global_avg_pool():
    from pprint import pprint
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_globalaveragepool.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': []})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("json_no_ops")
    pprint(res)

def test_genesys_gemm():
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_gemm.srdfg")
    batch_size_pass = pm.UpdateBatchSize(32, graph.op_name)
    graph = batch_size_pass(graph)
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': []})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("json_no_ops")
    pprint(res)

def test_genesys_conv():
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_conv.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': []})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("string_final")
    print(res)




