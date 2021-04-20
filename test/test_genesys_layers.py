from codelets.examples.genesys import genesys_instructions, define_genesys,\
    GENESYS_CFG, GENESYS_DTYPES, DTYPE_MAP, compile_genesys_layer
import polymath as pm
from pprint import pprint
from codelets import initialize_program, tile, hoist, pad_operands, update_operand_dtypes
from collections import namedtuple
import json
import pytest
from pathlib import Path

CWD = Path(f"{__file__}").parent
TEST_DIR = f"{CWD}/input_files"
BENCH_DIR = f"{CWD}/../benchmarks"
MODEL_DIR = f"{BENCH_DIR}/models/srdfg"
LAYER_DIR = f"{BENCH_DIR}/layers/srdfg"
TILING_DIR = f"{BENCH_DIR}/tiling_info"

TestDfgNode = namedtuple('TestDfgNode', ['input_components', 'input_shapes', 'attrs'])
GENESYS_CFG_PATH = f"{CWD}/scratch/genesys_cfg.json"

def update_genesys_cfg_from_dtypes():
    GENESYS_CFG['DATA_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight']].bits()
    GENESYS_CFG['WGT_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight']].bits()
    GENESYS_CFG['BIAS_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out']].bits()
    GENESYS_CFG['ACC_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out']].bits()

def test_genesys_add():
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_add.srdfg")
    GENESYS_DTYPES['SIMD'] = 'FXP16'
    GENESYS_DTYPES['SYSTOLIC_ARRAY'] = {}
    GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight'] = 'FXP4'
    GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out'] = 'FXP16'
    update_genesys_cfg_from_dtypes()
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("update_operand_dtypes", update_operand_dtypes, preproc=True, stage_kwargs={'dtype_map': GENESYS_DTYPES})
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': {}})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile(tiling_path=f"{TILING_DIR}/resnet18_add_tiling_info1.json")
    # program.store_tiling(f"{TILING_DIR}")
    res = program.emit("json_no_ops")
    pprint(res)

def test_genesys_relu():
    from pprint import pprint
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_relu.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': {}})
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
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': {}})
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
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': {}})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("json_no_ops")
    pprint(res)

def test_genesys_batch_norm():
    from pprint import pprint
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_train_batchnormalization.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': {}})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("json_no_ops")
    pprint(res)

def test_genesys_flatten():
    from pprint import pprint
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_flatten.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': {}})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("json_no_ops")
    pprint(res)



def test_genesys_gemm():
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_gemm.srdfg")
    GENESYS_DTYPES['SIMD'] = 'FXP16'
    GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight'] = 'FXP4'
    GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out'] = 'FXP16'
    update_genesys_cfg_from_dtypes()
    batch_size_pass = pm.UpdateBatchSize(64, graph.op_name)
    graph = batch_size_pass(graph)
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("update_operand_dtypes", update_operand_dtypes, preproc=True, stage_kwargs={'dtype_map': GENESYS_DTYPES})
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': {}})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("json_no_ops")
    pprint(res)

def test_genesys_conv():
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_conv.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': {}})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("string_final")
    print(res)

def test_genesys_conv_bias():
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_conv_bias.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': {}})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("string_final")
    print(res)

def test_genesys_conv_resnet50():
    layer_name = "resnet50_conv"
    batch_size = 16
    update_cfg_dtypes = False
    tiling_path = None
    store_tiling = False
    store_json_output = False
    json_output_filename = None
    BENCH_DIR = Path(f"{CWD}/../benchmarks").absolute()

    # This function returns
    program = compile_genesys_layer(layer_name,
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
    res = program.emit("json_no_ops")
    pprint(res)


@pytest.mark.parametrize('filtered_layers',[
    # (['sgd1d', 'sgd2d', 'sgd3d', 'sgd4d']),
    # (['cross_entropy_loss']),
    # (['max_pool_grad']),
    # (['global_average_pool_grad']),
    # (['relu_grad']),
    # (['elem_add_grad']),
    # (['cross_entropy_loss_grad']),
    (['reduce_sum']),
])
def test_filtered_layers(filtered_layers):
    graph = pm.pb_load(f"{MODEL_DIR}/resnet18_train.srdfg")

    train_graph = pm.create_training_graph(graph)
    multi_dim_pass = pm.RenameMultiDimOps()
    train_graph = multi_dim_pass(train_graph)
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(train_graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': {}})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile(verbose=False, sequence_algorithm='filtered', filtered_layers=filtered_layers)
    res = program.emit("json_no_ops")
    pprint(res)

