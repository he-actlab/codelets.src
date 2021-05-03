from codelets.examples.genesys import genesys_instructions, define_genesys,\
    GENESYS_CFG, GENESYS_DTYPES, DTYPE_MAP, compile_genesys_layer
import polymath as pm
from pprint import pprint
from codelets import initialize_program, tile, hoist, pad_operands, update_operand_dtypes
from collections import namedtuple
from .util import store_compilation_output
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


def test_genesys_conv_resnet50():
    # layer_name = "resnet50_relu"
    layer_name = "lenet_gemm"
    # layer_name = "resnet50_globalaveragepool"
    # layer_name = "resnet50_add"
    # layer_name = "resnet50_maxpool"
    batch_size = 1
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
                              verbose=False,
                              benchmark_path=BENCH_DIR,
                              factor_fn='default',
                            batch_size=batch_size,
                            do_hoist_stage=True,
                            do_tile_stage=True,
                            print_config=True
                              )
    import networkx as nx
    import matplotlib.pyplot as plt
    dfg = program.create_cdlt_dfg()
    colors = list(nx.get_node_attributes(dfg, 'color').values())
    labels = nx.get_node_attributes(dfg, 'label')

    nx.draw(dfg, pos=nx.spring_layout(dfg), node_color=colors, font_weight='bold', labels=labels)
    plt.savefig(f"{layer_name}.png")
    # res = program.emit("json_no_ops")
    # pprint(res)

    # store_compilation_output(program, "json_no_ops", extension="json")
    # store_compilation_output(program, "string_final", extension="txt")
    # store_compilation_output(program, "decimal", extension="txt")
    # store_compilation_output(program, "binary", extension="txt")



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

