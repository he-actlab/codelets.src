from codelets.compiler.relocation_table import RelocationTable
from examples.genesys import GENESYS_CFG, GENESYS_DTYPES, DTYPE_MAP, \
    compile_genesys_layer, compile_extracted_genesys_layer
from collections import namedtuple
from .util import create_reference_outputs, validate_program
from pathlib import Path
import pytest
import pprint
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

@pytest.mark.parametrize('source_model, layer_name',[
    # ("lenet", "conv"),
    ("resnet18_train", "batchnorm_grad"),
    # ("resnet18_train", "batch_norm"),
    # ("lenetbn_train", "batchnorm_grad"),
    # ("lenetbn_train", "batch_norm"),
    # ("lenetbn", "mean_var"),
    # ("resnet18_train", "cross_entropy_loss"),
    # ("lenet_train", "cross_entropy_loss"),
    # ("lenet_train", "cross_entropy_loss_grad"),
])
def test_extracted_layer(source_model, layer_name):
    train = False

    if "train" in source_model:
        source_model = source_model.split("_")[0]
        train = True

    batch_size = 1
    update_cfg_dtypes = False
    BENCH_DIR = Path(f"{CWD}/../benchmarks").absolute()

    program = compile_extracted_genesys_layer(source_model,
                                              layer_name,
                                              train=train,
                                              update_cfg_dtypes=update_cfg_dtypes,
                                              verbose=False,
                                              benchmark_path=BENCH_DIR,
                                              factor_fn='default',
                                              batch_size=batch_size,
                                              print_config=False)
    # print(program.emit("operations_idx"))

@pytest.mark.parametrize('layer_name',[
    "resnet18_gemm",
    # "custom_matmul_matmul",
    # "resnet18_train_batchnormalization",
    # "resnet18_relu",
    # "resnet18_add",
    # "resnet18_conv",
    # "resnet18_globalaveragepool",
    # "lenet_averagepool",
    # "lenet_gemm",
    # "lenet_bn_conv",
    # "custom_conv_conv",
    # "custom_gemm_gemm",
])
def test_genesys_layers(layer_name):
    batch_size = 1
    tile_method = "min_tiles"
    # tile_method = "valid_split"

    update_cfg_dtypes = False
    tiling_path = None
    store_tiling = False
    store_json_output = False
    json_output_filename = None
    BENCH_DIR = Path(f"{CWD}/../benchmarks").absolute()
    # offsets = [0, 2048, 4096]
    # reloc_offsets = {ns: offsets[i] for i, ns in enumerate(RelocationTable.MEM_LAYOUT)}

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
                            print_config=False,
                            tiling_search_algorithm=tile_method
                                    # relocation_offsets=reloc_offsets
                              )
    # print(program.codelets[0].loop_param_map)
    # print(program.codelets[0].param_tiling)
    # print(program.emit("operations_idx"))
    wbuf = program.hag.get_subgraph_node("WBUF")
    print(wbuf.capacity)
    # print(program.emit("string_final"))
    # print(program.emit("string_final"))


@pytest.mark.parametrize('layer_name',[
    # "resnet18_gemm",
    # "resnet50_conv_small",
    # "cc1_conv",
    # "resnet18_train_batchnormalization",
    # "resnet18_relu",
    # "resnet18_add",
    #  "resnet18_conv",
    # "resnet18_globalaveragepool",
    # "lenet_averagepool",
    # "lenet_gemm",
    # "lenet_bn_conv",
    "custom_conv_conv",
    # "custom_gemm_gemm",
])
def test_genesys_layers_min_tiles_search(layer_name):
    batch_size = 1
    update_cfg_dtypes = False
    tiling_path = None
    store_tiling = False
    store_json_output = False
    json_output_filename = None
    BENCH_DIR = Path(f"{CWD}/../benchmarks").absolute()
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
                            print_config=False,
                            tiling_search_algorithm='min_tiles'
                              )
    # pprint.pprint(program.emit("json_no_ops"))
    normal = program.emit("operations")
    print(normal)




def test_reference_creation():
    batch_size = 1
    update_cfg_dtypes = False
    # names = ["resnet18", "lenet", "lenet_train"]
    # names = []
    names = ["resnet18_relu", "resnet18_add", "resnet18_conv", "resnet18_gemm", "resnet18_globalaveragepool", "lenet_averagepool", "lenet_conv", "lenet_gemm",
                 "resnet18_train_batchnormalization"]
    create_reference_outputs(names, batch_size=batch_size, update_cfg_dtypes=update_cfg_dtypes,
                             verbose=False)
#