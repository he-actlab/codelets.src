from codelets.examples import GENESYS_CFG, define_genesys, compile_genesys_layer
from codelets.adl.serialization import deserialize_hag
import json
from pathlib import Path
import pytest

CWD = Path(f"{__file__}").parent


def test_genesys_serialization():
    def_cfg = GENESYS_CFG
    genesys = define_genesys(def_cfg)
    with open('genesys-hag.json', 'w') as f:
        json.dump(genesys.to_json(), f, indent=4)


# TODO: add a way to compare the two objects
def test_genesys_serialization_deserialization():
    def_cfg = GENESYS_CFG
    genesys = define_genesys(def_cfg)
    with open('genesys-hag.json', 'w') as f:
        json.dump(genesys.to_json(), f, indent=4)
    deserialized_genesys = deserialize_hag('genesys-hag.json')


@pytest.mark.parametrize('layer_name', [
    "resnet18_relu",
    "resnet18_add",
    "resnet18_conv",
    "resnet18_gemm",
    "resnet18_globalaveragepool",
    "lenet_averagepool",
    "lenet_gemm",
    "lenet_conv",
])
# TODO: Compare both the generated programs
def test_genesys_deserialize_and_compile_layer(layer_name):
    batch_size = 1
    update_cfg_dtypes = False
    tiling_path = None
    store_tiling = False
    store_json_output = False
    json_output_filename = None
    BENCH_DIR = Path(f"{CWD}/../benchmarks").absolute()

    # This function saves the json file
    program_original = compile_genesys_layer(layer_name,
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
                                             save_genesys_filename='hag.json'
                                             )

    program_from_json = compile_genesys_layer(layer_name,
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
                                              load_genesys_filename='hag.json'
                                              )
