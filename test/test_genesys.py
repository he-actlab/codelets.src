from codelets.adl.serialization import serialize_graph, deserialize_graph, generate_hw_cfg
from codelets.examples.genesys import generate_simd_capabilities, generate_systolic_array_capabilities, generate_genesys
import polymath as pm
from pathlib import Path
from codelets import compile, deserialize_graph
from collections import namedtuple
import pytest
import json
from pathlib import Path
from jsondiff import diff

CWD = Path(f"{__file__}").parent
BENCH_DIR = f"{CWD}/input_files"

TestDfgNode = namedtuple('TestDfgNode', ['input_components', 'input_shapes', 'attrs'])
GENESYS_CFG_PATH = f"{CWD}/scratch/genesys_cfg.json"


def parse_cfg():

    with open(GENESYS_CFG_PATH) as f:
        genesys = json.load(f)
    return genesys

def test_generate_genesys():
    genesys_cfg = parse_cfg()
    genesys = generate_genesys(genesys_cfg)

def test_genesys_resnet18():

    # graph = pm.pb_load(f"{BENCH_DIR}/resnet18v1.mgdfg")
    graph = pm.pb_load(f"{BENCH_DIR}/resnet18v1.srdfg")
    genesys_cfg = parse_cfg()
    genesys = generate_genesys(genesys_cfg)
    compile(graph, genesys, f"{BENCH_DIR}", store_output=True, output_type="json")

def test_genesys_serialization():
    genesys_cfg = parse_cfg()
    genesys = generate_genesys(genesys_cfg)
    json_genesys = serialize_graph(genesys, f"{CWD}/genesys.json")
    deser_genesys = deserialize_graph(f"{CWD}/genesys.json")
    json_genesys_deser = serialize_graph(deser_genesys, f"{CWD}/deser_genesys.json")
    assert json_genesys_deser == json_genesys

# def test_serialize_relu():
#     genesys_cfg = parse_cfg()
#
#     genesys = generate_genesys(genesys_cfg)
#     adl_graph2 = deserialize_graph(f"{CWD}/genesys.json", validate_load=True)

