
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

def test_resnet_simulation():
    # Load polymath program
    graph = pm.pb_load(f"{BENCH_DIR}/resnet18v1.srdfg")

    # Create Genesys architecture graph
    genesys_cfg = parse_cfg()
    genesys = generate_genesys(genesys_cfg)

    # Compile the program
    program = compile(graph, genesys, f"{BENCH_DIR}", store_output=True, output_type="json")

    # SIMULATOR TESTS HERE
