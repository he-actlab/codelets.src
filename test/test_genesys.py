from codelets.examples.genesys import SysArrayConfig, SIMDConfig, MemConfig, generate_simd, generate_genesys
from codelets.adl.serialization import serialize_graph, deserialize_graph, compare_graphs, generate_hw_cfg
from codelets import util
from collections import namedtuple
import pytest
import json
from pathlib import Path
CWD = Path(f"{__file__}").parent
TestDfgNode = namedtuple('TestDfgNode', ['input_components', 'input_shapes', 'attrs'])
GENESYS_CFG_PATH = f"{CWD}/genesys_cfg.json"


def parse_cfg():

    with open(GENESYS_CFG_PATH) as f:
        genesys = json.load(f)
    return genesys

def test_generate_genesys():
    genesys_cfg = parse_cfg()
    genesys = generate_genesys(genesys_cfg)
    serialize_graph(genesys, "genesys.json", validate_graph=True)
    adl_graph2 = deserialize_graph(f"{CWD}/genesys.json", validate_load=True)
    if not compare_graphs(genesys, adl_graph2):
        raise RuntimeError

    generate_hw_cfg(genesys, "genesys_hw_cfg.json")

def test_generate_simd():
    genesys_cfg = parse_cfg()

    # simd_array = generate_simd(genesys_cfg['compute']['SIMD'])
#