from codelets.compiler.serialization import serialize_graph
from codelets.examples.genesys import generate_genesys,\
    genesys_instructions, define_genesys
import polymath as pm
from codelets import deserialize_graph, initialize_program, tile, hoist
from collections import namedtuple
import json
from pathlib import Path

CWD = Path(f"{__file__}").parent
BENCH_DIR = f"{CWD}/input_files"

TestDfgNode = namedtuple('TestDfgNode', ['input_components', 'input_shapes', 'attrs'])
GENESYS_CFG_PATH = f"{CWD}/scratch/genesys_cfg.json"

if __name__ == "__main__":
    genesys = define_genesys("transformation")
    pe_array = genesys.get_subgraph_node("pe_array")
    print(pe_array.dimensions)