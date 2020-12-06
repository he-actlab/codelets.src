import polymath as pm
from pathlib import Path
from codelets import compile, deserialize_graph


CWD = Path(f"{__file__}").parent

BENCH_DIR = f"{CWD}/input_files"

def test_resnet18():
    graph = pm.pb_load(f"{BENCH_DIR}/resnet18v1.mgdfg")
    hag = deserialize_graph(f"{CWD}/genesys.json", validate_load=True)
    compile(graph, hag, f"{BENCH_DIR}")


def test_lenet():
    graph = pm.pb_load(f"{BENCH_DIR}/lenet.mgdfg")
    hag = deserialize_graph(f"{CWD}/genesys.json", validate_load=True)
    compile(graph, hag, f"{BENCH_DIR}")
