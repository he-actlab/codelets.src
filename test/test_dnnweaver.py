from codelets.adl.example.dnnweaver import generate_dnnweaver, SysArrayConfig, SIMDConfig, MemConfig
from codelets.adl.serialization import serialize_graph, deserialize_graph, compare_graphs
import pytest
from pathlib import Path
CWD = Path(f"{__file__}").parent

@pytest.mark.parametrize('sys_array_hw, simd_lanes, mem_cfg',[
    ((32,32), 32, MemConfig(size=128, read_bw=32, write_bw=32, access_type="RAM")),
])
def test_dnnweaver(sys_array_hw, simd_lanes, mem_cfg):
    output_name = "dnnweaver.json"
    sys_array_cfg = SysArrayConfig(height=sys_array_hw[0], width=sys_array_hw[1], ibuf=mem_cfg, obuf=mem_cfg, bbuf=mem_cfg, wbuf=mem_cfg)
    simd_cfg = SIMDConfig(lanes=simd_lanes, mem_cfg=MemConfig(size=simd_lanes, read_bw=32, write_bw=32, access_type="FIFO"))
    extern_mem = MemConfig(size=1024, read_bw=32, write_bw=32, access_type="RAM")
    adl_graph = generate_dnnweaver(sys_array_cfg, simd_cfg, extern_mem)
    serialize_graph(adl_graph, output_name, validate_graph=True)
    adl_graph2 = deserialize_graph(f"{CWD}/{output_name}", validate_load=True)
    if not compare_graphs(adl_graph, adl_graph2):
        raise RuntimeError