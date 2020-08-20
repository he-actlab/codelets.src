from codelets.adl.example.dnnweaver import generate_dnnweaver, SysArrayConfig, SIMDConfig, ExtMem
import pytest

@pytest.mark.parametrize('sys_array_cfg, simd_cfg, extern_mem',[
    (SysArrayConfig(height=32, width=32, ibuf=128, wbuf=128, bbuf=128, obuf=128), SIMDConfig(lanes=32), ExtMem(size=1024)),
])
def test_dnnweaver(sys_array_cfg, simd_cfg, extern_mem):
    adl_graph = generate_dnnweaver(sys_array_cfg, simd_cfg, extern_mem)
    adl_graph.networkx_visualize("dnnweaver")