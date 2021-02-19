from codelets.examples import generate_dnnweaver
from codelets.examples.dnnweaver import SysArrayConfig, SIMDConfig, MemConfig
from codelets.compiler.serialization import serialize_graph, deserialize_graph, compare_graphs
from collections import namedtuple
import pytest
from pathlib import Path
CWD = Path(f"{__file__}").parent
TestDfgNode = namedtuple('TestDfgNode', ['input_components', 'input_shapes', 'attrs'])

@pytest.mark.parametrize('sys_array_hw, simd_lanes, mem_cfg',[
    ((32, 32), 32, MemConfig(size=128, read_bw=32, write_bw=32, access_type="RAM")),
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


@pytest.mark.parametrize('sys_array_hw, simd_lanes, mem_cfg',[
    ((32, 32), 32, MemConfig(size=128, read_bw=32, write_bw=32, access_type="RAM")),
])
def test_dnnweaver_visualization(sys_array_hw, simd_lanes, mem_cfg):
    sys_array_cfg = SysArrayConfig(height=sys_array_hw[0], width=sys_array_hw[1], ibuf=mem_cfg, obuf=mem_cfg, bbuf=mem_cfg, wbuf=mem_cfg)
    simd_cfg = SIMDConfig(lanes=simd_lanes, mem_cfg=MemConfig(size=simd_lanes, read_bw=32, write_bw=32, access_type="FIFO"))
    extern_mem = MemConfig(size=1024, read_bw=32, write_bw=32, access_type="RAM")
    adl_graph = generate_dnnweaver(sys_array_cfg, simd_cfg, extern_mem)
    output_name = "dnnweaver"
    adl_graph.networkx_visualize(output_name)


@pytest.mark.parametrize('sys_array_hw, simd_lanes, mem_cfg, inp_shape, ceil_mode, kernel_shape, pads, strides', [
    ((32, 32), 32, MemConfig(size=128, read_bw=32, write_bw=32, access_type="RAM"), (1, 64, 112, 112), 0, [3, 3], [0, 0, 1, 1], [2, 2]),
])
def test_dnnweaver_max_pool(sys_array_hw, simd_lanes, mem_cfg, inp_shape, ceil_mode, kernel_shape, pads, strides):
    sys_array_cfg = SysArrayConfig(height=sys_array_hw[0], width=sys_array_hw[1], ibuf=mem_cfg, obuf=mem_cfg, bbuf=mem_cfg, wbuf=mem_cfg)
    simd_cfg = SIMDConfig(lanes=simd_lanes, mem_cfg=MemConfig(size=simd_lanes, read_bw=32, write_bw=32, access_type="FIFO"))
    extern_mem = MemConfig(size=1024, read_bw=32, write_bw=32, access_type="RAM")
    adl_graph = generate_dnnweaver(sys_array_cfg, simd_cfg, extern_mem)
    # max_pool_attrs = {'ceil_mode': ceil_mode, 'kernel_shape': kernel_shape, 'pads': pads, 'strides': strides}
    # max_pool_node = TestDfgNode(input_components=['x'], input_shapes=[inp_shape], attrs=max_pool_attrs)
    # max_pool_codelet = MaxPool(max_pool_node)
    # groupings = max_pool_codelet.create_pool_indices()

def test_genesys_simd():
    pass



