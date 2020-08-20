from codelets.adl import ArchitectureGraph, ComputeNode, CommunicationNode, StorageNode
from collections import namedtuple

SysArrayConfig = namedtuple('SysArrayConfig', ['width', 'height', 'ibuf', 'obuf', 'bbuf', 'wbuf'])
SIMDConfig = namedtuple('SIMDConfig', ['lanes', 'mem_cfg'])
MemConfig = namedtuple('MemConfig', ['size', 'read_bw', 'write_bw', 'access_type'])
ExtMem = namedtuple('ExtMem', ['size', 'read_bw', 'write_bw', 'access_type'])

def generate_dnnweaver(sys_array_cfg, simd_cfg, extern_mem):
    
    # dnnweaver
    dnnw_graph = ComputeNode("DnnWeaver")
    sys_array = generate_systolic_array(sys_array_cfg)
    dnnw_graph.add_subgraph_node(sys_array)
    simd_array = generate_simd_array(simd_cfg)
    dnnw_graph.add_subgraph_node(simd_array)
    dnnw_graph.add_subgraph_edge(sys_array.get_subgraph_node("OBUF"), simd_array.get_subgraph_node("ALUArray"))


    mem_bus = CommunicationNode("Bus", "bus", 128)
    ext_mem = StorageNode("ExternalMem", extern_mem.read_bw, extern_mem.write_bw, extern_mem.access_type, extern_mem.size)
    dnnw_graph.add_subgraph_node(mem_bus)
    dnnw_graph.add_subgraph_node(ext_mem)
    dnnw_graph.add_subgraph_edge(mem_bus, sys_array.get_subgraph_node("OBUF"))
    dnnw_graph.add_subgraph_edge(sys_array.get_subgraph_node("OBUF"), mem_bus)
    dnnw_graph.add_subgraph_edge(mem_bus, sys_array.get_subgraph_node("IBUF"))
    dnnw_graph.add_subgraph_edge(mem_bus, sys_array.get_subgraph_node("WBUF"))
    dnnw_graph.add_subgraph_edge(mem_bus, sys_array.get_subgraph_node("BBUF"))
    dnnw_graph.add_subgraph_edge(ext_mem, mem_bus)
    dnnw_graph.add_subgraph_edge(mem_bus, ext_mem)

    dnnw_graph.add_subgraph_edge(mem_bus, simd_array.get_subgraph_node("VectorRF"))
    dnnw_graph.add_subgraph_edge(simd_array.get_subgraph_node("VectorRF"), mem_bus)


    return dnnw_graph

def generate_buffers(sys_array_cfg: SysArrayConfig):
    # Set buffer config options
    ibuf_cfg = sys_array_cfg.ibuf
    ibuf_node = StorageNode("IBUF", ibuf_cfg.read_bw, ibuf_cfg.write_bw, ibuf_cfg.access_type, ibuf_cfg.size)

    bbuf_cfg = sys_array_cfg.bbuf

    bbuf_node = StorageNode("BBUF", bbuf_cfg.read_bw, bbuf_cfg.write_bw, bbuf_cfg.access_type, bbuf_cfg.size)

    wbuf_cfg = sys_array_cfg.wbuf
    wbuf_node = StorageNode("WBUF", wbuf_cfg.read_bw, wbuf_cfg.write_bw, wbuf_cfg.access_type, wbuf_cfg.size)

    obuf_cfg = sys_array_cfg.bbuf
    obuf_node = StorageNode("OBUF", obuf_cfg.read_bw, obuf_cfg.write_bw, obuf_cfg.access_type, obuf_cfg.size)

    return ibuf_node, bbuf_node, wbuf_node, obuf_node


def generate_pe_array(sys_array_cfg):
    pe_array = ComputeNode(name="PEArray")
    pe_array.set_attr("width", sys_array_cfg.width)
    pe_array.set_attr("height", sys_array_cfg.height)
    return pe_array

def generate_systolic_array(sys_array_cfg: SysArrayConfig):

    # Create node and add systolic array dimensions
    sys_array = ComputeNode(name="SystolicArray")
    ibuf_node, bbuf_node, wbuf_node, obuf_node = generate_buffers(sys_array_cfg)
    sys_array.add_subgraph_node(ibuf_node)
    sys_array.add_subgraph_node(bbuf_node)
    sys_array.add_subgraph_node(wbuf_node)
    sys_array.add_subgraph_node(obuf_node)

    pe_array = generate_pe_array(sys_array_cfg)
    sys_array.add_subgraph_node(pe_array)

    sys_array.add_subgraph_edge(ibuf_node, pe_array)
    sys_array.add_subgraph_edge(wbuf_node, pe_array)
    sys_array.add_subgraph_edge(bbuf_node, pe_array)
    sys_array.add_subgraph_edge(obuf_node, pe_array)
    sys_array.add_subgraph_edge(pe_array, obuf_node)

    return sys_array

def generate_simd_array(simd_cfg: SIMDConfig):
    # Create node and add systolic array dimensions
    simd_array = ComputeNode(name="SIMDArray")
    alu_array = ComputeNode(name="ALUArray")
    alu_array.set_attr("width", simd_cfg.lanes)
    rf_cfg = simd_cfg.mem_cfg
    vec_rf = StorageNode("VectorRF", rf_cfg.read_bw, rf_cfg.write_bw, rf_cfg.access_type, rf_cfg.size)
    vec_rf.set_attr("size", simd_cfg.lanes)

    simd_array.add_subgraph_node(alu_array)
    simd_array.add_subgraph_node(vec_rf)
    simd_array.add_subgraph_edge(vec_rf, alu_array)
    simd_array.add_subgraph_edge(alu_array, vec_rf)

    return simd_array