from codelets.adl import ArchitectureGraph, ComputeNode, CommunicationNode, StorageNode, Codelet
from collections import namedtuple
from codelets.codelet import Codelet
import numpy as np
# compute node:       list of capabilities
# edge:               latency and bw are written into the "subgraph_edge"
# storage node:       capacity, r/w bw, access type in "MemConfig" TODO I/O ports
# communication node: fan i/o through graph API, communication type input, bw input...
# capabilities:       ???
# composite capabilities: ???

SysArrayConfig = namedtuple('SysArrayConfig', ['width', 'height', 'ibuf', 'obuf', 'bbuf', 'wbuf'])
SIMDConfig = namedtuple('SIMDConfig', ['lanes', 'mem_cfg'])
MemConfig = namedtuple('MemConfig', ['size', 'read_bw', 'write_bw', 'access_type'])
ExtMem = namedtuple('ExtMem', ['size', 'read_bw', 'write_bw', 'access_type'])

def generate_genesys(genesys_cfg):
    genesys = ComputeNode("Genesys")
    compute = genesys_cfg['compute']
    storage = genesys_cfg['storage']
    sys_array_nodes = ["IBUF", "WBUF", "BBUF", "OBUF", "pe_array"]
    systolic_array = ComputeNode("systolic_array")

    for name, attr in storage.items():
        mem = StorageNode(name, **attr)
        if name in sys_array_nodes:
            systolic_array.add_subgraph_node(mem)
        else:
            genesys.add_subgraph_node(mem)

    for name, attr in compute.items():
        comp = ComputeNode(name, **attr)
        if name in sys_array_nodes:
            systolic_array.add_subgraph_node(comp)
        else:
            genesys.add_subgraph_node(comp)


    systolic_array.add_subgraph_edge('IBUF', 'pe_array')
    systolic_array.add_subgraph_edge('WBUF', 'pe_array')
    systolic_array.add_subgraph_edge('BBUF', 'pe_array')
    systolic_array.add_subgraph_edge('OBUF', 'pe_array')
    systolic_array.add_subgraph_edge('pe_array', 'OBUF')
    genesys.add_subgraph_node(systolic_array)

    genesys.add_subgraph_edge('VMEM', 'SIMD')
    genesys.add_subgraph_edge('SIMD', 'VMEM')
    genesys.add_subgraph_edge('IMM', 'SIMD')
    genesys.add_subgraph_edge('SIMD', 'IMM')

    genesys.add_subgraph_edge('SIMD', 'DRAM')
    genesys.add_subgraph_edge('DRAM', 'SIMD')
    genesys.add_subgraph_edge('OBUF', 'SIMD')
    return genesys


def generate_systolic_array(systolic_array_cfg):
    pass

def generate_simd(simd_cfg):
    pass