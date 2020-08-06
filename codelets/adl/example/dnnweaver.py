from codelets.adl import ArchitectureGraph, ComputeNode, CommunicationNode, StorageNode

def generate_dnnweaver(array_height, array_width, simd_lanes, wbuf_size, ibuf_size, obuf_size, bbuf_size, oc_bandwidth):
    
    # dnnweaver
    dnnweaver = ArchitectureGraph()

    # compute nodes: array, simd
    # memory nodes: wbuf, ibuf, obuf, bbuf
    # communication nodes: fifo, bus

    return dnnweaver
