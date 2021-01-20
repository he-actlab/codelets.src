from codelets.adl.graph import ArchitectureGraph, ComputeNode, CommunicationNode, StorageNode
from codelets.adl import Codelet
from collections import namedtuple
import numpy as np
# compute node:       list of primitives
# edge:               latency and bw are written into the "subgraph_edge"
# storage node:       capacity, r/w bw, access type in "MemConfig" TODO I/O ports
# communication node: fan i/o through graph API, communication type input, bw input...
# primitives:       ???
# composite primitives: ???

SysArrayConfig = namedtuple('SysArrayConfig', ['bitwidth', 'height', 'ibuf', 'obuf', 'bbuf', 'wbuf'])
SIMDConfig = namedtuple('SIMDConfig', ['lanes', 'mem_cfg'])
MemConfig = namedtuple('MemConfig', ['size', 'read_bw', 'write_bw', 'access_type'])
ExtMem = namedtuple('ExtMem', ['size', 'read_bw', 'write_bw', 'access_type'])

def generate_pe_array(array_cfg):
    pass

def generate_dnnweaver(sys_array_cfg, simd_cfg, extern_mem):
    
    # dnnweaver
    dnnw_graph = ComputeNode("DnnWeaver")
    sys_array = generate_systolic_array(sys_array_cfg)
    # TODO: Ask soroush the latency and bandwidth of the bus to external memory
    mem_bus = CommunicationNode("Bus", "bus", 1, 128)
    ext_mem = StorageNode("EXTMEM", extern_mem.read_bw, extern_mem.write_bw, extern_mem.access_type, extern_mem.size)
    dnnw_graph.add_subgraph_node(sys_array)
    dnnw_graph.add_subgraph_node(mem_bus)
    dnnw_graph.add_subgraph_node(ext_mem)
    simd_array = generate_simd_array(dnnw_graph, sys_array, simd_cfg)
    dnnw_graph.add_subgraph_node(simd_array)
    dnnw_graph.add_subgraph_edge(sys_array.get_subgraph_node("OBUF"), simd_array.get_subgraph_node("ALUArray"), {'latency':1, 'bw':32})

    dnnw_graph.add_subgraph_edge(mem_bus, sys_array.get_subgraph_node("OBUF"), {'latency': 1,'bw':32})
    dnnw_graph.add_subgraph_edge(sys_array.get_subgraph_node("OBUF"), mem_bus, {'latency': 1, 'bw':32})
    dnnw_graph.add_subgraph_edge(mem_bus, sys_array.get_subgraph_node("WBUF"), {'latency': 1, 'bw':32})
    dnnw_graph.add_subgraph_edge(mem_bus, sys_array.get_subgraph_node("IBUF"), {'latency': 1, 'bw':32})
    dnnw_graph.add_subgraph_edge(mem_bus, sys_array.get_subgraph_node("BBUF"), {'latency': 1, 'bw':32})
    dnnw_graph.add_subgraph_edge(ext_mem, mem_bus, {'latency': 1, 'bw':32})
    dnnw_graph.add_subgraph_edge(mem_bus, ext_mem, {'latency': 1, 'bw':32})

    dnnw_graph.add_subgraph_edge(mem_bus, simd_array.get_subgraph_node("VMEM"))
    dnnw_graph.add_subgraph_edge(simd_array.get_subgraph_node("VMEM"), mem_bus)

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


# def generate_pe_array(sys_array_cfg):
#     pe_array = ComputeNode(field_name="PEArray")
#     pe_array.set_attr("bitwidth", sys_array_cfg.bitwidth)
#     pe_array.set_attr("height", sys_array_cfg.height)
#
#     # add primitives
#     matmul = Codelet('MatrixMatrixMul')
#     matmul.add_input(field_name='ifmap', src=["IBUF"], dims=['b', 'x'])
#     matmul.add_input(field_name='weight', src=["WBUF"], dims=['x', 'y'])
#     # matmul.add_dataflow(field_name='os',
#     #                     delay=lambda b,x,y: max(pe_array.get_attr('array_height'),
#     #                                             pe_array.get_attr('array_width'))+x,
#     #                     constraint={'b': (0, pe_array.get_attr('array_height')),
#     #                                 'x': (0, 'inf'),
#     #                                 'y': (0, pe_array.get_attr('array_width'))}
#     #                    )
#     # matmul.add_dataflow(field_name='ws',
#     #                     delay=lambda b,x,y: pe_array.get_attr('array_height')+b,
#     #                     constraint={'b': (0, 'inf'),
#     #                                 'x': (0, pe_array.get_attr('array_height')),
#     #                                 'y': (0, pe_array.get_attr('array_width'))}
#     #                    )
#     # matmul.add_dataflow(field_name='is',
#     #                     delay=lambda b,x,y: pe_array.get_attr('array_height')+b,
#     #                     constraint={'b': (0, pe_array.get_attr('array_width')),
#     #                                 'x': (0, pe_array.get_attr('array_height')),
#     #                                 'y': (0, 'inf')}
#     #                    )
#     pe_array.add_primitive(matmul)
#
#     # TODO MatrixVectorMul & VectorVectorMul
#
#     return pe_array

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

    sys_array.add_subgraph_edge(ibuf_node, pe_array, {'latency':1, 'bw':32})
    sys_array.add_subgraph_edge(wbuf_node, pe_array, {'latency':1, 'bw':32})
    sys_array.add_subgraph_edge(bbuf_node, pe_array, {'latency':1, 'bw':32})
    sys_array.add_subgraph_edge(obuf_node, pe_array, {'latency':1, 'bw':32})
    sys_array.add_subgraph_edge(pe_array, obuf_node, {'latency':1, 'bw':32})

    # TODO make sure all primitives of the enclosed nodes are detected through some API

    return sys_array

def generate_simd_array(dnnweaver, systolic_array, simd_cfg: SIMDConfig):
    # Create node and add systolic array dimensions
    simd_array = ComputeNode(name="SIMDArray")
    # TODO SIMD codelets
    
    alu_array = ComputeNode(name="ALUArray")
    # TODO ALU codelets
    
    alu_array.set_attr("bitwidth", simd_cfg.lanes)
    rf_cfg = simd_cfg.mem_cfg
    vec_rf = StorageNode("VMEM", rf_cfg.read_bw, rf_cfg.write_bw, rf_cfg.access_type, rf_cfg.size)
    vec_rf.set_attr("size", simd_cfg.lanes)

    imm_ns = StorageNode("IMM", rf_cfg.read_bw, rf_cfg.write_bw, rf_cfg.access_type, 1)
    imm_ns.set_attr("size", simd_cfg.lanes)

    simd_array.add_subgraph_node(alu_array)
    simd_array.add_subgraph_node(vec_rf)

    simd_array.add_subgraph_edge(vec_rf, alu_array, {'latency':1, 'bw':32})
    simd_array.add_subgraph_edge(alu_array, systolic_array.get_subgraph_node("IBUF"), {'latency':1, 'bw':32})
    simd_array.add_subgraph_edge(alu_array, imm_ns, {'latency':1, 'bw':32})
    simd_array.add_subgraph_edge(alu_array, dnnweaver.get_subgraph_node("EXTMEM"), {'latency': 1, 'bw':32})
    simd_array.add_subgraph_edge(dnnweaver.get_subgraph_node("EXTMEM"), alu_array, {'latency': 1, 'bw':32})
    simd_array.add_subgraph_edge(imm_ns, alu_array, {'latency': 1, 'bw':32})
    simd_array.add_subgraph_edge(systolic_array.get_subgraph_node("OBUF"), alu_array, {'latency': 1, 'bw':32})
    simd_array.add_subgraph_edge(alu_array, vec_rf, {'latency': 1, 'bw':32})

    add_simd_capabilities(alu_array)


    return simd_array

def add_simd_capabilities(alu_array):
    # Addition
    ALU_OPS = ['ADD', 'SUB', 'MUL', 'MACC', 'DIV', 'MAX', 'MIN', 'MIN', 'RSHIFT', 'LSHIFT', 'MOVE', 'COND_MOVE']
    CALC_OPS = ['RELU', 'LEAKY_RELU', 'SIGMOID', 'TANH', 'EXP', 'LN', 'SQRT', 'INV_SQRT', 'LOG2']
    COMPARISON_OPS = ['EQUAL', 'NEQ', 'GT', 'GTE', 'LT', 'LTE']
    CAST_OPS = ['FXP32_FXP16', 'FXP32_FXP8', 'FXP16_FXP32', 'FXP8_FXP32', 'FP32_FP16', 'FP16_FP32']
    DTYPE_CFG_OPS = ['FXP32', 'FXP16', 'FXP8', 'FP32', 'FP16']
    LOCK_NS_OPS = ['LOCK', 'UNLOCK']
    ITER_OPS = ['BASE_SIGNEXT', 'BASE_LOW', 'BASE_HIGH', 'BASE_ZERO_FILL', 'STRIDE_SIGNEXT', 'STRIDE_LOW', 'STRIDE_HIGH', 'STRIDE_ZEROFILL']
    LOOP_OPS = ['SHORT_LOOP', 'LONG_LOOP_INST', 'LONG_LOOP_ITER']
    add_op = Add("add")

    for op_name in ALU_OPS:
        alu_op = Codelet(op_name)
        alu_op.add_input(name='op1', src=["OBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])
        alu_op.add_input(name='op2', src=["OBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])
        alu_op.add_output(name='op_dst', dst=["IBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])
        alu_array.add_primitive(alu_op)

    for op_name in CALC_OPS:
        calc_op = Codelet(op_name)
        calc_op.add_input(name='op1', src=["OBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])
        calc_op.add_output(name='op_dst', dst=["IBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])
        alu_array.add_primitive(calc_op)

    for op_name in COMPARISON_OPS:
        comp_op = Codelet(op_name)
        comp_op.add_input(name='op1', src=["OBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])
        comp_op.add_input(name='op2', src=["OBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])
        comp_op.add_output(name='op_dst', dst=["IBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])
        alu_array.add_primitive(comp_op)

    for op_name in CAST_OPS:
        cast_op = Codelet(op_name)
        cast_op.add_input(name='op1', src=["OBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])
        cast_op.add_input(name='op2', src=["IMM"], dims=['w'])
        cast_op.add_output(name='op_dst', dst=["IBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])
        alu_array.add_primitive(cast_op)

    for op_name in DTYPE_CFG_OPS:
        dtype_cfg_op = Codelet(op_name)
        dtype_cfg_op.add_input(name='op1', src=["IMM"], dims=['w'])
        alu_array.add_primitive(dtype_cfg_op)

    for op_name in LOCK_NS_OPS:
        lock_op = Codelet(op_name)
        lock_op.add_input(name='op1', src=["OBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])
        lock_op.add_input(name='op2', src=["OBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])
        lock_op.add_output(name='op_dst', dst=["IBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])
        alu_array.add_primitive(lock_op)

    for op_name in ITER_OPS:
        iter_op = Codelet(op_name)
        iter_op.add_input(name='op1', src=["CONST"], dims=['w'])
        iter_op.add_output(name='op_dst', dst=["IBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])
        alu_array.add_primitive(iter_op)

    for op_name in LOOP_OPS:
        loop_op = Codelet(op_name)
        loop_op.add_input(name='op1', src=["CONST"], dims=['w'])
        loop_op.add_output(name='op_dst', dst=["IBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])
        alu_array.add_primitive(loop_op)

    max_pool = Codelet("max_pool")
    max_pool.add_input(name='op1', src=["OBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])
    max_pool.add_input(name='op2', src=["OBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])
    max_pool.add_output(name='op_dst', dst=["IBUF", "VMEM", "IMM", "EXTMEM"], dims=['w'])


class Add(Codelet):
    def __init__(self, name):
        super(Add, self).__init__(name)

class MaxPool(Codelet):
    def __init__(self, dfg_node):
        super(MaxPool, self).__init__('max_pool', dfg_node)
        # TODO: Use actual data structure for dfg node attributes
        self._ceil_mode = dfg_node.attrs['ceil_mode']
        self._kernel_shape = dfg_node.attrs['kernel_shape']
        self._pads = dfg_node.attrs['pads']
        self._strides = dfg_node.attrs['strides']
        self._input = dfg_node.input_components[0]
        self._input_shape = dfg_node.input_shapes[0]
        self._shape_fn = np.floor if self.ceil_mode == 0 else np.ceil
        self._out_shape = self.compute_output_shape()
        self._padded_shape = self.compute_padded_shape()

    @property
    def ceil_mode(self):
        return self._ceil_mode

    def get_out_shape(self):
        return self._out_shape

    def get_padded_shape(self):
        return self._padded_shape

    @property
    def kernel_shape(self):
        return self._kernel_shape

    @property
    def pads(self):
        return self._pads

    @property
    def strides(self):
        return self._strides

    @property
    def input_name(self):
        return self._input

    @property
    def shape_fn(self):
        return self._shape_fn

    @property
    def input_shape(self):
        return self._input_shape

    def compute_padded_shape(self):
        pad_shape = [self.pads[0] + self.pads[-2], self.pads[1] + self.pads[-1]]
        return tuple(list(self.input_shape[:2]) + list(np.add(self.input_shape[2:], pad_shape)))

    def compute_output_shape(self):
        output_shape = list(self.input_shape[:2])
        pad_shape = [self.pads[0] + self.pads[-2], self.pads[1] + self.pads[-1]]
        for i in range(len(self.input_shape[2:])):
            dim_size = (self.input_shape[2+i] + pad_shape[i] - ((self.kernel_shape[i] - 1) + 1)) / self.strides[i] + 1
            output_shape.append(self.shape_fn(dim_size).astype('int'))
        return tuple(output_shape)

    def create_pool_indices(self):
        groupings = []

        for b in range(self.input_shape[0]):
            for c in range(self.input_shape[1]):
                for y in range(self.get_out_shape()[-2]):
                    for x in range(self.get_out_shape()[-1]):
                        max_indices = []
                        for m in range(self.kernel_shape[0]):
                            for n in range(self.kernel_shape[1]):
                                idx = [b, c, m + self.strides[0]*y, n + self.strides[0]*x]
                                max_indices.append(idx)
                        groupings.append(max_indices)
        groupings = np.asarray(groupings)
        idx_map = {}
        for i in range(groupings.shape[0]):
            raveled = np.ravel_multi_index(groupings[i].T, self.get_padded_shape(), order='F')
            idx_map[f'grouping{i}'] = {}
            for j in range(groupings.shape[1]):
                idx_map[f'grouping{i}'][tuple(groupings[i][j])] = raveled[j]

            print(sorted(idx_map[f'grouping{i}'].values()))
        return idx_map

    def instantiate(self, hag):
        pass

