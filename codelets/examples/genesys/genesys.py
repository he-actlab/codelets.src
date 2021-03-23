from codelets.adl.graph import ComputeNode, StorageNode
from codelets.adl.flex_template import Instruction
from .genesys_instructions import GENESYS_INSTRUCTIONS
from .genesys_templates import GENESYS_TEMPLATES
from .genesys_codelets import GENESYS_CODELETS
import numpy as np
from . import SIMD_NS, SIMD_OPCODE_BITWIDTH, OP_DTYPES, \
    OP_LOCATIONS, NS_BITWIDTH, NS_IDX_BITWIDTH

def generate_genesys(genesys_cfg):
    genesys = ComputeNode("Genesys1")
    compute = genesys_cfg['compute']
    storage = genesys_cfg['storage']
    sys_array_nodes = ["IBUF", "WBUF", "BBUF", "OBUF", "pe_array"]
    # simd_caps = generate_simd_capabilities()
    # sa_caps = generate_systolic_array_capabilities()
    systolic_array = ComputeNode("systolic_array")

    for name, attr in storage.items():
        if "primitives" in attr:
            attr.pop("primitives")
        mem = StorageNode(name, **attr)

        if name == "DRAM":
            mem.on_chip = False

        if name in sys_array_nodes:
            systolic_array.add_subgraph_node(mem)
        else:
            genesys.add_subgraph_node(mem)

    for name, attr in compute.items():
        if "primitives" in attr:
            attr.pop("primitives")
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

    for p in GENESYS_INSTRUCTIONS['systolic_array']:
        systolic_array.add_primitive(p)


    simd = genesys.get_subgraph_node("SIMD")
    #
    # for c in simd_caps:
    #     simd.add_primitive(c)

    genesys.add_subgraph_edge('VMEM', 'SIMD')
    genesys.add_subgraph_edge('SIMD', 'VMEM')
    genesys.add_subgraph_edge('IMM', 'SIMD')
    genesys.add_subgraph_edge('SIMD', 'IMM')

    genesys.add_subgraph_edge('SIMD', 'DRAM')
    genesys.add_subgraph_edge('DRAM', 'SIMD')
    genesys.add_subgraph_edge('OBUF', 'SIMD')
    add_genesys_templates(genesys)
    add_genesys_codelets(genesys)

    genesys.add_util_fn("extract_bits", ["val", "nb", "pos"], "((((1 << nb) - 1) << pos) & val) >> pos")
    return genesys

def define_genesys(conv_selection="nchw"):
    # TODO: Add capabilties to PE array not systolic_array
    GENESYS_CFG = {}
    GENESYS_CFG['ARRAY_N'] = 32
    GENESYS_CFG['ARRAY_M'] = 32
    GENESYS_CFG['DATA_WIDTH'] = 8 // 8
    GENESYS_CFG['WGT_WIDTH'] = 8 // 8
    GENESYS_CFG['BIAS_WIDTH'] = 32 // 8
    GENESYS_CFG['ACC_WIDTH'] = 32 // 8
    GENESYS_CFG['SIMD_WIDTH'] = 32
    GENESYS_CFG['IBUF_CAPACITY'] = GENESYS_CFG['ARRAY_N']*GENESYS_CFG['DATA_WIDTH']*4096
    GENESYS_CFG['WBUF_CAPACITY'] = GENESYS_CFG['ARRAY_M']*GENESYS_CFG['ARRAY_N']*GENESYS_CFG['DATA_WIDTH']*2048
    GENESYS_CFG['OBUF_CAPACITY'] = GENESYS_CFG['ARRAY_M']*GENESYS_CFG['ACC_WIDTH']*4096
    GENESYS_CFG['BBUF_CAPACITY'] = GENESYS_CFG['ARRAY_M']*GENESYS_CFG['BIAS_WIDTH']*4096
    GENESYS_CFG['VMEM_CAPACITY'] = GENESYS_CFG['OBUF_CAPACITY']*2
    GENESYS_CFG['IMM_CAPACITY'] = 32 * GENESYS_CFG['ACC_WIDTH']
    GENESYS_CFG['DRAM_CAPACITY'] = 1000000

    with ComputeNode("Genesys") as hag:
        vmem1 = StorageNode("VMEM1", access_type='RAM', size=GENESYS_CFG['VMEM_CAPACITY'],
                           latency=1, input_ports=2, output_ports=2)

        vmem2 = StorageNode("VMEM2", access_type='RAM', size=GENESYS_CFG['VMEM_CAPACITY'],
                           latency=1, input_ports=2, output_ports=2)

        imm = StorageNode("IMM", access_type='RAM', size=GENESYS_CFG['IMM_CAPACITY'],
                          latency=1, input_ports=2, output_ports=2)

        # TODO: Does this need to be added?
        # instr_mem = StorageNode("INSTR_MEM",access_type='RAM', size=16,
        #                         latency=1, input_ports=2, output_ports=2)

        dram = StorageNode("DRAM", access_type='RAM', size=GENESYS_CFG['DRAM_CAPACITY'],
                           latency=1, input_ports=2, output_ports=2,
                           on_chip=False)

        with ComputeNode("systolic_array") as systolic_array:
            pe_array = ComputeNode("pe_array", dimensions=[GENESYS_CFG['ARRAY_N'], GENESYS_CFG['ARRAY_M']])
            # TODO: Need to formalize the storage node sizes by # elements, width, and datatype
            ibuf = StorageNode("IBUF",
                               access_type='RAM', size=GENESYS_CFG['IBUF_CAPACITY'],
                               latency=1, input_ports=2, output_ports=2)
            wbuf = StorageNode("WBUF", access_type='RAM', size=GENESYS_CFG['WBUF_CAPACITY'],
                               latency=1, input_ports=2, output_ports=2)
            bbuf = StorageNode("BBUF", access_type='RAM', size=GENESYS_CFG['BBUF_CAPACITY'],
                               latency=1, input_ports=2, output_ports=2)
            obuf = StorageNode("OBUF", access_type='RAM', size=GENESYS_CFG['OBUF_CAPACITY'],
                               latency=1, input_ports=2, output_ports=2)
            # TODO: BW for DRAM is 64bits/cycle
            systolic_array.add_subgraph_edge('DRAM', 'IBUF', bandwidth=GENESYS_CFG['IBUF_CAPACITY'])
            systolic_array.add_subgraph_edge('DRAM', 'WBUF', bandwidth=GENESYS_CFG['WBUF_CAPACITY'])
            systolic_array.add_subgraph_edge('DRAM', 'BBUF', bandwidth=GENESYS_CFG['BBUF_CAPACITY'])
            systolic_array.add_subgraph_edge('DRAM', 'OBUF', bandwidth=GENESYS_CFG['OBUF_CAPACITY'])

            systolic_array.add_subgraph_edge('IBUF', 'pe_array', bandwidth=pe_array.dimensions[0]*GENESYS_CFG['DATA_WIDTH'])
            systolic_array.add_subgraph_edge('WBUF', 'pe_array', bandwidth=np.prod(pe_array.dimensions)*GENESYS_CFG['DATA_WIDTH'])
            systolic_array.add_subgraph_edge('BBUF', 'pe_array', bandwidth=pe_array.dimensions[1]*GENESYS_CFG['ACC_WIDTH'])
            systolic_array.add_subgraph_edge('OBUF', 'pe_array', bandwidth=pe_array.dimensions[1]*GENESYS_CFG['ACC_WIDTH'])
            systolic_array.add_subgraph_edge('OBUF', 'DRAM', bandwidth=GENESYS_CFG['OBUF_CAPACITY'])
            # TODO: Add OBUF TO DRAM EDGE
            systolic_array.add_subgraph_edge('pe_array', 'OBUF', bandwidth=pe_array.dimensions[1]*GENESYS_CFG['ACC_WIDTH'])
            for p in GENESYS_INSTRUCTIONS['systolic_array']:
                systolic_array.add_primitive(p)
        simd = ComputeNode("SIMD", dimensions=[32])
        hag.add_subgraph_edge('VMEM1', 'SIMD', bandwidth=GENESYS_CFG['SIMD_WIDTH']*GENESYS_CFG['ACC_WIDTH'])
        hag.add_subgraph_edge('SIMD', 'VMEM1', bandwidth=GENESYS_CFG['SIMD_WIDTH']*GENESYS_CFG['ACC_WIDTH'])
        hag.add_subgraph_edge('VMEM2', 'SIMD', bandwidth=GENESYS_CFG['SIMD_WIDTH']*GENESYS_CFG['ACC_WIDTH'])
        hag.add_subgraph_edge('SIMD', 'VMEM2', bandwidth=GENESYS_CFG['SIMD_WIDTH']*GENESYS_CFG['ACC_WIDTH'])
        hag.add_subgraph_edge('IMM', 'SIMD', bandwidth=GENESYS_CFG['SIMD_WIDTH']*GENESYS_CFG['ACC_WIDTH'])
        hag.add_subgraph_edge('SIMD', 'IMM', bandwidth=GENESYS_CFG['SIMD_WIDTH']*GENESYS_CFG['ACC_WIDTH'])
        # hag.add_subgraph_edge('SIMD', 'DRAM', bandwidth=GENESYS_CFG['SIMD_WIDTH']*GENESYS_CFG['ACC_WIDTH'])
        hag.add_subgraph_edge('DRAM', 'VMEM1', bandwidth=GENESYS_CFG['VMEM_CAPACITY'])
        hag.add_subgraph_edge('VMEM1', 'DRAM', bandwidth=GENESYS_CFG['VMEM_CAPACITY'])

        hag.add_subgraph_edge('DRAM', 'VMEM2', bandwidth=GENESYS_CFG['VMEM_CAPACITY'])
        hag.add_subgraph_edge('VMEM2', 'DRAM', bandwidth=GENESYS_CFG['VMEM_CAPACITY'])
        hag.add_subgraph_edge('OBUF', 'SIMD', bandwidth=GENESYS_CFG['SIMD_WIDTH']*GENESYS_CFG['ACC_WIDTH'])
        for p in GENESYS_INSTRUCTIONS['SIMD']:
            systolic_array.add_primitive(p)

        ## Set templates
        # Config
        for hag_node, templates in GENESYS_TEMPLATES['config'].items():
            hag.add_start_template(hag_node, templates['start'](hag))
            hag.add_end_template(hag_node, templates['end'](hag))

        # Transfer
        for hag_node, template in GENESYS_TEMPLATES['transfer'].items():
            hag.add_transfer_template(*hag_node, template(hag))

        # Compute
        for hag_node, template in GENESYS_TEMPLATES['compute'].items():
            hag.add_compute_template(*hag_node, template(hag))

        # Loop
        hag.add_loop_template("systolic_array", GENESYS_TEMPLATES['loop'](hag))

        for op_name, cdlt in GENESYS_CODELETS.items():
            cdlt_instance = cdlt(hag)
            hag.add_codelet(cdlt_instance)

        hag.add_util_fn("extract_bits", ["val", "nb", "pos"], "((((1 << nb) - 1) << pos) & val) >> pos")

    return hag


def add_genesys_templates(hag: ComputeNode):
    # Config
    for hag_node, templates in GENESYS_TEMPLATES['config'].items():
        hag.add_start_template(hag_node, templates['start'](hag))
        hag.add_end_template(hag_node, templates['end'](hag))

    # Transfer
    for hag_node, template in GENESYS_TEMPLATES['transfer'].items():
        hag.add_transfer_template(*hag_node, template(hag))

    # Compute
    for hag_node, template in GENESYS_TEMPLATES['compute'].items():
        hag.add_compute_template(*hag_node, template(hag))

    # Loop
    hag.add_loop_template("systolic_array", GENESYS_TEMPLATES['loop'](hag))


def add_genesys_codelets(hag: ComputeNode):
    for op_name, cdlt in GENESYS_CODELETS.items():
        hag.add_codelet(cdlt(hag))