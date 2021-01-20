from codelets.adl.graph import ComputeNode, StorageNode
from codelets.adl.flex_template import Instruction
from .genesys_instructions import GENESYS_INSTRUCTIONS
from .genesys_templates import GENESYS_TEMPLATES
from .genesys_codelets import GENESYS_CODELETS, conv2d_nhwc_test
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
        systolic_array.add_primitive(p())


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

def define_genesys(testing=False):
    with ComputeNode("Genesys") as hag:
        vmem = StorageNode("VMEM", read_bw=128, write_bw=128,
                           access_type='RAM', size=64,
                           latency=1, input_ports=2, output_ports=2)

        imm = StorageNode("IMM", read_bw=128, write_bw=128,
                          access_type='RAM', size=8,
                          latency=1, input_ports=2, output_ports=2)

        instr_mem = StorageNode("INSTR_MEM", read_bw=128, write_bw=128,
                                access_type='RAM', size=16,
                                latency=1, input_ports=2, output_ports=2)

        dram = StorageNode("DRAM", read_bw=512, write_bw=512,
                           access_type='RAM', size=1000000,
                           latency=1, input_ports=2, output_ports=2,
                           on_chip=False)
        with ComputeNode("systolic_array") as systolic_array:
            ibuf = StorageNode("IBUF", read_bw=128, write_bw=128,
                               access_type='RAM', size=32,
                               latency=1, input_ports=2, output_ports=2)
            wbuf = StorageNode("WBUF", read_bw=128, write_bw=128,
                               access_type='RAM', size=64,
                               latency=1, input_ports=2, output_ports=2)
            bbuf = StorageNode("BBUF", read_bw=128, write_bw=128,
                               access_type='RAM', size=8,
                               latency=1, input_ports=2, output_ports=2)
            obuf = StorageNode("OBUF", read_bw=128, write_bw=128,
                               access_type='RAM', size=128,
                               latency=1, input_ports=2, output_ports=2)
            pe_array = ComputeNode("pe_array", dimensions=[32, 32])
            systolic_array.add_subgraph_edge('IBUF', 'pe_array')
            systolic_array.add_subgraph_edge('WBUF', 'pe_array')
            systolic_array.add_subgraph_edge('BBUF', 'pe_array')
            systolic_array.add_subgraph_edge('OBUF', 'pe_array')
            systolic_array.add_subgraph_edge('pe_array', 'OBUF')
            for p in GENESYS_INSTRUCTIONS['systolic_array']:
                systolic_array.add_primitive(p())
        simd = ComputeNode("SIMD", dimensions=[32])
        hag.add_subgraph_edge('VMEM', 'SIMD')
        hag.add_subgraph_edge('SIMD', 'VMEM')
        hag.add_subgraph_edge('IMM', 'SIMD')
        hag.add_subgraph_edge('SIMD', 'IMM')
        hag.add_subgraph_edge('SIMD', 'DRAM')
        hag.add_subgraph_edge('DRAM', 'SIMD')
        hag.add_subgraph_edge('OBUF', 'SIMD')

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
        if testing:
            hag.add_codelet(conv2d_nhwc_test(hag))
        else:
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
