from codelets.adl.graph import ComputeNode, StorageNode
from codelets.adl.flex_template import Instruction
from codelets import initialize_program, tile, hoist, pad_operands, update_operand_dtypes
from .genesys_instructions import GENESYS_INSTRUCTIONS
from .genesys_templates import GENESYS_TEMPLATES
from .genesys_inference_codelets import GENESYS_CODELETS
from . import GENESYS_CFG, GENESYS_DTYPES, DTYPE_MAP
import numpy as np
from pathlib import Path
import json
from pprint import pprint

from . import SIMD_NS, SIMD_OPCODE_BITWIDTH, OP_DTYPES, \
    OP_LOCATIONS, NS_BITWIDTH, NS_IDX_BITWIDTH
import polymath as pm

CWD = Path(f"{__file__}").parent
BENCH_DIR = f"{CWD}/../../../benchmarks"
OUT_DIR = f"{BENCH_DIR}/compiler_outputs"
MODEL_DIR = f"{BENCH_DIR}/models/srdfg"
TILING_DIR = f"{BENCH_DIR}/tiling_info"



LOOPS_PER_LEVEL = 7
INCR_MAP = "{'LD': {'IBUF': 0, 'WBUF': 1, 'OBUF': 2, 'BBUF': 3}," \
           "'ST': {'IBUF': 4, 'WBUF': 5, 'OBUF': 6, 'BBUF': 7}}"
LD_ST_MAP = "{'LD': 0, 'ST': 1}"

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

def define_genesys(cfg):
    # TODO: Add capabilties to PE array not systolic_array

    with ComputeNode("Genesys") as hag:
        vmem1 = StorageNode("VMEM1", access_type='RAM', banks=cfg['SIMD_WIDTH'],
                            width=cfg['ACC_WIDTH'], depth=cfg['VMEM_DEPTH'],
                            latency=1, input_ports=2, output_ports=2)

        vmem2 = StorageNode("VMEM2", access_type='RAM', banks=cfg['SIMD_WIDTH'],
                            width=cfg['ACC_WIDTH'], depth=cfg['VMEM_DEPTH'],
                           latency=1, input_ports=2, output_ports=2)

        imm = StorageNode("IMM", access_type='RAM', banks=cfg['SIMD_WIDTH'],
                            width=cfg['ACC_WIDTH'], depth=cfg['IMM_DEPTH'],
                          latency=1, input_ports=2, output_ports=2)

        # TODO: Does this need to be added?
        instr_mem = StorageNode("INSTR_MEM",access_type='RAM', width=cfg['INSTR_WIDTH'],
                                banks=cfg['INSTR_BANKS'], depth=cfg['INSTR_DEPTH'],
                                latency=1, input_ports=2, output_ports=2)

        dram = StorageNode("DRAM", access_type='RAM', banks=cfg['DRAM_BANKS'],
                            width=cfg['ACC_WIDTH'], depth=cfg['DRAM_DEPTH'],
                           latency=1, input_ports=2, output_ports=2,
                           on_chip=False)

        with ComputeNode("systolic_array") as systolic_array:
            pe_array = ComputeNode("pe_array", dimensions=[cfg['ARRAY_N'], cfg['ARRAY_M']])
            # TODO: Need to formalize the storage node sizes by # elements, width, and datatype
            ibuf = StorageNode("IBUF",
                               access_type='RAM', banks=cfg['ARRAY_N'],
                                width=cfg['DATA_WIDTH'], depth=cfg['IBUF_DEPTH'],
                               latency=1, input_ports=2, output_ports=2)
            wbuf = StorageNode("WBUF", access_type='RAM', banks=cfg['ARRAY_N']*cfg['ARRAY_M'],
                                width=cfg['DATA_WIDTH'], depth=cfg['WBUF_DEPTH'],
                               latency=1, input_ports=2, output_ports=2)
            bbuf = StorageNode("BBUF", access_type='RAM', banks=cfg['ARRAY_M'],
                                width=cfg['DATA_WIDTH'], depth=cfg['BBUF_DEPTH'],
                               latency=1, input_ports=2, output_ports=2)
            obuf = StorageNode("OBUF", access_type='RAM', banks=cfg['ARRAY_M'],
                                width=cfg['DATA_WIDTH'], depth=cfg['OBUF_DEPTH'],
                               latency=1, input_ports=2, output_ports=2)
            # TODO: BW for DRAM is 64bits/cycle
            # Channel bandwidth = axi bandwidth
            # Request chunk of data for each iteration
            # Request size * data width * banks
            # iters = tile_size / (Request size * data width * banks)

            systolic_array.add_subgraph_edge('DRAM', 'IBUF', bandwidth=cfg['IBUF_CHANNEL_BW'])
            systolic_array.add_subgraph_edge('DRAM', 'WBUF', bandwidth=cfg['PARAM_BUF_CHANNEL_BW'])
            systolic_array.add_subgraph_edge('DRAM', 'BBUF', bandwidth=cfg['PARAM_BUF_CHANNEL_BW'])
            systolic_array.add_subgraph_edge('DRAM', 'OBUF', bandwidth=cfg['OBUF_CHANNEL_BW'])
            systolic_array.add_subgraph_edge('DRAM', 'INSTR_MEM', bandwidth=cfg['INSTR_CHANNEL_BW'])

            systolic_array.add_subgraph_edge('IBUF', 'pe_array', bandwidth=pe_array.dimensions[0]*cfg['DATA_WIDTH'])
            systolic_array.add_subgraph_edge('WBUF', 'pe_array', bandwidth=np.prod(pe_array.dimensions)*cfg['DATA_WIDTH'])
            systolic_array.add_subgraph_edge('BBUF', 'pe_array', bandwidth=pe_array.dimensions[1]*cfg['ACC_WIDTH'])
            systolic_array.add_subgraph_edge('OBUF', 'pe_array', bandwidth=pe_array.dimensions[1]*cfg['ACC_WIDTH'])
            systolic_array.add_subgraph_edge('OBUF', 'DRAM', bandwidth=cfg['OBUF_CHANNEL_BW'])
            # TODO: Add OBUF TO DRAM EDGE
            systolic_array.add_subgraph_edge('pe_array', 'OBUF', bandwidth=pe_array.dimensions[1]*cfg['ACC_WIDTH'])
            for p in GENESYS_INSTRUCTIONS['systolic_array']:
                systolic_array.add_primitive(p)
        simd = ComputeNode("SIMD", dimensions=[cfg['SIMD_WIDTH']])
        hag.add_subgraph_edge('VMEM1', 'SIMD', bandwidth=cfg['SIMD_WIDTH']*cfg['ACC_WIDTH'])
        hag.add_subgraph_edge('SIMD', 'VMEM1', bandwidth=cfg['SIMD_WIDTH']*cfg['ACC_WIDTH'])
        hag.add_subgraph_edge('VMEM2', 'SIMD', bandwidth=cfg['SIMD_WIDTH']*cfg['ACC_WIDTH'])
        hag.add_subgraph_edge('SIMD', 'VMEM2', bandwidth=cfg['SIMD_WIDTH']*cfg['ACC_WIDTH'])
        hag.add_subgraph_edge('IMM', 'SIMD', bandwidth=cfg['SIMD_WIDTH']*cfg['ACC_WIDTH'])
        hag.add_subgraph_edge('SIMD', 'IMM', bandwidth=cfg['SIMD_WIDTH']*cfg['ACC_WIDTH'])
        # hag.add_subgraph_edge('SIMD', 'DRAM', bandwidth=GENESYS_CFG['SIMD_WIDTH']*GENESYS_CFG['ACC_WIDTH'])
        hag.add_subgraph_edge('DRAM', 'VMEM1', bandwidth=cfg['SIMD_CHANNEL_BW'])
        hag.add_subgraph_edge('VMEM1', 'DRAM', bandwidth=cfg['SIMD_CHANNEL_BW'])

        hag.add_subgraph_edge('DRAM', 'VMEM2', bandwidth=cfg['SIMD_CHANNEL_BW'])
        hag.add_subgraph_edge('VMEM2', 'DRAM', bandwidth=cfg['SIMD_CHANNEL_BW'])
        hag.add_subgraph_edge('OBUF', 'SIMD', bandwidth=cfg['SIMD_WIDTH']*cfg['ACC_WIDTH'])
        for p in GENESYS_INSTRUCTIONS['SIMD']:
            simd.add_primitive(p)

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
        hag.add_util_fn("get_loop_level_id", ["buffer_name", "loop_id", "level", "ld_st"], f"(loop_id % {LOOPS_PER_LEVEL}) + {LOOPS_PER_LEVEL} * level + ({INCR_MAP})[ld_st][buffer_name]")
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

def update_genesys_cfg_from_dtypes(inp_cfg=None, dtypes=None):
    if inp_cfg:
        assert dtypes is not None
        inp_cfg['DATA_WIDTH'] = DTYPE_MAP[dtypes['SYSTOLIC_ARRAY']['inp_weight']].bits()
        inp_cfg['WGT_WIDTH'] = DTYPE_MAP[dtypes['SYSTOLIC_ARRAY']['inp_weight']].bits()
        inp_cfg['BIAS_WIDTH'] = DTYPE_MAP[dtypes['SYSTOLIC_ARRAY']['bias_out']].bits()
        inp_cfg['ACC_WIDTH'] = DTYPE_MAP[dtypes['SYSTOLIC_ARRAY']['bias_out']].bits()
        out_cfg = inp_cfg
    else:
        GENESYS_CFG['DATA_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight']].bits()
        GENESYS_CFG['WGT_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight']].bits()
        GENESYS_CFG['BIAS_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out']].bits()
        GENESYS_CFG['ACC_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out']].bits()
        out_cfg = GENESYS_CFG
    return out_cfg


def compile_genesys(model_name,
                    train=False,
                    update_cfg_dtypes=False,
                    tiling_path=None,
                    batch_size=1,
                    store_tiling=False,
                    store_json_output=False,
                    json_output_filename=None,
                    verbose=False,
                    benchmark_path=None,
                    genesys_cfg=None,
                    dtypes=None,
                    print_config=True,
                    store_ops=False,
                    factor_fn='default'):
    MODEL_DIR = f"{benchmark_path}/models/srdfg"
    OUT_DIR = f"{benchmark_path}/compiler_outputs"

    TILING_DIR = f"{benchmark_path}/tiling_info"
    dtypes = dtypes or GENESYS_DTYPES
    if update_cfg_dtypes:
        def_cfg = update_genesys_cfg_from_dtypes(inp_cfg=genesys_cfg, dtypes=dtypes)
    else:
        def_cfg = GENESYS_CFG

    if print_config:
        print(f"Compiling model with the following config:\n")
        pprint(def_cfg)
    if model_name not in ['resnet50', 'resnet18', 'maskrcnn']:
        raise RuntimeError(f"Invalid model name for compilation")
    if train:
        model_name = f"{model_name}_train"
    graph = pm.pb_load(f"{MODEL_DIR}/{model_name}.srdfg")

    if batch_size > 1:
        batch_size_pass = pm.UpdateBatchSize(batch_size, graph.op_name)
        graph = batch_size_pass(graph)

    if train:
        graph = pm.create_training_graph(graph)

    layout_pass = pm.UpdateLayout('nchw', 'nhwc')
    multi_dim_pass = pm.RenameMultiDimOps()
    graph = multi_dim_pass(graph)
    graph = layout_pass(graph)
    genesys = define_genesys(def_cfg)
    mode = "training" if train else "inference"
    program = initialize_program(graph, genesys, mode=mode)
    program.add_compilation_step("update_operand_dtypes", update_operand_dtypes, preproc=True, stage_kwargs={'dtype_map': dtypes})
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': {}})
    tile_kwargs = {'factor_fn_name': factor_fn}
    if store_tiling:
        tile_kwargs['checkpoint_file'] = str(Path(f"{TILING_DIR}/{graph.name}_tiling_info_checkpoint.json").absolute())
    program.add_compilation_step("tile", tile, stage_kwargs=tile_kwargs)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])

    if tiling_path is not None:
        program.compile(tiling_path=f"{TILING_DIR}/{tiling_path}", verbose=verbose)
    else:
        program.compile(verbose=verbose)

    if store_tiling:
        program.store_tiling(f"{TILING_DIR}")

    if store_json_output:
        out_type = "json" if store_ops else "json_no_ops"
        res = program.emit(out_type)

        if json_output_filename is not None:
            with open(json_output_filename, "w") as outfile:
                json.dump(res, outfile, indent=4)
        else:
            store_dir = f"{OUT_DIR}/{model_name}_compiled"
            p = Path(f"{store_dir}.json")
            if p.exists():
                count = 0
                while Path(f"{store_dir}{count}.json").exists():
                    count += 1
                with open(f"{store_dir}{count}.json", "w") as outfile:
                    json.dump(res, outfile, indent=4)
            else:
                with open(f"{store_dir}.json", "w") as outfile:
                    json.dump(res, outfile, indent=4)
    return program


def compile_genesys_layer(layer_file,
                    update_cfg_dtypes=False,
                    tiling_path=None,
                    store_tiling=False,
                    store_json_output=False,
                    json_output_filename=None,
                    verbose=False,
                    benchmark_path=None,
                    genesys_cfg=None,
                    dtypes=None,
                    print_config=True,
                    store_ops=False,
                          store_checkpoint=False,factor_fn='default',
                          batch_size=1):
    LAYER_DIR = f"{benchmark_path}/layers/srdfg"
    OUT_DIR = f"{benchmark_path}/compiler_outputs"

    TILING_DIR = f"{benchmark_path}/tiling_info"
    dtypes = dtypes or GENESYS_DTYPES
    if update_cfg_dtypes:
        def_cfg = update_genesys_cfg_from_dtypes(inp_cfg=genesys_cfg, dtypes=dtypes)
    else:
        def_cfg = GENESYS_CFG

    if print_config:
        print(f"Compiling model with the following config:\n")
        pprint(def_cfg)

    graph = pm.pb_load(f"{LAYER_DIR}/{layer_file}.srdfg")
    if batch_size > 1:
        batch_size_pass = pm.UpdateBatchSize(batch_size, graph.op_name)
        graph = batch_size_pass(graph)
    layout_pass = pm.UpdateLayout('nchw', 'nhwc')
    multi_dim_pass = pm.RenameMultiDimOps()
    graph = multi_dim_pass(graph)
    graph = layout_pass(graph)
    genesys = define_genesys(def_cfg)
    mode = "inference"
    program = initialize_program(graph, genesys, mode=mode)
    program.add_compilation_step("update_operand_dtypes", update_operand_dtypes, preproc=True, stage_kwargs={'dtype_map': dtypes})
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': {}})
    tile_kwargs = {'factor_fn_name': factor_fn}
    if store_tiling and store_checkpoint:
        tile_kwargs['checkpoint_file'] = str(Path(f"{TILING_DIR}/{graph.name}_tiling_info_checkpoint.json").absolute())
    program.add_compilation_step("tile", tile, stage_kwargs=tile_kwargs)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])

    if tiling_path is not None:
        program.compile(tiling_path=f"{TILING_DIR}/{tiling_path}", verbose=verbose)
    else:
        program.compile(verbose=verbose)

    if store_tiling:
        program.store_tiling(f"{TILING_DIR}")

    if store_json_output:
        out_type = "json" if store_ops else "json_no_ops"
        res = program.emit(out_type)

        if json_output_filename is not None:
            with open(json_output_filename, "w") as outfile:
                json.dump(res, outfile, indent=4)
        else:
            store_dir = f"{OUT_DIR}/{layer_file}_compiled"
            p = Path(f"{store_dir}.json")
            if p.exists():
                count = 0
                while Path(f"{store_dir}{count}.json").exists():
                    count += 1
                with open(f"{store_dir}{count}.json", "w") as outfile:
                    json.dump(res, outfile, indent=4)
            else:
                with open(f"{store_dir}.json", "w") as outfile:
                    json.dump(res, outfile, indent=4)
    return program


def add_genesys_codelets(hag: ComputeNode):
    for op_name, cdlt in GENESYS_CODELETS.items():
        hag.add_codelet(cdlt(hag))
