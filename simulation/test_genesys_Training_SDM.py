from examples import define_genesys, compile_genesys,\
    GENESYS_CFG, GENESYS_DTYPES, DTYPE_MAP
import polymath as pm
from pprint import pprint
from codelets import initialize_program, tile, hoist, pad_operands
from collections import namedtuple
import json
import datetime
from pathlib import Path
import numpy as np

CWD = Path(f"{__file__}").parent
TEST_DIR = f"{CWD}/input_files"
#BENCH_DIR = f"{CWD}/../benchmarks"
BENCH_DIR = Path(f"{CWD}/../benchmarks").absolute()
MODEL_DIR = f"{BENCH_DIR}/models/srdfg"
LAYER_DIR = f"{BENCH_DIR}/layers/srdfg"

TestDfgNode = namedtuple('TestDfgNode', ['input_components', 'input_shapes', 'attrs'])
GENESYS_CFG_PATH = f"{CWD}/scratch/genesys_cfg.json"

def update_genesys_cfg_from_dtypes():
    GENESYS_CFG['DATA_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight']].bits()
    GENESYS_CFG['WGT_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight']].bits()
    GENESYS_CFG['BIAS_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out']].bits()
    GENESYS_CFG['ACC_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out']].bits()

def parse_cfg():
    with open(GENESYS_CFG_PATH) as f:
        genesys = json.load(f)
    return genesys

def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()

def test_genesys_model(Hardware_config, Batch_size, CompOutput_file_name, CompilerOut_directory):
    #graph = pm.pb_load(f"{LAYER_DIR}/resnet18_gemm.srdfg")
    BENCH_DIR = Path(f"{CWD}/../benchmarks").absolute()
    #print(Hardware_config)
    #pprint(GENESYS_CFG)
    ######################## Parsing the hardware configuration file ####################
    GENESYS_CFG['ARRAY_N'] = Hardware_config['ARRAY_N']
    GENESYS_CFG['ARRAY_M'] = Hardware_config['ARRAY_M']

    GENESYS_CFG['INSTR_WIDTH'] =  Hardware_config['ARRAY_N']  # these two are fixed parameters across designs
    GENESYS_CFG['SIMD_WIDTH'] = Hardware_config['ARRAY_N']

    GENESYS_CFG['PARAM_BUF_CHANNEL_BW'] = Hardware_config['PARAMBUF_AXI_DATA_WIDTH']
    GENESYS_CFG['IBUF_CHANNEL_BW'] = Hardware_config['IBUF_AXI_DATA_WIDTH']
    GENESYS_CFG['OBUF_CHANNEL_BW'] = Hardware_config['OBUF_AXI_DATA_WIDTH']
    GENESYS_CFG['INSTR_CHANNEL_BW'] = Hardware_config['INST_MEM_AXI_DATA_WIDTH']
    GENESYS_CFG['SIMD_CHANNEL_BW'] = Hardware_config['SIMD_AXI_DATA_WIDTH']

    GENESYS_CFG['IBUF_DEPTH'] = Hardware_config['IBUF_CAPACITY_BITS']/(Hardware_config['ARRAY_N'] * Hardware_config['ACT_DATA_WIDTH'])
    GENESYS_CFG['WBUF_DEPTH'] =  Hardware_config['WBUF_CAPACITY_BITS']/(Hardware_config['ARRAY_N'] * Hardware_config['ARRAY_M'] * Hardware_config['WGT_DATA_WIDTH'])
    GENESYS_CFG['OBUF_DEPTH'] =  2 * Hardware_config['OBUF_CAPACITY_BITS']/(Hardware_config['ARRAY_M'] * Hardware_config['ACC_WIDTH'])
    GENESYS_CFG['BBUF_DEPTH'] =  Hardware_config['BBUF_CAPACITY_BITS']/(Hardware_config['ARRAY_M'] * Hardware_config['BIAS_WIDTH'])
    GENESYS_CFG['INSTR_DEPTH'] =  2 * Hardware_config['INST_MEM_CAPACITY_BITS']/32
    GENESYS_CFG['IMM_DEPTH'] =  32
    GENESYS_CFG['VMEM_DEPTH'] =  2 * Hardware_config['SIMD_VMEM_CAPACITY_BITS']/(Hardware_config['ARRAY_M'] * 32)
    
    #print("IBUF_DEPTH:", GENESYS_CFG['IBUF_DEPTH'])
    #print("WBUF_DEPTH:", GENESYS_CFG['WBUF_DEPTH'])
    #print("OBUF_DEPTH:", GENESYS_CFG['OBUF_DEPTH'])
    #print("BBUF_DEPTH:", GENESYS_CFG['BBUF_DEPTH'])
    #print("INSTR_DEPTH:", GENESYS_CFG['INSTR_DEPTH'])
    #print("VMEM_DEPTH:", GENESYS_CFG['VMEM_DEPTH'])

    GENESYS_CFG['VMEM_BANKS'] = Hardware_config['ARRAY_M']
    GENESYS_CFG['INSTR_BANKS'] = 1
    #GENESYS_CFG['DRAM_BANKS'] =  1
    #GENESYS_CFG['DRAM_DEPTH'] =  100000

    # In current designs, weight and activation bitwidths are same and bias and accumulation bitwidths are same
    bw_weight = Hardware_config['WGT_DATA_WIDTH']
    bw_bias = Hardware_config['BIAS_WIDTH']
    print(bw_weight)

    GENESYS_DTYPES['SIMD'] = 'FXP32'
    GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight'] = 'FXP' + str(bw_weight)
    GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out'] = 'FXP' + str(bw_bias)
    #update_genesys_cfg_from_dtypes()
    pprint(GENESYS_CFG)

    ################################### Generating compiler output##################
    #batch_size_pass = pm.UpdateBatchSize(Batch_size, graph.op_name)
    #graph = batch_size_pass(graph)
    #genesys = define_genesys(GENESYS_CFG)
    #program = initialize_program
    # (graph, genesys)
    #program.add_compilation_step("update_operand_dtypes", update_operand_dtypes, preproc=True, stage_kwargs={'dtype_map': GENESYS_DTYPES})
    #program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': []})
    #program.add_compilation_step("tile", tile)
    #program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    #program.compile()
    
    program = compile_genesys('resnet18',
                          train=True,
                          update_cfg_dtypes=True,
                          tiling_path=None,
                          batch_size=Batch_size,
                          store_tiling=False,
                          store_json_output=False,
                          benchmark_path=BENCH_DIR,
                          genesys_cfg = GENESYS_CFG,
                          dtypes = GENESYS_DTYPES,
                          verbose = True
                           )

    res = program.emit("json_no_ops")
    pprint(res)

    compiler_output_file_name = "compiled_" + CompOutput_file_name + "_nop" + ".json"
    with open(CompilerOut_directory/compiler_output_file_name, "w") as f:
        json.dump(res, f, indent=4)
    

def test_genesys_resnet18():
    graph = pm.pb_load(f"{MODEL_DIR}/resnet18.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': []})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("json")
    #print(res)

    with open("compiled_resnet18_full_withop.json", "w") as f:
        json.dump(res, f, indent=4, default=myconverter)


if __name__ == '__main__':
    #test_genesys_resnet18()
    #test_genesys_gemm()

    ####### Code to autometically generate compiler output for multiple design points, Training
    Hardware_directory = Path(f"{CWD}/hw_json_training_cfg/")
    CompilerOut_directory = Path(f"{CWD}/compiler_output/")

    DNN_benchmark = "ResNet18_"
    Compute_Phase =  "Train_"

    Batch_size = 256  #provide the intendent batch size here

    if Batch_size == 1:
        Batching = "SingleB_"
    elif Batch_size > 1:
        Batching = "MultiB_"

    hardware_list_inf = ['01', '02', '05', '06', '08', '09', '11', '12']  #list of design hardware for Training
    #hardware_list_inf = ['01'] # to test run only one design
    for hrd_design_no in hardware_list_inf:
        HD_json_name = "genesys_params_design" + hrd_design_no + '.json'
        print(HD_json_name)
        CompOutput_file_name = DNN_benchmark + Compute_Phase + Batching + "Design" + hrd_design_no
        print(CompOutput_file_name)

        with open(Hardware_directory/HD_json_name) as f:
            Hardware_config = json.load(f)
        #print(Hardware_config)

        test_genesys_model(Hardware_config, Batch_size, CompOutput_file_name, CompilerOut_directory)
















