import json

from codelets import Datatype

OP_DTYPES = [Datatype(type='FXP', bitwidth=8, fractional=4, exp=4),
             Datatype(type='FXP', bitwidth=16, fractional=8, exp=8),
             Datatype(type='FXP', bitwidth=32, fractional=16, exp=16),
             Datatype(type='FP', bitwidth=16), Datatype(type='FP', bitwidth=32), Datatype(type='FXP', bitwidth=4)]

DTYPE_MAP = {}
DTYPE_MAP['FXP32'] = Datatype(type='FXP', bitwidth=32)
DTYPE_MAP['FXP16'] = Datatype(type='FXP', bitwidth=16)
DTYPE_MAP['FXP8'] = Datatype(type='FXP', bitwidth=8)
DTYPE_MAP['FXP4'] = Datatype(type='FXP', bitwidth=4)
DTYPE_MAP['FP32'] = Datatype(type='FXP', bitwidth=32)
DTYPE_MAP['FP16'] = Datatype(type='FXP', bitwidth=16)

GENESYS_DTYPES = {}
GENESYS_DTYPES['SIMD'] = 'FXP32'
GENESYS_DTYPES['SYSTOLIC_ARRAY'] = {}
GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight'] = 'FXP8'
GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out'] = 'FXP32'

def set_defaults(cfg):
    def set_default(name, default_val):
        if name not in cfg:
            cfg[name] = default_val

    if 'DATA_WIDTH' not in cfg:
        cfg['DATA_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight']].bits()
    if 'WGT_WIDTH' not in cfg:
        cfg['WGT_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['inp_weight']].bits()
    if 'BIAS_WIDTH' not in cfg:
        cfg['BIAS_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out']].bits()
    if 'ACC_WIDTH' not in cfg:
        cfg['ACC_WIDTH'] = DTYPE_MAP[GENESYS_DTYPES['SYSTOLIC_ARRAY']['bias_out']].bits()

    if 'TRAINING' not in cfg:
        cfg['TRAINING'] = False


    assert cfg['DATA_WIDTH'] == cfg['WGT_WIDTH']
    assert cfg['DATA_WIDTH']*4 == cfg['ACC_WIDTH']
    assert cfg['DATA_WIDTH']*4 == cfg['BIAS_WIDTH']
    if 'INSTR_WIDTH' not in cfg:
        cfg['INSTR_WIDTH'] = 32

    if 'BATCH_SIZE' not in cfg:
        cfg['BATCH_SIZE'] = 1

    assert isinstance(cfg['BATCH_SIZE'], int)

    assert 'ARRAY_N' in cfg, f"Invalid config, key 'ARRAY_N' not found"
    assert 'ARRAY_M' in cfg, f"Invalid config, key 'ARRAY_M' not found"
    assert 'IBUF_DEPTH' in cfg, f"Invalid config, key 'IBUF_DEPTH' not found"
    assert 'WBUF_DEPTH' in cfg, f"Invalid config, key 'WBUF_DEPTH' not found"
    assert 'OBUF_DEPTH' in cfg, f"Invalid config, key 'OBUF_DEPTH' not found"
    assert 'BBUF_DEPTH' in cfg, f"Invalid config, key 'BBUF_DEPTH' not found"
    assert 'PARAM_BUF_CHANNEL_BW' in cfg, f"Invalid config, key PARAM_BUF_CHANNEL_BW not found"

    if 'IBUF_CHANNEL_BW' not in cfg:
        cfg['IBUF_CHANNEL_BW'] = cfg['PARAM_BUF_CHANNEL_BW']

    if 'OBUF_CHANNEL_BW' not in cfg:
        cfg['OBUF_CHANNEL_BW'] = cfg['PARAM_BUF_CHANNEL_BW']

    if 'INSTR_CHANNEL_BW' not in cfg:
        cfg['INSTR_CHANNEL_BW'] = cfg['PARAM_BUF_CHANNEL_BW']

    if 'SIMD_CHANNEL_BW' not in cfg:
        cfg['SIMD_CHANNEL_BW'] = cfg['PARAM_BUF_CHANNEL_BW']

    assert cfg['PARAM_BUF_CHANNEL_BW'] == cfg['IBUF_CHANNEL_BW']
    assert cfg['PARAM_BUF_CHANNEL_BW'] == cfg['OBUF_CHANNEL_BW']
    assert cfg['PARAM_BUF_CHANNEL_BW'] == cfg['INSTR_CHANNEL_BW']
    assert cfg['PARAM_BUF_CHANNEL_BW'] == cfg['SIMD_CHANNEL_BW']

    if 'SIMD_WIDTH' not in cfg:
        cfg['SIMD_WIDTH'] = cfg['ARRAY_M']
    else:
        assert cfg['SIMD_WIDTH'] == cfg['ARRAY_M']

    if 'INSTR_DEPTH' not in cfg:
        cfg['INSTR_DEPTH'] = 1024
    if 'IMM_DEPTH' not in cfg:
        cfg['IMM_DEPTH'] = 64
    if 'DRAM_DEPTH' not in cfg:
        cfg['DRAM_DEPTH'] = 10000000
    if 'VMEM_DEPTH' not in cfg:
        cfg['VMEM_DEPTH'] = cfg['OBUF_DEPTH'] // 2 - 1
    else:
        assert cfg['VMEM_DEPTH'] == cfg['OBUF_DEPTH'] // 2 - 1
    if 'VMEM_BANKS' not in cfg:
        cfg['VMEM_BANKS'] = cfg['SIMD_WIDTH']
    else:
        assert cfg['VMEM_BANKS'] == cfg['SIMD_WIDTH']

    if 'INSTR_BANKS' not in cfg:
        cfg['INSTR_BANKS'] = 1

    if 'DRAM_WIDTH' not in cfg:
        cfg['DRAM_WIDTH'] = 8


    if 'DRAM_BANKS' not in cfg:
        cfg['DRAM_BANKS'] = cfg['SIMD_CHANNEL_BW'] // cfg['DRAM_WIDTH']
    else:
        assert cfg['DRAM_BANKS'] == cfg['SIMD_CHANNEL_BW'] // cfg['DRAM_WIDTH']
    set_default('SW_PIPELINE_TEST', False)
    set_default('ASIC_CONFIG', False)
    set_default('ADDR_GEN_TEST', False)

    assert 'SW_PIPELINE_TEST' in cfg
    assert 'USE_QUANTIZATION' in cfg
    assert 'ADDR_GEN_TEST' in cfg
    assert 'FUSE_LAYERS' in cfg


    if cfg['FUSE_LAYERS']:
        cfg['FUSION_CONSTRAINTS'] = True
    else:
        cfg['FUSION_CONSTRAINTS'] = False

    if 'GENERATE_INSTRUCTIONS' not in cfg:
        cfg['GENERATE_INSTRUCTIONS'] = True

    if 'SHARED_DATAGEN' not in cfg:
        cfg['SHARED_DATAGEN'] = False

    if 'LOOP_OVERHEAD' not in cfg:
        cfg['LOOP_OVERHEAD'] = False

    if 'DATAGEN' not in cfg:
        cfg['DATAGEN'] = False

    if 'DEBUG_MMUL_COORDS' not in cfg:
        cfg['DEBUG_MMUL_COORDS'] = None
    else:
        assert isinstance(cfg['DEBUG_MMUL_COORDS'], dict)
        assert 'COORD' in cfg['DEBUG_MMUL_COORDS'] and isinstance(cfg['DEBUG_MMUL_COORDS']['COORD'], list)
        assert cfg['DATAGEN']

    if 'SIMD_ONLY_FUSIONS' not in cfg:
        cfg['SIMD_ONLY_FUSIONS'] = False

    if cfg['SIMD_ONLY_FUSIONS']:
        assert cfg['FUSE_LAYERS']

    assert 'ASIC_CONFIG' in cfg
    assert 'SA_TILE_CONSTR' in cfg

    if 'SIMD_BASE_ADDR' in cfg:
        assert isinstance(cfg['SIMD_BASE_ADDR'], dict)
        assert 'LD_VMEM1' in cfg['SIMD_BASE_ADDR']
        assert 'LD_VMEM2' in cfg['SIMD_BASE_ADDR']
        assert 'ST_VMEM1' in cfg['SIMD_BASE_ADDR']
        assert 'ST_VMEM2' in cfg['SIMD_BASE_ADDR']
    else:
        # cfg['SIMD_BASE_ADDR'] = {}
        # cfg['SIMD_BASE_ADDR']['LD_VMEM1'] = 0
        # cfg['SIMD_BASE_ADDR']['LD_VMEM2'] = 1024 << 10
        # cfg['SIMD_BASE_ADDR']['ST_VMEM1'] = 1024 << 11
        # cfg['SIMD_BASE_ADDR']['ST_VMEM2'] = 1024 << 12
        cfg['SIMD_BASE_ADDR'] = {}
        cfg['SIMD_BASE_ADDR']['LD_VMEM1'] = 0
        cfg['SIMD_BASE_ADDR']['LD_VMEM2'] = 0
        cfg['SIMD_BASE_ADDR']['ST_VMEM1'] = 0
        cfg['SIMD_BASE_ADDR']['ST_VMEM2'] = 0

    if 'MERGE_LDST_LOOPS' not in cfg:
        cfg['MERGE_LDST_LOOPS'] = True
    else:
        assert isinstance(cfg['MERGE_LDST_LOOPS'], bool)

    if 'SA_BASE_ADDR' not in cfg:
        # cfg['SA_BASE_ADDR'] = {
        #     "INSTR": 0, "OBUF": 0, "BBUF": 4096, "WBUF": 24576, "IBUF": 4259840
        # }
        cfg['SA_BASE_ADDR'] = {
            "INSTR": 0, "OBUF": 0, "BBUF": 0, "WBUF": 0, "IBUF": 0
        } 
    else:
        assert "INSTR" in cfg['SA_BASE_ADDR']
        assert "OBUF" in cfg['SA_BASE_ADDR']
        assert "BBUF" in cfg['SA_BASE_ADDR']
        assert "WBUF" in cfg['SA_BASE_ADDR']
        assert "IBUF" in cfg['SA_BASE_ADDR']

    if 'OBUF_TO_VMEM_TEST' not in cfg:
        cfg['OBUF_TO_VMEM_TEST'] = False

    if 'SINGLE_PROGRAM_COMPILATION' not in cfg:
        cfg['SINGLE_PROGRAM_COMPILATION'] = False
    else:
        assert isinstance(cfg['SINGLE_PROGRAM_COMPILATION'], bool)

    if 'LD_ST_OVERHEAD' not in cfg:
        cfg['LD_ST_OVERHEAD'] = False

    if 'TPU_TEST' not in cfg:
        cfg['TPU_TEST'] = False

    assert not (cfg['TPU_TEST'] and cfg['LD_ST_OVERHEAD'])

    if 'INSTR_MEM_ALIGN' not in cfg:
        cfg['INSTR_MEM_ALIGN'] = 4096*8

    if 'ADDR_ALIGNMENT' not in cfg:
        cfg['ADDR_ALIGNMENT'] = 4096*8

    return cfg

def load_config(fpath):
    with open(fpath, "r") as f:
        genesys_cfg = json.load(f)

    return set_defaults(genesys_cfg)