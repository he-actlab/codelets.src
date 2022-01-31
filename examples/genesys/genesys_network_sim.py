from pathlib import Path
from examples.genesys import compile_genesys, get_arch
from tools.compile_layer import store_compilation_output
from codelets.compiler.program import CodeletProgram
from pprint import pprint
import os
CWD = Path(f"{__file__}").parent

BENCH_DIR = Path(f"{Path(__file__).parent}/../../benchmarks")
MODEL_DIR = Path(f"{Path(__file__).parent}/../../benchmarks/models")

def generate_inputs(inputs):
    pass

def generate_values(cdlt, layer_name, inputs=None):
    pass

def store_model_values(program: CodeletProgram,  model_name, base_path, use_random=True, format="nhwc",
                       load_path=None, actual_data=None, store_partials=None):
    value_dict = {"inputs": {},
                  "intermediate": {},
                  "outputs": {}
                  }
    for c in program.codelets:
        for i in c.inputs:
            assert i.node_name in program.operand_mapping

    operand_map = {}
    # for c in program.codelets:
    #     print(c.op_name)

def store_model_outputs(model_name,
                  training_mode,
                  batch_size=1,
                  verbose=False,
                  emit_to_stdout=None,
                  load_path=None,
                  dir_ext=None,
                  actual_data=False,
                  use_random=False,
                  store_partials=False,
                  program=None, added_constr=None):
    name = model_name
    tile_method = "min_tiles"
    # tile_method = "valid_split"

    tiling_path = None
    store_tiling = False
    store_json_output = False
    json_output_filename = None
    update_cfg_dtypes = False

    BENCH_DIR = Path(f"{CWD}/../benchmarks").absolute()
    print(f"Creating compilation output for {name}\n")
    if program is None:

        model_files = [d.name.split(".")[0] for d in os.scandir(f"{BENCH_DIR}/models/srdfg")]
        if name in model_files and not training_mode:
            program = compile_genesys(name,
                                      update_cfg_dtypes=update_cfg_dtypes,
                                      tiling_path=tiling_path,
                                      store_tiling=store_tiling,
                                      store_checkpoint=False,
                                      store_json_output=store_json_output,
                                      json_output_filename=json_output_filename,
                                      verbose=verbose,
                                      benchmark_path=BENCH_DIR,
                                      factor_fn='default',
                                    batch_size=batch_size,
                                    do_hoist_stage=True,
                                    tiling_search_algorithm=tile_method,
                                    do_tile_stage=True,
                                    print_config=False,
                                            do_compile=False
                                      )
        elif name in model_files:
            name = name.split("_")[0]
            program = compile_genesys(name,
                                      train=True,
                                      update_cfg_dtypes=update_cfg_dtypes,
                                      tiling_path=tiling_path,
                                      store_tiling=store_tiling,
                                      store_checkpoint=False,
                                      store_json_output=store_json_output,
                                      json_output_filename=json_output_filename,
                                      verbose=verbose,
                                      benchmark_path=BENCH_DIR,
                                      factor_fn='default',
                                    batch_size=batch_size,
                                    do_hoist_stage=True,
                                    tiling_search_algorithm=tile_method,
                                    do_tile_stage=True,
                                    print_config=False,
                                            do_compile=False
                                      )
        else:
            raise RuntimeError(f"Invalid layer name for compilation : {name}")

    if added_constr:
        program = update_tile_constraints(program, added_constr, model_name)
    program.compile(verbose=False, finalize=True)

    arch_cfg = get_arch(None, None, update_cfg_dtypes)
    print(f"Configuration for program:")
    # pprint.pprint(arch_cfg)
    if emit_to_stdout is not None:
        assert isinstance(emit_to_stdout, str)
        if "json" in emit_to_stdout:
            pprint.pprint(program.emit(emit_to_stdout))

    base_path = store_compilation_output(program, "arch_cfg", extension="json", arch_cfg=arch_cfg, dir_ext=dir_ext)
    store_compilation_output(program, "operations_idx", extension="txt", dir_ext=dir_ext)
    store_compilation_output(program, "json", extension="json", dir_ext=dir_ext)
    store_compilation_output(program, "string_final", extension="txt", dir_ext=dir_ext)
    store_compilation_output(program, "decimal", extension="txt", dir_ext=dir_ext)
    store_compilation_output(program, "binary", extension="txt", dir_ext=dir_ext)
    store_model_values(program, model_name, base_path, use_random=use_random, load_path=load_path,
                 actual_data=actual_data,
                 store_partials=store_partials)
    return program

def update_tile_constraints(program, layer_constraints, orig_constraint=None):
    # TODO: Fix this to add constraints on a per-layer or per-network basis
    for layer_type, constr in layer_constraints.items():
        if 'LEVEL1_hint' not in program.hag.codelets[layer_type].compilation_params.keys():
            program.hag.codelets[layer_type].compilation_params['LEVEL1_hint'] = constr
        elif constr not in program.hag.codelets[layer_type].compilation_params['LEVEL1_hint']:
            orig = program.hag.codelets[layer_type].compilation_params['LEVEL1_hint']
            new_constraint = f"{orig} and {constr}"
            program.hag.codelets[layer_type].compilation_params['LEVEL1_hint'] = new_constraint

    return program

def compile_full_model(model_name, store_compile=False, dir_ext=None,
                              partials=False, added_constr=None, train_mode=False,
                       verbose=False):

    model_path = f"{MODEL_DIR}/{model_name}.onnx"

    batch_size = 1
    tile_method = "min_tiles"

    update_cfg_dtypes = False
    tiling_path = None
    store_tiling = False
    store_json_output = False
    json_output_filename = None

    # This function returns
    program = compile_genesys(model_name,
                              update_cfg_dtypes=update_cfg_dtypes,
                              tiling_path=tiling_path,
                              store_tiling=store_tiling,
                              store_checkpoint=False,
                              train=train_mode,
                              store_json_output=store_json_output,
                              json_output_filename=json_output_filename,
                              verbose=verbose,
                              benchmark_path=BENCH_DIR,
                              factor_fn='default',
                            batch_size=batch_size,
                            do_hoist_stage=True,
                            do_tile_stage=True,
                            print_config=False,
                            tiling_search_algorithm=tile_method,
                                    do_compile=False
                              )
    if store_compile:
        if added_constr:
            program = update_tile_constraints(program, added_constr)

        dir_ext = dir_ext or ''
        print(f"Codelet length: {len(program.codelets)}")

        store_model_outputs(model_name, False,
                      batch_size=1,
                      verbose=verbose,
                      emit_to_stdout=None,
                      use_random=True,
                      dir_ext=f"{dir_ext}",
                      actual_data=False,
                      store_partials=partials,
                            program=program)

    return program



