from pathlib import Path
from examples.genesys import compile_genesys, get_arch
from tools.compile_layer import store_compilation_output
from codelets.compiler.program import CodeletProgram
import os
BENCH_DIR = Path(f"{Path(__file__).parent}/../benchmarks")
MODEL_DIR = Path(f"{Path(__file__).parent}/../benchmarks/models")

def generate_inputs(inputs):
    pass

def generate_values(cdlt, layer_name, inputs=None):
    pass

def store_model_values(program: CodeletProgram, base_path, use_random=True, format="nhwc"):
    operand_map = {}
    for c in program.codelets:
        print(c.op_name)

def store_model(model_name,
                  batch_size=1,
                  verbose=False,
                  emit_to_stdout=None,
                  load_path=None,
                  dir_ext=None,
                  actual_data=False,
                  use_random=False,
                  store_partials=False,
                  program=None, added_constr=None):
    tiling_path = None
    store_tiling = False
    store_json_output = False
    json_output_filename = None
    update_cfg_dtypes = False
    print(f"Creating compilation output for {model_name}\n")
    if program is None:
        model_files = [d.name.split(".")[0] for d in os.scandir(f"{BENCH_DIR}/models/srdfg")]
        if model_name in model_files:
            program = compile_genesys(model_name,
                                      train=False,
                                      update_cfg_dtypes=update_cfg_dtypes,
                                      tiling_path=tiling_path,
                                      batch_size=batch_size,
                                      store_tiling=store_tiling,
                                      store_json_output=store_json_output,
                                      json_output_filename=json_output_filename,
                                      verbose=verbose,
                                      benchmark_path=BENCH_DIR,
                                      factor_fn='default',
                                      print_config=False
                                      )
        else:
            raise RuntimeError(f"Cannot find model {model_name} in directory")

    program.compile(verbose=False, finalize_instructions=True)
    arch_cfg = get_arch(None, None, update_cfg_dtypes)

    base_path = store_compilation_output(program, "arch_cfg", extension="json", arch_cfg=arch_cfg, dir_ext=dir_ext)
    store_compilation_output(program, "operations_idx", extension="txt", dir_ext=dir_ext)
    store_compilation_output(program, "json", extension="json", dir_ext=dir_ext)
    store_compilation_output(program, "string_final", extension="txt", dir_ext=dir_ext)
    store_compilation_output(program, "decimal", extension="txt", dir_ext=dir_ext)
    store_compilation_output(program, "binary", extension="txt", dir_ext=dir_ext)
    store_model_values(program, base_path, use_random=use_random)
    return program

def main(model_name):
    batch_size = 1
    tile_method = "min_tiles"

    update_cfg_dtypes = False
    tiling_path = None
    store_tiling = False
    store_json_output = False
    json_output_filename = None
    program = compile_genesys(model_name,
                              update_cfg_dtypes=update_cfg_dtypes,
                              tiling_path=tiling_path,
                              store_tiling=store_tiling,
                              store_json_output=store_json_output,
                              json_output_filename=json_output_filename,
                              verbose=False,
                              benchmark_path=BENCH_DIR,
                              factor_fn='default',
                            batch_size=batch_size,
                            print_config=False,
                                    do_compile=False
                                    # relocation_offsets=reloc_offsets
                              )

    store_model_values(program, )


if __name__ == "__main__":
    onnx_file = ""

