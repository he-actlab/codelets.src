from pathlib import Path
import os
from codelets.compiler.program import CodeletProgram
from codelets.examples import compile_genesys_layer, compile_genesys, get_arch
from dataclasses import is_dataclass
import polymath as pm
import json
ALL_LAYER_NAMES = ["resnet18_relu", "resnet18_add", "resnet18_conv", "resnet18_conv_bias", "resnet18_gemm", "resnet18_globalaveragepool",
                   "resnet18_train_batchnormalization", "lenet_averagepool", "lenet_conv", "lenet_gemm"]
ALL_MODEL_NAMES = ["resnet18", "resnet50", "lenet"]
ALL_MODEL_TRAIN_NAMES = ["resnet18_train", "resnet50_train", "lenet_train"]

CWD = Path(f"{__file__}").parent
BENCH_DIR = f"{CWD}/input_files"

def get_single_conv2d_node():
    resnet18 = pm.pb_load(f"{BENCH_DIR}/resnet18v1.srdfg")
    for name, node in resnet18.nodes.items():
        if node.name == "conv":
            return node

def create_dirs(fpath, verification):
    cwd = Path(f"{__file__}").parent
    if verification:
        base_path = f"{cwd}/compilation_output/reference_output/{Path(fpath).stem}"
    else:
        base_path = f"{cwd}/compilation_output/{Path(fpath).stem}"

    if not Path(f"{base_path}").exists():
        try:
            os.mkdir(base_path)
        except OSError as e:
            print(f"Creation of directory {base_path} failed:\n {e}")
        else:
            print(f"Successfully created of directory {base_path}")
    else:
        print(f"Directory {base_path} already exists.")
    return base_path

def store_compilation_output(program: CodeletProgram, output_type, extension="txt", verification=False, arch_cfg=None):
    out_path = create_dirs(program.name, verification)
    if output_type == "arch_cfg":
        result = arch_cfg
    else:
        result = program.emit(output_type)
    if not isinstance(result, str):
        assert isinstance(result, dict)
        result = json.dumps(result, indent=2)
    with open(f"{out_path}/{program.name}_{output_type}.{extension}", "w") as outfile:
        outfile.write(result)

def create_reference_outputs(names, batch_size=1, update_cfg_dtypes=False,
                            tiling_path=None,
                            store_tiling=False,
                            store_json_output=False,
                            json_output_filename=None,
                             verbose=False):
    BENCH_DIR = Path(f"{CWD}/../benchmarks").absolute()
    for name in names:
        print(f"Creating compilation output for {name}\n")

        if name in ALL_LAYER_NAMES:
            program = compile_genesys_layer(name,
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
                                    do_tile_stage=True,
                                    print_config=False
                                      )
        elif name in ALL_MODEL_NAMES:
            program = compile_genesys(name,
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
        elif name in ALL_MODEL_TRAIN_NAMES:
            name = name.split("_")[0]
            program = compile_genesys(name,
                                      train=True,
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
            raise RuntimeError(f"Invalid layer name for compilation : {name}")
        arch_cfg = get_arch(None, None, update_cfg_dtypes)

        store_compilation_output(program, "arch_cfg", extension="json", verification=True, arch_cfg=arch_cfg)
        store_compilation_output(program, "operations_idx", extension="txt", verification=True)
        store_compilation_output(program, "json", extension="json", verification=True)
        store_compilation_output(program, "string_final", extension="txt", verification=True)
        store_compilation_output(program, "decimal", extension="txt", verification=True)
        store_compilation_output(program, "binary", extension="txt", verification=True)

def ordered(obj):
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj

def validate_program(program, check_ops=True, check_instr=True, check_decimal=True, check_json=True, check_bin=True,
                     print_difference=False):

    if program.name not in (ALL_MODEL_TRAIN_NAMES + ALL_MODEL_NAMES + ALL_LAYER_NAMES):
        raise RuntimeError(f"Unable to find validation output for program {program.name}")

    cwd = Path(f"{__file__}").parent
    base_path = f"{cwd}/compilation_output/reference_output/{program.name}"

    if check_ops:
        program_op_str = program.emit("operations_idx")
        # Check string_instr
        with open(f"{base_path}/{program.name}_operations_idx.txt", "r") as instr_file:
            ref_op_str = instr_file.read()

        if program_op_str != ref_op_str:
            if print_difference:
                # str_diff = create_diff_map(program_op_str, ref_op_str)
                # pprint.pprint(str_diff)
                print(ref_op_str)
                print()
                print(program_op_str)
            raise RuntimeError(f"Instruction string outputs do not match for program {program.name}.\n"
                               f"Difference:\n"
                               f"Reference: {ref_op_str}\n"
                               f"New: {program_op_str}")
    if check_instr:
        program_instr_str = program.emit("string_final")
        # Check string_instr
        with open(f"{base_path}/{program.name}_string_final.txt", "r") as instr_file:
            ref_instr_str = instr_file.read()


        if program_instr_str != ref_instr_str:
            str_diff = create_diff_map(program_instr_str, ref_instr_str)
            print(ref_instr_str)
            print()
            print(program_instr_str)
            raise RuntimeError(f"Instruction string outputs do not match for program {program.name}.\n"
                               f"Difference:\n"
                               f"Reference: {ref_instr_str}\n"
                               f"New: {program_instr_str}")

    if check_decimal:
        program_dec_str = program.emit("decimal")
        # Check decimal
        with open(f"{base_path}/{program.name}_decimal.txt", "r") as dec_file:
            ref_dec_str = dec_file.read()

        if program_dec_str != ref_dec_str:
            raise RuntimeError(f"Decimal string outputs do not match for program {program.name}")


    if check_bin:
        # Check bin
        program_bin_str = program.emit("binary")

        with open(f"{base_path}/{program.name}_binary.txt", "r") as bin_file:
            ref_bin_str = bin_file.read()


        if program_bin_str != ref_bin_str:
            raise RuntimeError(f"Binary string outputs do not match for program {program.name}")


def compare_dataclasses(ref_obj, test_obj, skip_fields=None):
    skip_fields = skip_fields or []
    for k in ref_obj.__dataclass_fields__.keys():
        if k not in skip_fields:
            ref_field = getattr(ref_obj, k)
            test_field = getattr(test_obj, k)
            assert type(ref_field) == type(test_field), f"Field {k} do not match:\n" \
                                                        f"Reference field: {ref_field}\n" \
                                                        f"Test field: {test_field}"
            if is_dataclass(ref_field):
                compare_dataclasses(ref_field, test_field, skip_fields=skip_fields)
            elif isinstance(ref_field, (list, tuple)) and len(ref_field) > 0 and is_dataclass(ref_field[0]):
                assert len(ref_field) == len(test_field)
                for idx in range(len(ref_field)):
                    compare_dataclasses(ref_field[idx], test_field[idx], skip_fields=skip_fields)
            else:
                assert ref_field == test_field, f"Field {k} do not match:\n" \
                                                f"Reference field: {ref_field}\n" \
                                                f"Test field: {test_field}"

def create_diff_map(a, b):
    a_list = a.split("\n")
    b_list = b.split("\n")
    diff_map = {}
    if len(a_list) > len(b_list):
        for i, str_b in enumerate(b_list):
            if str_b != a_list[i]:
                diff_map[i] = [str_b, a_list[i]]
    else:
        for i, str_a in enumerate(a_list):
            if str_a != b_list[i]:
                diff_map[i] = [str_a, b_list[i]]
    return diff_map

# def compare_np_torch(np_fn, torch_fn, inputs, outputs):
#
#     np_fn(*(inputs + outputs))
#     convert_func = lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
#     torch_inputs = tuple([convert_func(i) for i in inputs])
#     torch_outputs = tuple([convert_func(o) for o in outputs])
#     torch_fn(*(torch_inputs + torch_outputs))
#
#     for idx, o in enumerate(outputs):
#         np.testing.assert_allclose(o, torch_outputs[idx].numpy())


