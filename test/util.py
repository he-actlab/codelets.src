from pathlib import Path
import os
from codelets.compiler.program import CodeletProgram
import polymath as pm
import json

CWD = Path(f"{__file__}").parent
BENCH_DIR = f"{CWD}/input_files"

def get_single_conv2d_node():
    resnet18 = pm.pb_load(f"{BENCH_DIR}/resnet18v1.srdfg")
    for name, node in resnet18.nodes.items():
        if node.name == "conv":
            return node

def create_dirs(fpath):
    cwd = Path(f"{__file__}").parent
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

def store_compilation_output(program: CodeletProgram, output_type, extension="txt"):
    out_path = create_dirs(program.name)
    result = program.emit(output_type)
    if not isinstance(result, str):
        assert isinstance(result, dict)
        result = json.dumps(result, indent=2)
    with open(f"{out_path}/{program.name}_{output_type}.{extension}", "w") as outfile:
        outfile.write(result)
