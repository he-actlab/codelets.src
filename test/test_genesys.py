from codelets.examples.genesys import genesys_instructions, define_genesys, GENESYS_CFG
import polymath as pm
from codelets import initialize_program, tile, hoist, pad_operands
from collections import namedtuple
import json
from pathlib import Path

CWD = Path(f"{__file__}").parent
TEST_DIR = f"{CWD}/input_files"
BENCH_DIR = f"{CWD}/../benchmarks"
MODEL_DIR = f"{BENCH_DIR}/models/srdfg"
LAYER_DIR = f"{BENCH_DIR}/layers/srdfg"

TestDfgNode = namedtuple('TestDfgNode', ['input_components', 'input_shapes', 'attrs'])
GENESYS_CFG_PATH = f"{CWD}/scratch/genesys_cfg.json"


def parse_cfg():
    with open(GENESYS_CFG_PATH) as f:
        genesys = json.load(f)
    return genesys

def test_genesys_add():
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_add.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': []})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("string_final")
    print(res)

def test_genesys_relu():
    from pprint import pprint
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_relu.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': []})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("json_no_ops")
    pprint(res)

def test_genesys_gemm():
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_gemm.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': []})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("string_final")
    print(res)

def test_genesys_conv():
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_conv.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': []})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("string_final")
    print(res)

def test_genesys_resnet18():
    graph = pm.pb_load(f"{MODEL_DIR}/resnet18.srdfg")
    genesys = define_genesys(GENESYS_CFG)
    program = initialize_program(graph, genesys)
    program.add_compilation_step("pad_operands", pad_operands, preproc=True, stage_kwargs={'shaped_nodes': []})
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("string_final")
    print(res)
    # res = program.emit("json_no_ops")
    # pprint(res)


    # with open("compiled_resnet18.json", "w") as f:
    #     json.dump(res, f, cls=CodeletJSONEncoder, indent=2)

def test_genesys_instr():
    t = genesys_instructions.loop_instr()
    t = genesys_instructions.loop_stride_instr()
    t = genesys_instructions.group_instr()
    t = genesys_instructions.block_instr()

def test_generate_genesys():
    genesys_def = define_genesys(GENESYS_CFG)

def test_flex_template():
    genesys = define_genesys(GENESYS_CFG)
    cdlt = genesys.get_codelet_template("conv", is_instance=True)


def test_sympy():
    import sympy
    def split(expr, variables):
        """Split affine, linear, and nonlinear part of expr w.r.t. variables."""
        if isinstance(expr, float):
            return expr, 0, 0

        input_is_list = True
        if not isinstance(variables, list):
            input_is_list = False
            variables = [variables]

        # See <https://github.com/sympy/sympy/issues/11475> on why we need expand() here.
        expr = expr.expand()

        # Get the affine part by removing all terms with any of the variables.
        affine = expr
        for var in variables:
            affine = affine.coeff(var, n=0)

        # Extract the linear coefficients by extracting the affine parts of the derivatives.
        linear = []
        for var in variables:
            d = sympy.diff(expr, var)
            for var2 in variables:
                d = d.coeff(var2, n=0)
            linear.append(d)

        # The rest is nonlinear
        nonlinear = expr - affine
        for var, coeff in zip(variables, linear):
            nonlinear -= var * coeff
        nonlinear = sympy.simplify(nonlinear)

        if not input_is_list:
            assert len(linear) == 1
            linear = linear[0]

        return affine, linear, nonlinear

    i = sympy.Symbol("i")
    j = sympy.Symbol("j")
    k = sympy.Symbol("k")

    expr = 2*i + 3*j + k
    aff, lin, nl = split(expr, [i, j])
    print(aff)
    print(type(aff))
    print(lin)