from codelets.compiler.serialization import serialize_graph
from codelets.examples.genesys import generate_genesys,\
    genesys_instructions, define_genesys
import polymath as pm
from codelets import deserialize_graph, initialize_program, tile, hoist
from collections import namedtuple
import json
from pathlib import Path
import copy
from codelets.compiler import CodeletJSONEncoder

CWD = Path(f"{__file__}").parent
BENCH_DIR = f"{CWD}/input_files"

TestDfgNode = namedtuple('TestDfgNode', ['input_components', 'input_shapes', 'attrs'])
GENESYS_CFG_PATH = f"{CWD}/scratch/genesys_cfg.json"


def parse_cfg():
    with open(GENESYS_CFG_PATH) as f:
        genesys = json.load(f)
    return genesys

def test_genesys_resnet18():
    from pprint import pprint
    graph = pm.pb_load(f"{BENCH_DIR}/resnet18.srdfg")
    genesys = define_genesys("transformation")
    program = initialize_program(graph, genesys)
    program.add_compilation_step("tile", tile)
    program.add_compilation_step("hoist", hoist, dependencies=["tile"])
    program.compile()
    res = program.emit("json_no_ops")
    with open("compiled_resnet18.json", "w") as f:
        json.dump(res, f, cls=CodeletJSONEncoder)

    # print(type(res))


def test_genesys_serialization():
    genesys_cfg = parse_cfg()
    genesys = generate_genesys(genesys_cfg)
    json_genesys = serialize_graph(genesys, f"{CWD}/genesys.json")
    deser_genesys = deserialize_graph(f"{CWD}/genesys.json")
    json_genesys_deser = serialize_graph(deser_genesys, f"{CWD}/deser_genesys.json")
    assert json_genesys_deser == json_genesys

def test_genesys_instr():
    t = genesys_instructions.loop_instr()
    t = genesys_instructions.loop_stride_instr()
    t = genesys_instructions.group_instr()
    t = genesys_instructions.block_instr()

def test_generate_genesys():
    genesys_def = define_genesys()

def test_flex_template():
    genesys = define_genesys()
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