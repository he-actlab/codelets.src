import polymath as pm
from codelets.adl.operation import Operand
from codelets.templates.codelet_template import CodeletTemplate
from examples.genesys import OP_DTYPES, define_genesys, GENESYS_CFG
from examples.genesys.genesys_inference_codelets import relu, averagepool2d, gemm
from examples.genesys.genesys_codelets import averagepool2d as avgpool_template,\
    gemm as gemm_template
from pathlib import Path

from .util import compare_dataclasses

CWD = Path(f"{__file__}").parent
BENCH_DIR = Path(f"{CWD}/../benchmarks").absolute()
LAYER_DIR = f"{BENCH_DIR}/layers/srdfg"


def test_shape_dummy_op():
    with pm.Node(name="test_transpose") as graph:
        inp = pm.input(name="data", shape=(3,4,5,6))
        out = pm.output(name="out", shape=(4,3,5,6))
        test_op = pm.tensor_transpose(inp, out, (1,0,2,3))

    with CodeletTemplate("relu") as cdlt:
        op1_shape = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])

    instance_args = {"NodePlaceholder": test_op}
    result = op1_shape.evaluate(instance_args)
    assert result == test_op.inputs[0].shape[0]

def test_dummy_op_arithmetic():
    with pm.Node(name="test_transpose") as graph:
        inp = pm.input(name="data", shape=(3,4,5,6))
        out = pm.output(name="out", shape=(4,3,5,6))
        test_op = pm.tensor_transpose(inp, out, (1,0,2,3))

    with CodeletTemplate("relu") as cdlt:
        denom = cdlt.node.inputs[0].shape[0]*cdlt.node.inputs[0].shape[1]
        int_mul = cdlt.node.inputs[0].shape[0]*3
        div_val = cdlt.node.inputs[0].shape[3] // cdlt.node.inputs[0].shape[0]
        sub_val = cdlt.node.inputs[0].shape[3] - cdlt.node.inputs[0].shape[0]
        int_sub_val1 = cdlt.node.inputs[0].shape[3] - 4
        int_sub_val2 = 20 - cdlt.node.inputs[0].shape[3]

    instance_args = {"NodePlaceholder": test_op}
    res1 = denom.evaluate(instance_args)
    assert res1 == (test_op.inputs[0].shape[0] * test_op.inputs[0].shape[1])

    res2 = int_mul.evaluate(instance_args)

    assert res2 == (test_op.inputs[0].shape[0] * 3)

    res3 = div_val.evaluate(instance_args)
    assert res3 == (test_op.inputs[0].shape[3] // test_op.inputs[0].shape[0])

    res4 = sub_val.evaluate(instance_args)
    assert res4 == (test_op.inputs[0].shape[3] - test_op.inputs[0].shape[0])

    res5 = int_sub_val1.evaluate(instance_args)
    assert res5 == (test_op.inputs[0].shape[3] - 4)

    res6 = int_sub_val2.evaluate(instance_args)
    assert res6 == (20 - test_op.inputs[0].shape[3])

def test_dummy_param():

    with pm.Node(name="test_transpose") as graph:
        inp = pm.input(name="data", shape=(3,4,5,6))
        out_node = pm.output(name="out", shape=(4,3,5,6))
        temp_op = pm.tensor_transpose(inp, out_node, (1,0,2,3))

    with CodeletTemplate("check_perm") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])
        out_shape = cdlt.dummy_param("out_shape", "[shape[perm[i]] for i in range(len(shape))]", ["perm", "shape"], (cdlt.node.perm, cdlt.node.inputs[0].shape))


    test_result = out_shape.evaluate({"NodePlaceholder": temp_op})

    assert test_result == list(out_node.shape)

def test_dummy_offset():
    with pm.Node(name="test_transpose") as graph:
        inp = pm.input(name="data", shape=(3,4,5,6))
        out_node = pm.output(name="out", shape=(4,3,5,6))
        temp_op = pm.tensor_transpose(inp, out_node, (1,0,2,3))

    with CodeletTemplate("check_perm") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])
        out_shape = cdlt.dummy_param("out_shape",
                                     "[shape[perm[i]] for i in range(len(shape))]", ["perm", "shape"],
                                     (cdlt.node.perm, cdlt.node.inputs[0].shape))
        out = cdlt.create_operand_template("out", OP_DTYPES, out_shape, default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(op1[n,c,h,w], ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM1")
                        cdlt.compute("MOVE", [op1], [out], target="SIMD")
                        offset_fn = f"[n,c,h,w][perm"
                        fn_args_str = ["perm", "n", "c", "h", "w"]
                        fn_args = (cdlt.node.perm, n, c, h, w)
                        offs = tuple([cdlt.dummy_param(f"off{i}", f"{offset_fn}[{i}]]", fn_args_str, fn_args) for i in range(4)])

                        cdlt.transfer(out[offs], ["VMEM1", "DRAM"])

    instance_args = {"NodePlaceholder": temp_op, 'HAGPlaceholder': None}
    cdlt_res = cdlt.instantiate(instance_args)
    instance_args['CodeletTemplate'] = cdlt_res
    out_offset_res = tuple([o.evaluate(instance_args).op_str for o in offs])

    assert out_offset_res == (c.op_str, n.op_str, h.op_str, w.op_str)


def test_dummy_implementation():


    with CodeletTemplate("check_perm") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])
        out_shape = cdlt.dummy_param("out_shape",
                                     "[shape[perm[i]] for i in range(len(shape))]", ["perm", "shape"],
                                     (cdlt.node.perm, cdlt.node.inputs[0].shape))
        out = cdlt.create_operand_template("out", OP_DTYPES, out_shape, default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(op1[n,c,h,w], ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM1")
                        cdlt.compute("MOVE", [op1], [out], target="SIMD")
                        out_offset = cdlt.dummy_param("out_offset",
                                                      "tuple([[n,c,h,w][perm[i]] for i in range(len(perm))])",
                                                      ["perm", "n", "c", "h", "w"],
                                                      (cdlt.node.perm, n, c, h, w))
                        cdlt.transfer(out[out_offset], ["VMEM1", "DRAM"])

def test_codelet_instantiation():
    with CodeletTemplate("relu") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])

        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])
        cdlt.configure("start", "SIMD")
        with cdlt.loop(N) as n:
            with cdlt.loop(C) as c:
                with cdlt.loop(H) as h:
                    with cdlt.loop(W) as w:
                        cdlt.transfer(op1[n, c, h, w], ["DRAM", "VMEM1"])
                        out.set_write_destination("VMEM1")
                        cdlt.compute("RELU", [op1], [out], target="SIMD")
                        cdlt.transfer(out[n, c, h, w], ["VMEM1", "DRAM"])

    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_relu.srdfg")
    target_node = None
    for name, node in graph.nodes.items():
        if node.op_name == "relu":
            target_node = node
            break
    assert target_node is not None
    hag = define_genesys(GENESYS_CFG)
    reference_codelt = relu(hag)
    new_cdlt = cdlt.instantiate({"HAGPlaceholder": hag, "NodePlaceholder": target_node})
    print(new_cdlt.emit("operations_idx"))
    print(reference_codelt.emit("operations_idx"))

def test_operand_instantiation():
    with CodeletTemplate("relu_template") as cdlt:
        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])
        op1 = cdlt.create_operand_template("op1test", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1])

        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_outputs([out])
    hag = define_genesys(GENESYS_CFG)
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_relu.srdfg")
    target_node = None
    for name, node in graph.nodes.items():
        if node.op_name == "relu":
            target_node = node
            break
    assert target_node is not None

    instance_args = {"HAGPlaceholder": hag, "NodePlaceholder": target_node}
    test_op1 = [i.instantiate(instance_args) for i in cdlt.inputs][0]

    ref_op1 = Operand("op1", OP_DTYPES, ["N", "C", "H", "W"], dtype=OP_DTYPES[2])

    for k in ref_op1.__dataclass_fields__.keys():
        assert getattr(ref_op1, k) == getattr(test_op1, k) or k == "name"


def test_loop_offset():
    hag = define_genesys(GENESYS_CFG)

    ref_impl = averagepool2d(hag)

    template_impl = avgpool_template(hag)

    #
    graph = pm.pb_load(f"{LAYER_DIR}/lenet_averagepool.srdfg")
    target_node = None
    #
    for name, node in graph.nodes.items():
        if node.op_name == "avg_pool":
            target_node = node
            break
    assert target_node is not None
    #
    instance_args = {"HAGPlaceholder": hag, "NodePlaceholder": target_node}
    new_cdlt = template_impl.instantiate(instance_args)
    offset_transfer = new_cdlt.op_map['transfer0']
    ref_transfer = ref_impl.op_map['transfer0']


    ref_op1 = ref_transfer.operand
    test_op1 = offset_transfer.operand

    compare_dataclasses(ref_op1, test_op1, skip_fields=["name", "operand_name"])

    ref_op2 = ref_impl.temps[0]
    test_op2 = new_cdlt.temps[0]

    compare_dataclasses(ref_op2, test_op2, skip_fields=["name", "operand_name", "supported_dtypes"])

def test_op_levels():
    hag = define_genesys(GENESYS_CFG)

    ref_impl = gemm(hag)

    template_impl = gemm_template(hag)

    #
    graph = pm.pb_load(f"{LAYER_DIR}/resnet18_gemm.srdfg")
    target_node = None
    #
    for name, node in graph.nodes.items():
        if node.op_name == "gemm":
            target_node = node
            break
    assert target_node is not None
    #
    instance_args = {"HAGPlaceholder": hag, "NodePlaceholder": target_node}
    new_cdlt = template_impl.instantiate(instance_args)
    ref_emit_str = ref_impl.emit("operations_idx").split("\n")
    template_emit_str = new_cdlt.emit("operations_idx").split("\n")
    for idx in range(len(ref_emit_str)):
        if idx == 0:
            continue
        ref_line = ref_emit_str[idx]
        template_line = template_emit_str[idx]
        assert ref_line == template_line

def test_hag_storage_levels():
    hag = define_genesys(GENESYS_CFG)

