import pytest
from codelets.common.datatype import COMMON_DTYPES
from codelets.codelet_template import CodeletTemplate
from codelets.micro_templates.analysis.parameterization import set_targets, add_transfers
from examples.genesys import GENESYS_CFG, define_genesys

def test_set_targets():
    with CodeletTemplate("elem_add") as elem_add:
        N = elem_add.dummy_op("N", elem_add.node.inputs[0].shape[0])
        C = elem_add.dummy_op("C", elem_add.node.inputs[0].shape[1])
        H = elem_add.dummy_op("H", elem_add.node.inputs[0].shape[2])
        W = elem_add.dummy_op("W", elem_add.node.inputs[0].shape[3])
        op1 = elem_add.add_input("op1", [N, C, H, W], COMMON_DTYPES[2])
        op2 = elem_add.add_input("op2", [N, C, H, W], COMMON_DTYPES[2])
        elem_add.compute("ADD", [op1, op2])
    genesys = define_genesys(GENESYS_CFG)
    assert elem_add.op_map['compute0'].is_target_set() is False
    set_targets(elem_add, genesys)
    assert elem_add.op_map['compute0'].is_target_set() and elem_add.op_map['compute0'].target == 'SIMD'


def test_add_transfers_basic():
    with CodeletTemplate("elem_add") as elem_add:
        N = elem_add.dummy_op("N", elem_add.node.inputs[0].shape[0])
        C = elem_add.dummy_op("C", elem_add.node.inputs[0].shape[1])
        H = elem_add.dummy_op("H", elem_add.node.inputs[0].shape[2])
        W = elem_add.dummy_op("W", elem_add.node.inputs[0].shape[3])
        op1 = elem_add.add_input("op1", [N, C, H, W], COMMON_DTYPES[2])
        op2 = elem_add.add_input("op2", [N, C, H, W], COMMON_DTYPES[2])
        out = elem_add.add_output("out", [N, C, H, W], COMMON_DTYPES[2])
        add_result = elem_add.compute("ADD", [op1, op2])
        elem_add.transfer(add_result, out)
    genesys = define_genesys(GENESYS_CFG)
    set_targets(elem_add, genesys)
    # There should be only 2 op, ie the compute op and final transfer
    assert len(elem_add.ops) == 2
    add_transfers(elem_add, genesys)
    # There should be only 7 ops -
    # 2 transfers for op1, 2 transfers for op2, compute op, and 2 transfers back
    assert len(elem_add.ops) == 7


def test_add_transfers_tanh():
    with CodeletTemplate("elem_tanh") as tanh:
        N = tanh.dummy_op("N", tanh.node.inputs[0].shape[0])
        C = tanh.dummy_op("C", tanh.node.inputs[0].shape[1])
        H = tanh.dummy_op("H", tanh.node.inputs[0].shape[2])
        W = tanh.dummy_op("W", tanh.node.inputs[0].shape[3])
        op1 = tanh.add_input("op1", [N, C, H, W], COMMON_DTYPES[2])
        op2 = tanh.add_input("op2", [N, C, H, W], COMMON_DTYPES[2])
        out = tanh.add_output("out", [N, C, H, W], COMMON_DTYPES[2])

        with tanh.loop(N) as n:
            with tanh.loop(C) as c:
                with tanh.loop(H) as h:
                    with tanh.loop(W) as w:
                        compute_out = tanh.compute("TANH", [op1[n, c, h, w], op2[n, c, h, w]])
        _ = tanh.transfer(compute_out, out)

    genesys = define_genesys(GENESYS_CFG)
    assert (len(tanh.ops)) == 6
    set_targets(tanh, genesys)
    add_transfers(tanh, genesys)
    assert (len(tanh.ops)) == 11
