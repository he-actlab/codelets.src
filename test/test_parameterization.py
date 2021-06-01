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


def test_add_transfers():
    with CodeletTemplate("elem_add") as elem_add:
        N = elem_add.dummy_op("N", elem_add.node.inputs[0].shape[0])
        C = elem_add.dummy_op("C", elem_add.node.inputs[0].shape[1])
        H = elem_add.dummy_op("H", elem_add.node.inputs[0].shape[2])
        W = elem_add.dummy_op("W", elem_add.node.inputs[0].shape[3])
        op1 = elem_add.add_input("op1", [N, C, H, W], COMMON_DTYPES[2])
        op2 = elem_add.add_input("op2", [N, C, H, W], COMMON_DTYPES[2])
        elem_add.compute("ADD", [op1, op2])
    genesys = define_genesys(GENESYS_CFG)
    set_targets(elem_add, genesys)
    # There should be only 1 op, ie the compute op
    assert len(elem_add.ops) == 1
    add_transfers(elem_add, genesys)
    # There should be only 5 ops -
    # 2 transfers for op1, 2 transfers for op2 and compute op
    assert len(elem_add.ops) == 5
