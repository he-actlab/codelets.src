from examples.genesys import GENESYS_CFG as cfg
from codelets.common.datatype import COMMON_DTYPES
from codelets.micro_templates import TransferTemplate, ComputeTemplate
from codelets.codelet_template import CodeletTemplate
from codelets.adl.graph import ComputeNode, StorageNode
from codelets.analysis.codelet_pattern import create_pattern

def test_pattern_construction():
    with CodeletTemplate("gemm") as gemm:
        P = gemm.dummy_op("P", gemm.node.inputs[2].shape[0])
        N = gemm.dummy_op("N", gemm.node.inputs[0].shape[1])
        M = gemm.dummy_op("M", gemm.node.inputs[0].shape[0])
        data = gemm.add_input("data", [M, N], COMMON_DTYPES[0])
        weight = gemm.add_input("weight", [N, P], COMMON_DTYPES[0])
        out = gemm.add_output("out", [M, P], COMMON_DTYPES[2])
        zero_const = gemm.constant(0)
        _ = gemm.transfer(zero_const, out)
        with gemm.loop(P) as p:
            with gemm.loop(M) as m:
                with gemm.loop(N) as n:
                    mul_out = gemm.compute("MUL", [data[m, n], weight[n, p]])
                    macc_out = gemm.compute("ADD", [mul_out, out[m, p]])
                _ = gemm.transfer(macc_out, out[m, p])

    create_pattern(gemm)