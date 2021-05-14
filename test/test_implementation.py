import numpy as np
import pytest

from .util import compare_np_torch
from .torch_codelets import gemm as torch_gemm
from .codelet_implementations import gemm



@pytest.mark.parametrize('M, N, P',[
    (32, 512, 1000)
])
def test_gemm(M, N, P):
    data = np.random.rand(M, N)
    weight = np.random.rand(N, P)
    bias = np.random.rand(P)
    out = np.zeros((M, P))

    inputs = (data, weight, bias)
    outputs = (out,)

    compare_np_torch(gemm, torch_gemm, inputs, outputs)
