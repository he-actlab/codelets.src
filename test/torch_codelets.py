import torch.nn.functional as F

def gemm(data, weight, bias, out):
    out[:, :] = F.linear(data, weight.T, bias=bias)

