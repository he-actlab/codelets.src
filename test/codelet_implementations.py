import numpy as np

def gemm(data, weight, bias, out):
    out[:, :] = data.dot(weight) + bias


