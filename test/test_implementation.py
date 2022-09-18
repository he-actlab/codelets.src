import numpy as np
import pytest
from collections import OrderedDict
from .torch_codelets import gemm as torch_gemm
import torch
from torch import nn
from .codelet_implementations import gemm


def conv(data, weights, bias, stride, pad):
    N, IC, IH, IW = data.shape
    OC, _, KH, KW = weights.shape
    OH = (IH + 2*pad - (KH - 1) - 1) // stride + 1
    OW = (IW + 2*pad - (KW - 1) - 1) // stride + 1
    assert int(OH) - OH == 0
    assert int(OW) - OW == 0
    OH = int(OH)
    OW = int(OW)
    out = np.zeros((N, OC, OH, OW), dtype=data.dtype)
    data = np.pad(data, ((0, 0), (0,0), (pad, pad), (pad, pad)), 'constant')
    s = stride
    for oc in range(OC):
        for n in range(N):
            for ic in range(IC):
                for kh in range(KH):
                    for kw in range(KW):
                        for oh in range(OH):
                            for ow in range(OW):
                                macres = data[n, ic, oh * s + kh, ow * s + kw]*weights[oc, ic, kh, kw]
                                out[n, oc, oh, ow] += macres

    return out

def max_pool(data, kernel_size, stride, pad):
    KH, KW = kernel_size, kernel_size
    N, C, IH, IW = data.shape
    OH = (IH + 2*pad - (KH - 1) - 1) // stride + 1
    OW = (IW + 2*pad - (KW - 1) - 1) // stride + 1
    # out = np.full((N, C, OH, OW), -np.Inf, dtype=data.dtype)
    out = np.zeros((N, C, OH, OW), dtype=data.dtype)
    data = np.pad(data, ((0, 0),(0,0), (pad, pad), (pad, pad)), mode='constant')
    s = stride
    print(f"np Pool stride: {s}, pad: {pad}, kernel size: {kernel_size}")
    for n in range(N):
        for c in range(C):
            for oh in range(OH):
                for ow in range(OW):
                    out[n, c, oh, ow] = -np.Inf
                    for kh in range(KH):
                        for kw in range(KW):
                            out[n, c, oh, ow] = max(data[n, c, oh*s + kh, ow*s + kw], out[n, c, oh, ow])
    return out


def np_impl(data, weights, bias, stride, pad, max_pool_params):
    conv_out = conv(data, weights, bias, stride, pad)
    out = max_pool(conv_out, max_pool_params['kernel_size'], max_pool_params['stride'], max_pool_params['pad'])
    return out

def pytorch_impl(data, weights, bias, stride, pad, max_pool_params):
    n, ic, ih, iw = data.shape
    oc, _, kh, kw = weights.shape
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(ic, oc, kh, stride=stride, padding=pad, bias=False)),
        ('maxpool1', nn.MaxPool2d(max_pool_params['kernel_size'],
                                  stride=max_pool_params['stride'],
                                  padding=max_pool_params['pad'])),

    ]))
    model.eval()
    model.conv1.weight = nn.Parameter(weights)

    # out1 = model(data)
    out1 = model.conv1(data)
    return out1



def pytorch_impl_fused(data, weights, bias, stride, pad, max_pool_params):
    n, ic, ih, iw = data.shape
    oc, _, kh, kw = weights.shape
    mp_stride, mp_k, mp_pad = max_pool_params['stride'], max_pool_params['kernel_size'], max_pool_params['pad']
    ih_num = (ih + 2*pad - (kh - 1) - 1)
    iw_num = (iw + 2*pad - (kw - 1) - 1)
    ih1 = ih_num // stride + 1
    iw1 = iw_num // stride + 1

    oh1 = (ih1 + 2 * mp_pad - (mp_k - 1) - 1) // mp_stride + 1
    ow1 = (iw1 + 2 * mp_pad - (mp_k - 1) - 1) // mp_stride + 1
    ih_ = ((ih1 + 2*mp_pad) - 1)*stride + kh
    iw_ =  ((iw1 + 2*mp_pad) - 1)*stride + kw
    if ih_num % stride != 0:
        ih_ += 1
        iw_ += 1

    pad = (ih_ - ih) // 2
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(ic, oc, kh, stride=stride, padding=pad, bias=False)),
        ('maxpool1', nn.MaxPool2d(mp_k,
                                  stride=mp_stride,
                                  padding=0)),

    ]))
    model.eval()
    model.conv1.weight = nn.Parameter(weights)
    out1 = model.conv1(data)
    # out1 = model(data)
    return out1

@pytest.mark.parametrize('conv_params, max_pool_params',[
    ({"N": 1, "IC": 3, "IHW": 224, "OHW": 112, "KHW": 7, "OC": 64, "pad": 3, "stride": 2},
     {"kernel_size": 3, "pad": 1, "stride": 2}
     ),
    # ({"N": 1, "IC": 3, "IHW": 112, "OHW": 56, "KHW": 7, "OC": 32, "pad": 3, "stride": 2},
    #  {"kernel_size": 3, "pad": 1, "stride": 2}
    #  ),
])
def test_max_pool_conv_fusion(conv_params, max_pool_params):
    N, IC, IHW, OC, KHW = conv_params['N'], conv_params['IC'], conv_params['IHW'], conv_params['OC'], conv_params['KHW']
    stride, pad = conv_params['stride'], conv_params['pad']

    data = torch.randn((N, IC, IHW, IHW))
    weights = torch.randn((OC, IC, KHW, KHW))
    bias = torch.randn((OC,))

    # np_out = np_impl(data.numpy(), weights.numpy(), bias.numpy(), stride, pad, max_pool_params)
    # torch_out_orig = pytorch_impl(torch.clone(data), torch.clone(weights), torch.clone(bias), stride, pad, max_pool_params)
    torch_out_fused = pytorch_impl_fused(data, weights, bias, stride, pad, max_pool_params)
    print(torch_out_fused.shape)
    print(torch_out_fused[0, 0, 0, 0])
    print(torch_out_fused[0, 0, 113, 113])
    # torch.testing.assert_allclose(torch_out_fused, torch_out_orig)

def pytorch_impl_dw(data, w1, s1, p1, w2, s2, p2):
    n, ic1, ih1, iw1 = data.shape
    oc1, _, kh1, kw1 = w1.shape


    oc2, _, kh2, kw2 = w2.shape
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(ic1, oc1, kh1, stride=s1, padding=p1, bias=False)),
        ('dwconv1', nn.Conv2d(oc2, oc2, kh2, stride=s2, padding=p2, bias=False, groups=oc2)),

    ]))
    model.eval()
    model.conv1.weight = nn.Parameter(w1)
    model.dwconv1.weight = nn.Parameter(w2)
    out1 = model(data)
    return out1

def pytorch_impl_dw_fused(data, w1, s1, p1, w2, s2, p2):
    n, ic1, ih1, iw1 = data.shape
    oc1, _, kh1, kw1 = w1.shape

    oh1_num = (ih1 + 2*p1 - kh1)
    ow1_num = (iw1 + 2*p1 - kw1)

    oh1 = oh1_num // s1 + 1
    ow1 = ow1_num // s1 + 1

    ih1_ = ((oh1 + 2*p2) - 1)*s1 + kh1
    iw1_ = ((ow1 + 2*p2) - 1)*s1 + kw1
    if oh1_num % s1 != 0:
        ih1_ += 1
        iw1_ += 1

    p1 = (ih1_ - ih1) // 2

    oc2, _, kh2, kw2 = w2.shape
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(ic1, oc1, kh1, stride=s1, padding=p1, bias=False)),
        ('dwconv1', nn.Conv2d(oc2, oc2, kh2, stride=s2, padding=0, bias=False, groups=oc2)),

    ]))
    model.eval()
    model.conv1.weight = nn.Parameter(w1)
    model.dwconv1.weight = nn.Parameter(w2)
    out1 = model(data)

    return out1

@pytest.mark.parametrize('conv_params, dw_conv_params', [
    ({"N": 1, "IC": 32, "IHW": 56, "OHW": 56, "KHW": 1, "OC": 192, "pad": 0, "stride": 1},
     {"N": 1, "IC": 192, "IHW": 56, "OHW": 56, "KHW": 3, "OC": 192, "pad": 1, "stride": 1}
     ),

])
def test_conv_dw_conv_fusion(conv_params, dw_conv_params):
    N, IC1, IHW1, OC1, KHW1 = conv_params['N'], conv_params['IC'], conv_params['IHW'], conv_params['OC'], conv_params['KHW']
    s1, p1 = conv_params['stride'], conv_params['pad']

    IC2, IHW2, OC2, KHW2 = dw_conv_params['IC'], dw_conv_params['IHW'], dw_conv_params['OC'], dw_conv_params['KHW']
    s2, p2 = dw_conv_params['stride'], dw_conv_params['pad']

    data = torch.randn((N, IC1, IHW1, IHW1))
    w1 = torch.randn((OC1, IC1, KHW1, KHW1))
    w2 = torch.randn((OC2, 1, KHW2, KHW2))
    out1 = pytorch_impl_dw(data.clone(), w1.clone(), s1, p1, w2.clone(), s2, p2)
    out2 = pytorch_impl_dw_fused(data.clone(), w1, s1, p1, w2, s2, p2)

    torch.testing.assert_allclose(out1, out2)





