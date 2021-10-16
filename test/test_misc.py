import numpy as np
import pytest
WEIGHTS_CL_TO_CF = [3, 2, 0, 1] # (KH, KW, IC, OC) -> (OC, IC, KH, KW)
WEIGHTS_CF_TO_CL = [2, 3, 1, 0] # (OC, IC, KH, KW) -> (KH, KW, IC, OC)
ACT_CL_TO_CF = [0, 3, 1, 2] # (N, H, W, C) -> (N, C, H, W)
ACT_CF_TO_CL = [0, 2, 3, 1] # (N, C, H, W) -> (N, H, W, C)
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = np.int32((H + 2 * padding - field_height) / stride + 1)
    out_width = np.int32((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def conv_forward_im2col(x, w, b, conv_param):
    """
    A fast implementation of the forward pass for a convolutional layer
    based on im2col and col2im.
    """
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions

    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    # Create output
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    # x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache



@pytest.mark.parametrize('N,IC,OC,OHW,KHW,IHW,S,P',[
    (1,4,8,16,3,33,2,0)
])
def test_conv_as_mvmul_ws(N,IC,OC,OHW,KHW,IHW,S,P):
    import numpy as np
    x = np.random.randint(0, 100, (N, IC,IHW,IHW))
    w = np.random.randint(0, 100, (OC,IC,KHW,KHW))
    b = np.random.randint(0, 100, (OC,))
    y = np.zeros((N, OC,OHW, OHW))
    ref, _ = conv_forward_im2col(x, w, b, {"stride":S,"pad":P})
    x = x.transpose(*tuple(ACT_CF_TO_CL))
    w = w.transpose(*tuple(WEIGHTS_CF_TO_CL))
    y = y.transpose(*tuple(ACT_CF_TO_CL))
    sa_ic = 4
    sa_oc = 4
    n_step = 1
    khw_step = 1
    ohw_step = 1
    # m_step = 1
    # p_step = 8
    # p_iters = 0
    # n_iters = 0
    # m_iters = 0
    for oc in range(OC):
        y[:,:,:,oc] = b[oc]
    for oc1 in range(0,OC,sa_oc):
        bt = b[oc1:oc1 + sa_oc]
        for n1 in range(0, N, n_step):
            for ic1 in range(0, IC, sa_ic):
                for kh1 in range(0, KHW, khw_step):
                    for kw1 in range(0, KHW, khw_step):
                        wt = w[kh1:kh1 + khw_step,kw1:kw1 + khw_step, ic1:ic1+sa_ic, oc1:oc1+sa_oc]
                        for oh1 in range(0, OHW, ohw_step):
                            for ow1 in range(0, OHW, ohw_step):
                                partial_out = y[n1:n1+n_step, oh1:oh1+ohw_step,ow1:ow1+ohw_step, oc1:oc1+sa_oc]
                                xt = x[n1:n1+n_step,(kh1 + oh1*S), (kw1 + ow1*S), ic1:ic1+sa_ic]
                                partial_out += xt.dot(wt)
                                y[n1:n1 + n_step, oh1:oh1 + ohw_step, ow1:ow1 + ohw_step, oc1:oc1 + sa_oc] = partial_out.copy()


    np.testing.assert_allclose(y.transpose(*tuple(ACT_CL_TO_CF)), ref)



@pytest.mark.parametrize('M, N, P',[
    (32, 128, 64)
])
def test_matmul_as_mvmul(M, N, P):
    import numpy as np
    x = np.random.randint(0, 100, (M, N))
    w = np.random.randint(0, 100, (N, P))
    b = np.random.randint(0, 100, (P,))
    y = np.zeros((M, P))
    n_step = 8
    m_step = 1
    p_step = 8
    p_iters = 0
    n_iters = 0
    m_iters = 0
    for m in range(0, M, m_step):
        m_iters += 1
        for p in range(0, P, p_step):
            p_iters += 1
            partial_out = np.zeros((m_step, p_step))
            for n in range(0, N, n_step):
                n_iters +=1
                partial_out += np.dot(x[m:m+m_step, n:n+n_step], w[n:n+n_step, p:p+p_step])
            y[m:m+m_step, p:p+p_step] = partial_out.copy() + b[p:p+p_step]
    np.testing.assert_allclose(y, np.dot(x, w) + b)
    print(f"P iters: {p_iters}, M iters: {m_iters}, N iters: {n_iters}")
    y = np.zeros((M, P))
    p_iters = 0
    n_iters = 0
    m_iters = 0
    for m in range(M):
        y[m] = b.copy()
    for p in range(0, P, p_step):
        p_iters += 1
        for n in range(0, N, n_step):
            n_iters += 1
            for m in range(0, M, m_step):
                m_iters += 1
                y[m:m+m_step, p:p+p_step] += np.dot(x[m:m+m_step, n:n+n_step], w[n:n+n_step, p:p+p_step])

    np.testing.assert_allclose(y, np.dot(x, w) + b)
    print(f"P iters: {p_iters}, M iters: {m_iters}, N iters: {n_iters}")

@pytest.mark.parametrize('N,IC,OC,OHW,KHW,IHW,S,P',[
    (1,4,8,16,3,33,2,0)
])
def test_conv_as_mvmul_os(N,IC,OC,OHW,KHW,IHW,S,P):
    import numpy as np
    x = np.random.randint(0, 100, (N, IC,IHW,IHW))
    w = np.random.randint(0, 100, (OC,IC,KHW,KHW))
    b = np.random.randint(0, 100, (OC,))
    y = np.zeros((N, OC,OHW, OHW))
    ref, _ = conv_forward_im2col(x, w, b, {"stride":S,"pad":P})
    x = x.transpose(*tuple(ACT_CF_TO_CL))
    w = w.transpose(*tuple(WEIGHTS_CF_TO_CL))
    y = y.transpose(*tuple(ACT_CF_TO_CL))
    sa_ic = 4
    sa_oc = 4
    n_step = 1
    khw_step = KHW
    ohw_step = 1
    # m_step = 1
    # p_step = 8
    # p_iters = 0
    # n_iters = 0
    # m_iters = 0

    for oc1 in range(0,OC,sa_oc):
        bt = b[oc1:oc1 + sa_oc]
        for n1 in range(0, N, n_step):
            for oh1 in range(0, OHW, ohw_step):
                for ow1 in range(0, OHW, ohw_step):
                    partial_out = y[n1:n1 + n_step, oh1:oh1 + ohw_step, ow1:ow1 + ohw_step, oc1:oc1 + sa_oc]

                    for kh1 in range(0, KHW, khw_step):
                        for kw1 in range(0, KHW, khw_step):
                            for ic1 in range(0, IC, sa_ic):

                                wt = w[kh1:kh1 + khw_step,kw1:kw1 + khw_step, ic1:ic1+sa_ic, oc1:oc1+sa_oc].reshape(-1,sa_oc)

                                xt = x[n1:n1+n_step,
                                     (kh1 + oh1*S):(kh1 + oh1*S) + khw_step,
                                     (kw1 + ow1*S):(kw1 + ow1*S) + khw_step,
                                     ic1:ic1+sa_ic].reshape(n_step,-1)
                                partial_out += xt.dot(wt)
                    y[n1:n1 + n_step, oh1:oh1 + ohw_step, ow1:ow1 + ohw_step, oc1:oc1 + sa_oc] = partial_out.copy() + bt


    np.testing.assert_allclose(y.transpose(*tuple(ACT_CL_TO_CF)), ref)

