import numpy as np
from . import GENESYS_CFG

def pool2d(x, k, stride, padding=0):
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    N, C, H, W = x_padded.shape

    pool_height, pool_width = k, k

    # assert (H - pool_height) % stride == 0, 'Invalid height'
    # assert (W - pool_width) % stride == 0, 'Invalid width'

    out_height = (H - pool_height) // stride + 1
    out_width = (W - pool_width) // stride + 1

    x_split = x_padded.reshape(N * C, 1, H, W)
    x_cols = im2col_indices(x_split, pool_height, pool_width, padding=0, stride=stride)
    x_cols_argmax = np.argmax(x_cols, axis=0)
    x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
    out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)
    # cache = (x, x_cols, x_cols_argmax, pool_param)
    return out


def maxpool(image, f=2, s=2):
    '''
    Downsample `image` using kernel size `f` and stride `s`
    '''
    n_c, c, h_prev, w_prev = image.shape
    h = int((h_prev - f) / s) + 1
    w = int((w_prev - f) / s) + 1

    downsampled = np.zeros((n_c,c, h, w))
    for i in range(n_c):
        for j in range(c):
            # slide maxpool window over each part of the image and assign the max value at each step to the output
            curr_y = out_y = 0
            while curr_y + f <= h_prev:
                curr_x = out_x = 0
                while curr_x + f <= w_prev:
                    downsampled[i, j, out_y, out_x] = np.max(image[i,j, curr_y:curr_y + f, curr_x:curr_x + f])
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
    return downsampled

def pad_conv(layer_data):
    x = layer_data['input']
    out = layer_data['output']
    wgt = layer_data['params']['weight']
    b = layer_data['params']['bias']
    if x.shape[-1] % GENESYS_CFG['ARRAY_M'] != 0:
        ic_init = x.shape[-1]
        ic_pad = (GENESYS_CFG['ARRAY_M'] - ic_init) % GENESYS_CFG['ARRAY_M']
        assert (ic_pad + ic_init) % GENESYS_CFG['ARRAY_M'] == 0
        padding = (0, ic_pad)
        x_pad = ((0, 0), (0, 0), (0, 0), padding)

        x = np.pad(x, x_pad, "constant")
        assert wgt.shape[-1] == ic_init
        wgt = np.pad(wgt, x_pad, "constant")

    if out.shape[-1] % GENESYS_CFG['ARRAY_N'] != 0:
        oc_init = out.shape[-1]
        oc_pad = (GENESYS_CFG['ARRAY_N'] - oc_init) % GENESYS_CFG['ARRAY_N']
        assert (oc_pad + oc_init) % GENESYS_CFG['ARRAY_N'] == 0
        padding = (0, oc_pad)
        out_pad = ((0, 0), (0, 0), (0, 0), padding)
        out = np.pad(out, out_pad, "constant")
        assert wgt.shape[-2] == oc_init
        wgt_pad = ((0, 0), (0, 0), padding, (0, 0))
        wgt = np.pad(wgt, wgt_pad, "constant")
        assert b.shape[0] == oc_init
        b = np.pad(b, padding, "constant")
    return x, wgt, b, out


def pad_gemm(layer_data):
    x = layer_data['input']
    out = layer_data['output']
    wgt = layer_data['params']['weight']
    b = layer_data['params']['bias']
    if x.shape[-1] % GENESYS_CFG['ARRAY_M'] != 0:
        ic_init = x.shape[-1]
        ic_pad = (GENESYS_CFG['ARRAY_M'] - ic_init) % GENESYS_CFG['ARRAY_M']
        assert (ic_pad + ic_init) % GENESYS_CFG['ARRAY_M'] == 0
        padding = (0, ic_pad)
        x_pad = ((0, 0), padding)

        x = np.pad(x, x_pad, "constant")
        assert wgt.shape[0] == ic_init
        wgt_pad = (padding, (0, 0))
        wgt = np.pad(wgt, wgt_pad, "constant")

    if out.shape[-1] % GENESYS_CFG['ARRAY_N'] != 0:
        oc_init = out.shape[-1]
        oc_pad = (GENESYS_CFG['ARRAY_N'] - oc_init) % GENESYS_CFG['ARRAY_N']
        assert (oc_pad + oc_init) % GENESYS_CFG['ARRAY_N'] == 0
        padding = (0, oc_pad)
        out_pad = ((0, 0), padding)
        out = np.pad(out, out_pad, "constant")
        assert wgt.shape[-1] == oc_init
        wgt_pad = ((0, 0), padding)
        wgt = np.pad(wgt, wgt_pad, "constant")
        assert b.shape[0] == oc_init
        b = np.pad(b, padding, "constant")
    return x, wgt, b, out

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.
    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None

    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = x.shape
    F, C_filter, HH, WW = w.shape
    assert C == C_filter, 'Number of channels are not equal between input and filter'
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    H_new = int(1 + (H + 2*pad - HH) / stride)
    W_new = int(1 + (W + 2*pad - WW) / stride)

    out = np.zeros((N, F, H_new, W_new), dtype=x.dtype)

    last_row = H + 2*pad - HH + 1
    last_col = W + 2*pad - WW + 1

    for f in range(F):
        i_out = 0
        for i in range(0, last_row, stride):
            j_out = 0
            for j in range(0, last_col, stride):
                x_current = x_pad[:, :, i:(i+HH), j:(j+WW)]
                out[:, f, i_out, j_out] = np.dot(x_current.reshape((N, -1)), w[f].flatten()) + b[f]
                j_out += 1
            i_out += 1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache

def manual_gemm(inputs, weights, o_coord):
    M = inputs.shape[0]
    N = inputs.shape[1]
    P = weights.shape[1]
    outputs = np.zeros((M, P), dtype=np.int32)
    inputs = inputs.astype(np.int32)
    weights = weights.astype(np.int32)
    compilation_info = {i: [] for i in range(N)}
    for p in range(P):
        for n in range(N):
            for m in range(M):
                partial_sum = inputs[m, n] * weights[n, p]
                outputs[m, p] += partial_sum

                if (m, p) == o_coord:
                    all_coords = (m, n, p)
                    icoord = (m, n)
                    icoord_idx = np.ravel_multi_index([m, n], inputs.shape)
                    wcoord = (n, p)
                    wcoord_idx = np.ravel_multi_index([n, p], weights.shape)
                    ocoord = (m, p)
                    ocoord_idx = np.ravel_multi_index([m, p], outputs.shape)
                    compilation_info[n].append(
                        f'"{all_coords}", {ocoord_idx}, {icoord_idx}, {wcoord_idx}, {inputs[icoord]}, {weights[wcoord]}, {partial_sum}')
    return outputs, compilation_info

def manual_conv_from_existing(inputs, weights, out, stride):
    N, IH, IW, IC = inputs.shape
    KH, KW, IC_, OC = weights.shape
    N_, OH, OW, OC_ = out.shape
    assert N_ == N
    assert IC == IC_
    assert OC == OC_
    outputs = np.zeros(out.shape, dtype=np.int32)
    for oc in range(OC):
        for n in range(N):
            for ic in range(IC):
                for kh in range(KH):
                    for kw in range(KW):
                        for y in range(OH):
                            for x in range(OW):
                                partial_sum = inputs[n, kh + y * stride, kw + x * stride, ic] * weights[kh, kw, ic, oc]
                                outputs[n, y, x, oc] += partial_sum
    return outputs

def manual_conv(inputs, weights, cdlt, o_coord, layout="nhwc"):
    if layout == "nhwc":
        N, IH, IW, IC = inputs.shape
        KH, KW, IC_, OC = weights.shape
        # KH, KW, OC, IC_ = weights.shape
        N_, OH, OW, OC_ = cdlt.outputs[0].shape
        out_shape = cdlt.outputs[0].shape
    else:
        N, IC, IH, IW = inputs.shape
        OC, IC_, KH, KW,  = weights.shape
        N_, OH, OW, OC_ = cdlt.outputs[0].shape
        out_shape = (N_, OC_, OH, OW)
    assert isinstance(o_coord, tuple) and len(o_coord) == 4
    assert N_ == N
    assert IC == IC_
    assert OC == OC_
    outputs = np.zeros(out_shape, dtype=np.int32)
    inputs = inputs.astype(np.int32)
    weights = weights.astype(np.int32)
    stride = cdlt.required_params['stride'].value
    compilation_info = {i: [] for i in range(IC)}
    if layout == "nhwc":
        for oc in range(OC):
            for n in range(N):
                for ic in range(IC):
                    for kh in range(KH):
                        for kw in range(KW):
                            for y in range(OH):
                                for x in range(OW):


                                    partial_sum = inputs[n, kh + y*stride, kw + x*stride, ic] * weights[kh, kw, ic, oc]
                                    outputs[n, y, x, oc] += partial_sum
                                    if (n, y, x, oc) == o_coord:
                                        all_coords = (oc, n, ic, kh, kw, y, x)
                                        icoord = (n, kh + y*stride, kw + x*stride, ic)
                                        icoord_idx = np.ravel_multi_index([n, kh + y*stride, kw + x*stride, ic], inputs.shape)
                                        wcoord = (kh, kw, ic, oc)
                                        wcoord_idx = np.ravel_multi_index([kh, kw, ic, oc], weights.shape)
                                        ocoord = (n, y, x, oc)
                                        ocoord_idx = np.ravel_multi_index([n, y, x, oc], outputs.shape)
                                        compilation_info[ic].append(f'"{all_coords}", {ocoord_idx}, {icoord_idx}, {wcoord_idx}, {inputs[icoord]}, {weights[wcoord]}, {partial_sum}')

                                    # outputs[n, y, x, oc] += inputs[n, kh + y*stride, kw + x*stride, ic] * weights[kh, kw, oc, ic]

    else:
        compilation_info = {}
        for oc in range(OC):
            for n in range(N):
                for ic in range(IC):
                    for kh in range(KH):
                        for kw in range(KW):
                            for y in range(OH):
                                for x in range(OW):
                                    outputs[n, oc, y, x] += inputs[n, ic, kh + y * stride, kw + x * stride] * weights[
                                        oc, ic, kh, kw]
        outputs = outputs.transpose(0, 2, 3, 1)

    return outputs, compilation_info

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

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    # assert (H + 2 * padding - field_height) % stride == 0
    # assert (W + 2 * padding - field_height) % stride == 0
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