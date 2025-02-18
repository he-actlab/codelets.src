import fxpmath
import numpy as np
from fxpmath import Fxp
from codelets.examples.genesys import FXP_CONFIGS
from typing import List, Tuple, Optional

def save_array(path, data):
    with open(path, 'w') as f:
        f.write('\n'.join([str(i) for i in data.flatten().tolist()]))

def compute_range(fxp_dtype, scale=1):
    cfg = FXP_CONFIGS[fxp_dtype]
    upper_val = (1 << (np.int32(cfg['n_word']//scale) - 1)) - 1
    lower_val = -upper_val - 1

    return lower_val, upper_val



def from_fxp(v, dtype):
    fp = Fxp(v, **FXP_CONFIGS[dtype])
    fp.val = v
    return fp

def float_from_fxp(v):
    assert isinstance(v, fxpmath.Fxp)


def numpy_datagen(shape, bitwidth, scale=2, cast_to=None, vrange=None, fxp_dtype='FXP32', constant_val=None, print_range=False):
    if vrange is not None:
        assert isinstance(vrange, tuple) and len(vrange) == 2
        low, high = vrange
        assert high > low
        ref_low, ref_high = compute_range(fxp_dtype, scale)
        assert ref_low <= low and ref_high >= high
        v = np.random.randint(low=low, high=high,
                              size=shape, dtype=np.int64)
    elif constant_val is not None:
        v = np.full(shape, constant_val, dtype=np.int64)
    else:
        low, high = compute_range(fxp_dtype, scale)
        if print_range:
            print(f"High: {high}, Low: {low}")
        v = np.random.randint(low=low, high=high,
                              size=shape, dtype=np.int64)

    return v

def quantize_np(d, dtype, inpt=None):
    if dtype == "FXP32":
        high_bits = 16
        low_bits = 16
        dec_place = 32
    else:
        raise RuntimeError

    out = (d << (dec_place - high_bits)) >> (dec_place - high_bits)
    out = (out >> (dec_place - low_bits))

    return out

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


def pad_tensor(
    np_arr: np.array,
    pad_value: float,
    padding_before: List[int],
    padding_after: List[int],
    dtype: str,
) -> np.array:
    """Pad the spatial dimensions of the given array."""
    orig_shape = list(np_arr.shape)
    padded_shape = list(np_arr.shape)
    n = len(orig_shape)
    for dim in range(2, n):
        i = dim - 2
        padded_shape[dim] += padding_after[i] + padding_before[i]

    pad_np = (np.zeros(shape=padded_shape) + pad_value).astype(dtype)
    ranges_it = [range(padded_shape[0]), range(padded_shape[1])]
    for dim in range(2, n):
        i = dim - 2
        ranges_it.append(range(padding_before[i], padding_before[i] + orig_shape[dim]))
    pad_np[np.ix_(*ranges_it)] = np_arr
    return pad_np

def get_slice(
    spatial_dimensions: int,
    pad_np: np.array,
    dim_coord: Tuple[int],
    kernel: Tuple[int],
    strides: Tuple[int],
    dilation: Tuple[int],
) -> List[slice]:
    """
    Programmatically create a slice object of the right dimensions for pad_np.
    We assume pad_np's first two dimensions are not spatial and are not touched by the pad.
    pad_np[slice] should give the elements of the data that a pool operation will use for the
    step given in dim_coord.
    """
    assert isinstance(dim_coord, tuple)
    assert isinstance(kernel, tuple)
    assert isinstance(strides, tuple)
    assert isinstance(dilation, tuple)
    slices = [slice(None)] * spatial_dimensions

    for nd in range(spatial_dimensions):
        slices[nd] = slice(
            dim_coord[nd] * strides[nd],
            dim_coord[nd] * strides[nd] + (kernel[nd] - 1) * dilation[nd] + 1,
            dilation[nd],
        )

    # Add back batch and channel dimensions
    slices = [slice(None), slice(None)] + slices

    return tuple(slices)
