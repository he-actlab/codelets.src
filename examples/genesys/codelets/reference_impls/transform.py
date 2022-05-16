from typing import List

from collections import Iterable, namedtuple
from examples.genesys import FXP_CONFIGS
from fxpmath import Fxp
import numpy as np
from functools import partial
from . import ReferenceOp, quantize_np

class Transform(ReferenceOp):

    def __init__(self, transform_type, cdlt, hag):
        self.transform_type = transform_type
        self.dtype = "FXP32"
        self.axis = self.cdlt.required_params['axis'].value
        operands = [cdlt.inputs[0]]
        outputs = [cdlt.outputs[0]]
        super().__init__(cdlt, operands, outputs, hag)


    def fn_impl(self, inouts):
        data = inouts['inputs'][0].data
        out_shape = self.outputs[0].shape
        if len(data.shape) == 4:
            data = data.transpose((0, 3, 1, 2))

        if self.transform_type == "reshape":
            out = data.reshape(out_shape)
        elif self.transform_type == "squeeze":
            out = np.squeeze(data)
        else:
            raise RuntimeError("unknown reduction type")

        if len(out.shape) == 4:
            out = out.transpose((0, 2, 3, 1))
        inouts['outputs'] = [out]
        return inouts

def load_transform_impls(cfg):

    TRANSFORM_IMPLS = {
        'tensor_reshape4d2d': partial(Transform, 'reshape'),
        'tensor_reshape4d3d': partial(Transform, 'reshape'),
        'tensor_reshape3d4d': partial(Transform, 'reshape'),
        # 'tensor_flip': tensor_flip,
        # 'tensor_pad': tensor_pad,
        # 'concat': concat,
        'tensor_squeeze' : partial(Transform, 'squeeze'),
        # 'resize': partial(Transform, 'resize')
    }
    return TRANSFORM_IMPLS