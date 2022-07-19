from . import ReferenceOp, quantize_np

class Gradient(ReferenceOp):
    def __init__(self, cdlt, hag):
        operands = [cdlt.inputs[0], cdlt.inputs[1]]
        outputs = [cdlt.outputs[0]]
        super().__init__(cdlt, operands, outputs, hag)
        raise RuntimeError(f"Gradients are not yet supported")

def load_gradient_impls(cfg):
    GRADIENT_IMPLS = {'average_pool_grad': Gradient,
            "batchnorm_grad": Gradient,
            "cross_entropy_loss_grad": Gradient,
            'elem_tanh_grad': Gradient,
            'elem_tanh_grad2d': Gradient,
            "flatten_grad": Gradient,
            'global_average_pool_grad': Gradient,
            'max_pool_grad': Gradient,
            'relu_grad2d': Gradient,
            'relu_grad': Gradient,
            "sgd1d": Gradient,
            "sgd2d": Gradient,
            "sgd3d": Gradient,
            "sgd4d": Gradient
          }
    return GRADIENT_IMPLS