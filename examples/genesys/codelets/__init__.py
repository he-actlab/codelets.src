
from .util import range_from_cfg, create_immediate_with_operand, add_quantization,\
    add_scale_op, add_sys_array_cast, add_scale_and_cast_op
from .arch_constraints import add_simd_constraint, add_conv_constraints,\
    add_gemm_constraints, add_simd_tile_constraint

# SW Impl
from .reference_impls.fusion_layers import FUSION_IMPLS
from .reference_impls.unquantized_fusion_layers import UNQUANT_FUSION_IMPLS
from .reference_impls.gradients import GRADIENT_IMPLS
from .reference_impls.binary import BINARY_IMPLS
from .reference_impls.unary import UNARY_IMPLS
from .reference_impls.dnn import DNN_IMPLS
from .reference_impls.transform import TRANSFORM_IMPLS
from .reference_impls.systolic_array import SA_IMPLS
from .reference_impls.reduction import REDUCTION_IMPLS

# Codelets
from .fusion_layers import FUSION_CODELETS, FUSION_OP_INFO
from .unquantized_fusion_layers import UNQUANT_FUSION_CODELETS, UNQUANT_FUSION_OP_INFO
from .gradients import GRADIENT_CDLTS
from .binary import BINARY_CODELETS
from .unary import UNARY_CODELETS
from .dnn import DNN_CDLTS
from .transform import TRANSFORM_CDLTS
from .systolic_array import SA_CDLTS
from .reduction import REDUCTION_CODELETS
from examples.genesys import ALL_QUANT_OFF


if ALL_QUANT_OFF:
    GENESYS_IMPLS = {
        **UNQUANT_FUSION_IMPLS,
        **GRADIENT_IMPLS,
        **BINARY_IMPLS,
        **UNARY_IMPLS,
        **DNN_IMPLS,
        **TRANSFORM_IMPLS,
        **SA_IMPLS,
        **REDUCTION_IMPLS
    }
    GENESYS_CODELETS = {
        **UNQUANT_FUSION_CODELETS,
        **GRADIENT_CDLTS,
        **BINARY_CODELETS,
        **UNARY_CODELETS,
        **DNN_CDLTS,
        **TRANSFORM_CDLTS,
        **SA_CDLTS,
        **REDUCTION_CODELETS
    }
else:
    GENESYS_IMPLS = {
        **FUSION_IMPLS,
        **GRADIENT_IMPLS,
        **BINARY_IMPLS,
        **UNARY_IMPLS,
        **DNN_IMPLS,
        **TRANSFORM_IMPLS,
        **SA_IMPLS,
        **REDUCTION_IMPLS
    }
    GENESYS_CODELETS = {
        **FUSION_CODELETS,
        **GRADIENT_CDLTS,
        **BINARY_CODELETS,
        **UNARY_CODELETS,
        **DNN_CDLTS,
        **TRANSFORM_CDLTS,
        **SA_CDLTS,
        **REDUCTION_CODELETS
    }

SPLIT_INFO = {}
SPLIT_INFO['depthwise_conv_bias'] = ('bias_add', 3, [('depthwise_conv', 3, ([0, 1],
                                                                        {'stride': 'stride', 'pad': 'pad',
                                                                         'groups': 'groups', 'dilation': 'dilation'})),2],)
for k in GENESYS_CODELETS.keys():
    if k not in GENESYS_IMPLS:
        raise RuntimeError(f"Not all codelets have a software implementation: {k}")