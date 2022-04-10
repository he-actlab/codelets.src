
from .util import range_from_cfg, create_immediate_with_operand, add_quantization,\
    add_scale_op, add_sys_array_cast, add_scale_and_cast_op
from .arch_constraints import add_simd_constraint, add_conv_constraints,\
    add_gemm_constraints, add_simd_tile_constraint
from .fusion_layers import FUSION_CODELETS, FUSION_OP_INFO
from .gradients import GRADIENT_CDLTS
from .binary import BINARY_CODELETS
from .unary import UNARY_CODELETS
from .dnn import DNN_CDLTS
from .transform import TRANSFORM_CDLTS
from .systolic_array import SA_CDLTS
from .reduction import REDUCTION_CODELETS



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