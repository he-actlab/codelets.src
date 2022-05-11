from examples.genesys import OP_DTYPES, ASIC_CONFIG, \
    FXP_CONFIGS, QUANT_SCALE, SIGN_SHIFT, SW_PIPELINE_TEST
from functools import partial
from .fusion_layers import FusionOp


if SW_PIPELINE_TEST:
    FUSION_OP_INFO = {
        'div_add': {
            'cdlt': partial(FusionOp, 'div_add', use_quantization=False),
            'seq': ["Div", "Add"]
        },
        'add_relu': {
            'cdlt': partial(FusionOp, 'add_relu', use_quantization=False),
            'seq': ['Add', 'Relu'],
        },
        'add_leaky_relu': {
            'cdlt': partial(FusionOp, 'add_leaky_relu', use_quantization=False),
            'seq': ['Add', 'LeakyRelu'],
        },
        'leaky_relu_add': {
            'cdlt': partial(FusionOp, 'div_add', use_quantization=False),
            'seq': ['LeakyRelu', 'Add'],
        },
        'clip_depthwise_conv_bias': {
            'cdlt': partial(FusionOp, 'clip_depthwise_conv_bias', use_quantization=False),
            'seq': ['Clip', 'DepthwiseConv'],

        },
        'clip_depthwise_conv_bias_clip': {
            'cdlt': partial(FusionOp, 'clip_depthwise_conv_bias_clip', use_quantization=False),
            'seq': ['Clip', 'DepthwiseConv', 'Clip'],

        },
        'add_add': {
          'cdlt': partial(FusionOp, 'add_add', use_quantization=False),
          'seq': ["Add", "Add"]
        },
        'add_add4d': {
            'cdlt': partial(FusionOp, 'add_add4d', use_quantization=False),
            'seq': ["Add", "Add"]
        },
        'mul_add': {
            'cdlt': partial(FusionOp, 'mul_add', use_quantization=False),
            'seq': ["Mul", "Add"]
        },
        'sub_mul': {
            'cdlt': partial(FusionOp, 'sub_mul', use_quantization=False),
            'seq': ["Sub", "Mul"]
        },
        'sub_pow': {
            'cdlt': partial(FusionOp, 'sub_pow', use_quantization=False),
            'seq': ["Sub", "Pow"],
        },
        'add_sqrt_div': {
            'cdlt': partial(FusionOp, 'add_sqrt_div', use_quantization=False),
            'seq': ["Add", "Sqrt", "Div"],
        },
        'matmul_add': {
            'cdlt': partial(FusionOp, 'matmul_add', use_quantization=False),
            'seq': ["MatMul", "Add"]
        },

        'single_layer_info':
            {
                'Conv' : {'inputs': 3, 'outputs': 1},
                'Relu' : {'inputs': 1, 'outputs': 1},
                'LeakyRelu' : {'inputs': 1, 'outputs': 1},
                'Add' : {'inputs': 2, 'outputs': 1},
                'MaxPool': {'inputs': 1, 'outputs': 1}
            }
    }
else:
    FUSION_OP_INFO = {
    'add_add': {
      'cdlt': partial(FusionOp, 'add_add', use_quantization=False),
      'seq': ["Add", "Add"]
    },
    'mul_add': {
        'cdlt': partial(FusionOp, 'mul_add', use_quantization=False),
        'seq': ["Mul", "Add"]
    },
    'sub_mul': {
        'cdlt': partial(FusionOp, 'sub_mul', use_quantization=False),
        'seq': ["Sub", "Mul"]
    },

    'sub_pow': {
        'cdlt': partial(FusionOp, 'sub_pow', use_quantization=False),
        'seq': ["Sub", "Pow"],
    },
    'add_sqrt_div': {
        'cdlt': partial(FusionOp, 'add_sqrt_div', use_quantization=False),
        'seq': ["Add", "Sqrt", "Div"],
    },
    'matmul_add': {
        'cdlt': partial(FusionOp, 'matmul_add', use_quantization=False),
        'seq': ["MatMul", "Add"]
    },
    'matmul_add_add': {
        'cdlt': partial(FusionOp, 'matmul_add_add', use_quantization=False),
        'seq': ["MatMul", "Add", "Add"]
    },
    'matmul_add_gelu': {
        'cdlt': partial(FusionOp, 'matmul_add_gelu', use_quantization=False),
        'seq': ["MatMul", "Add", "Gelu"]
    },
    'matmul_div_add': {
        'cdlt': partial(FusionOp, 'matmul_div_add', use_quantization=False),
        'seq': ["MatMul", "Div", "Add"]
    },
    'conv_bias_relu': {
        'cdlt': partial(FusionOp, 'conv_relu', use_quantization=False),
        'seq': ['Conv', 'Relu']
    },
    'conv_bias_add_relu': {
        'cdlt': partial(FusionOp, 'conv_add_relu', use_quantization=False),
        'seq': ['Conv', 'Add', 'Relu'],
    },
    'conv_bias_add': {
        'cdlt': partial(FusionOp, 'conv_add', use_quantization=False),
        'seq': ['Conv', 'Add'],
    },
    'conv_bias_clip': {
        'cdlt': partial(FusionOp, 'conv_clip', use_quantization=False),
        'seq': ['Conv', 'Clip'],
    },
    'depthwise_conv_bias_clip': {
        'cdlt': partial(FusionOp, 'depthwise_conv_clip', use_quantization=False),
        'seq': ['DepthwiseConv', 'Clip'],
    },
    'conv_bias_leaky_relu': {
        'cdlt': partial(FusionOp, 'conv_leaky_relu', use_quantization=False),
        'seq': ['Conv', 'LeakyRelu']
    },
    'conv_bias_add_leaky_relu': {
        'cdlt': partial(FusionOp, 'conv_add_leaky_relu', use_quantization=False),
        'seq': ['Conv', 'Add', 'LeakyRelu'],
    },
    'conv_bias_leaky_relu_add': {
        'cdlt': partial(FusionOp, 'conv_leaky_relu_add', use_quantization=False),
        'seq': ['Conv', 'LeakyRelu', 'Add'],
    },
    'conv_bias_clip_depthwise_conv_bias': {
        'cdlt': partial(FusionOp, 'conv_bias_clip_depthwise_conv_bias', use_quantization=False),
        'seq': ['Conv', 'Clip', 'DepthwiseConv'],

    },
    'conv_bias_clip_depthwise_conv_bias_clip': {
        'cdlt': partial(FusionOp, 'conv_bias_clip_depthwise_conv_bias_clip', use_quantization=False),
        'seq': ['Conv', 'Clip', 'DepthwiseConv', 'Clip'],

    },
    'single_layer_info':
        {
            'Conv' : {'inputs': 3, 'outputs': 1},
            'Relu' : {'inputs': 1, 'outputs': 1},
            'LeakyRelu' : {'inputs': 1, 'outputs': 1},
            'Add' : {'inputs': 2, 'outputs': 1},
            'MaxPool': {'inputs': 1, 'outputs': 1}
        }
}
UNQUANT_FUSION_IMPLS = {k : v['cdlt'] for k, v in FUSION_OP_INFO.items() if k != 'single_layer_info'}