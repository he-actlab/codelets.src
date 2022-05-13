from examples.genesys import OP_DTYPES, ASIC_CONFIG, \
    FXP_CONFIGS, QUANT_SCALE, SIGN_SHIFT, SW_PIPELINE_TEST
from functools import partial
from .fusion_layers import FusionOp, DFG



FUSION_OP_INFO = {
        'div_add': {
            'cdlt': partial(FusionOp, 'div_add'),
            'dfg' : DFG('add', [DFG('div', [0, 'mul_rhs']),
                                1]),
            'seq': ["Div",
                    "Add"]
        },
        'add_relu': {
            'cdlt': partial(FusionOp, 'add_relu'),
            'dfg': DFG('relu', [DFG('add', [0, 1])]),
            'seq': ['Add', 'Relu'],
        },
        'add_leaky_relu': {
            'cdlt': partial(FusionOp, 'add_leaky_relu'),
            'dfg': DFG('leaky_relu', [DFG('add', [0, 1]), 'alpha']),
            'seq': ['Add', 'LeakyRelu'],
        },
        'leaky_relu_add': {
            'cdlt': partial(FusionOp, 'div_add'),
            'dfg': DFG('add', [DFG('leaky_relu', [0, 'alpha']), 1]),

            'seq': ['LeakyRelu', 'Add'],
        },
        'clip_depthwise_conv': {
            'cdlt': partial(FusionOp, 'clip_depthwise_conv'),
            'dfg': DFG('depthwise_conv',
                       [DFG('clip', [0, 'min', 'max']), 1, 'stride', 'pad']),
            'seq': ['Clip', 'DepthwiseConv'],
        },
        'add_add': {
          'cdlt': partial(FusionOp, 'add_add'),
            'dfg': DFG('add', [DFG('add', [0, 1]), 2]),
            'seq': ["Add", "Add"]
        },
        'add_add4d': {
            'cdlt': partial(FusionOp, 'add_add4d'),
            'dfg': DFG('add', [DFG('add', [0, 1]), 2]),
            'seq': ["Add", "Add"]
        },
        'mul_add': {
            'cdlt': partial(FusionOp, 'mul_add'),
            'dfg': DFG('add', [DFG('mul', [0, 1]), 2]),
            'seq': ["Mul", "Add"]
        },
    'bias_add_clip': {
        'cdlt': partial(FusionOp, 'bias_add_clip'),
        'dfg': DFG('clip', [DFG('bias_add', [0, 1]), 'min', 'max']),
        'seq': ['BiasAdd', 'Clip']
    },
        'sub_mul': {
            'cdlt': partial(FusionOp, 'sub_mul'),
            'dfg': DFG('mul', [DFG('sub', [0, "sub_rhs"]), "mul_rhs"]),
            'seq': ["Sub", "Mul"]
        },
        'sub_pow': {
            'cdlt': partial(FusionOp, 'sub_pow'),
            'dfg': DFG('square', [DFG('sub', [0, 1])]),
            'seq': ["Sub", "Pow"],
        },
        'add_sqrt_div': {
            'cdlt': partial(FusionOp, 'add_sqrt_div'),
            'dfg': DFG('div', [DFG('sqrt', [
                                            DFG('add', [0, 'add_lhs'])
                                        ]),
                               1]),

            'seq': ["Add", "Sqrt", "Div"],
        },
        'matmul_add': {
            'cdlt': partial(FusionOp, 'matmul_add'),
            'dfg': DFG('gemm', [0, 1, 2]),
            'seq': ["MatMul", "Add"]
        },

        'single_layer_info':
            {
                'Conv' : {'inputs': 3, 'outputs': 1},
                'Relu' : {'inputs': 1, 'outputs': 1},
                'LeakyRelu' : {'inputs': 1, 'outputs': 1},
                'Add' : {'inputs': 2, 'outputs': 1},
                'MaxPool': {'inputs': 1, 'outputs': 1}
            },
        'clip_depthwise_conv_bias': {
            'cdlt': partial(FusionOp, 'clip_depthwise_conv_bias'),
            'dfg': DFG('depthwise_conv',
                       [DFG('clip', [0, 'min', 'max']), 1, 2, 'stride', 'pad']),
            'seq': ['Clip', 'DepthwiseConv'],

        },
        'clip_depthwise_conv_bias_clip': {
            'cdlt': partial(FusionOp, 'clip_depthwise_conv_bias_clip'),
            'dfg': DFG('clip',
                       [DFG('depthwise_conv',
                            [DFG('clip', [0, 'min', 'max']), 1, 2, 'stride', 'pad']), 'min', 'max']),
            'seq': ['Clip', 'DepthwiseConv', 'Clip'],

        },
        'depthwise_conv_bias_clip': {
            'cdlt': partial(FusionOp, 'depthwise_conv_bias_clip'),
            'dfg': DFG('clip', [DFG('depthwise_conv', [0, 1, 2, 'stride', 'pad']), 'min', 'max']),
            'seq': ['DepthwiseConv', 'Clip'],
        },

    }

if not SW_PIPELINE_TEST:

    FUSION_OP_INFO.update({
        'matmul_add': {
            'cdlt': partial(FusionOp, 'matmul_add'),
            'dfg': DFG('gemm', [0, 1, 2]),
            'seq': ["MatMul", "Add"]
        },
        'matmul_add_add': {
            'cdlt': partial(FusionOp, 'matmul_add_add'),
            'dfg': DFG('add', [DFG('gemm', [0, 1, 2]), 3]),
            'seq': ["MatMul", "Add", "Add"]
        },
        'matmul_add_gelu': {
            'cdlt': partial(FusionOp, 'matmul_add_gelu'),
            'dfg': DFG('gelu', [DFG('gemm', [0, 1, 2])]),
            'seq': ["MatMul", "Add", "Gelu"]
        },
        'matmul_div_add': {
            'cdlt': partial(FusionOp, 'matmul_div_add'),
            'dfg': DFG('add', [DFG('mul', [DFG('matmul', [0, 1]), 'mul_rhs']), 2]),
            'seq': ["MatMul", "Div", "Add"]
        },
        'conv_relu': {
            'cdlt': partial(FusionOp, 'conv_relu'),
            'dfg': DFG('relu', [DFG('conv', [0,1,2,'stride', 'pad'])]),
            'seq': ['Conv', 'Relu']
        },
        'conv_bias_relu': {
            'cdlt': partial(FusionOp, 'conv_relu'),
            'dfg': DFG('relu', [DFG('conv', [0,1,2,'stride', 'pad'])]),
            'seq': ['Conv', 'Relu']
        },
        'conv_bias_add_relu': {
            'cdlt': partial(FusionOp, 'conv_add_relu'),
            'dfg' : DFG('relu', [DFG('add', [DFG('conv', [0,1,2, 'stride', 'pad']), 3])]),
            'seq': ['Conv', 'Add', 'Relu'],
        },
        'conv_bias_add': {
            'cdlt': partial(FusionOp, 'conv_add'),
            'dfg': DFG('add', [DFG('conv', [0,1,2,'stride', 'pad']), 3]),
            'seq': ['Conv', 'Add'],
        },
        'conv_bias_clip': {
            'cdlt': partial(FusionOp, 'conv_clip'),
            'dfg': DFG('clip', [DFG('conv', [0, 1, 2, 'stride', 'pad']), 'min', 'max']),
            'seq': ['Conv', 'Clip'],
        },
        'conv_clip': {
            'cdlt': partial(FusionOp, 'conv_clip'),
            'dfg': DFG('clip', [DFG('conv', [0, 1, 2, 'stride', 'pad']), 'min', 'max']),
            'seq': ['Conv', 'Clip'],
        },

        'conv_bias_leaky_relu': {
            'cdlt': partial(FusionOp, 'conv_leaky_relu'),
            'dfg': DFG('leaky_relu', [DFG('conv', [0, 1, 2, 'stride', 'pad']), 'alpha']),
            'seq': ['Conv', 'LeakyRelu']
        },
        'conv_bias_add_leaky_relu': {
            'cdlt': partial(FusionOp, 'conv_add_leaky_relu'),
            'dfg': DFG('leaky_relu', [DFG('add', [DFG('conv', [0, 1, 2, 'stride', 'pad']), 3]), 'alpha']),
            'seq': ['Conv', 'Add', 'LeakyRelu'],
        },
        'conv_bias_leaky_relu_add': {
            'cdlt': partial(FusionOp, 'conv_leaky_relu_add'),
            'dfg': DFG('add', [DFG('leaky_relu', [DFG('conv', [0, 1, 2, 'stride', 'pad']), 'alpha']), 3]),

            'seq': ['Conv', 'LeakyRelu', 'Add'],
        },
        'conv_bias_clip_depthwise_conv_bias_add': {
            'cdlt': partial(FusionOp, 'conv_bias_clip_depthwise_conv_bias_add'),
            'dfg': DFG('bias_add',
                           [DFG('depthwise_conv',
                                [DFG('clip',
                                     [DFG('conv', [0, 1, 2, 'stride', 'pad']),
                                      'min', 'max']),
                                 3, 's2', 'p2']),
                                    4]),
            'seq': ['Conv', 'Clip', 'DepthwiseConv', 'BiasAdd'],

        },
        'conv_bias_clip_depthwise_conv_bias': {
            'cdlt': partial(FusionOp, 'conv_bias_clip_depthwise_conv_bias_add'),
            'dfg': DFG('bias_add',
                       [DFG('depthwise_conv',
                            [DFG('clip',
                                 [DFG('conv', [0, 1, 2, 'stride', 'pad']),
                                  'min', 'max']),
                             3, 's2', 'p2']),
                        4]),
            'seq': ['Conv', 'Clip', 'DepthwiseConv', 'BiasAdd'],

        },
    'conv_bias_clip_depthwise_conv_bias_add_clip': {
        'cdlt': partial(FusionOp, 'conv_bias_clip_depthwise_conv_bias_add_clip'),
        'dfg': DFG('clip', [DFG('bias_add',
                               [DFG('depthwise_conv',
                                    [DFG('clip',
                                         [DFG('conv', [0, 1, 2, 'stride', 'pad']),
                                          'min', 'max']),
                                     3, 's2', 'p2']),
                                4]),
                                'min', 'max']),
        'seq': ['Conv', 'Clip', 'DepthwiseConv', 'BiasAdd','Clip'],

    },
    'conv_bias_clip_depthwise_conv_bias_clip': {
        'cdlt': partial(FusionOp, 'conv_bias_clip_depthwise_conv_bias_clip'),
        'dfg': DFG('clip', [DFG('bias_add',
                               [DFG('depthwise_conv',
                                    [DFG('clip',
                                         [DFG('conv', [0, 1, 2, 'stride', 'pad']),
                                          'min', 'max']),
                                     3, 's2', 'p2']),
                                4]),
                                'min', 'max']),
        'seq': ['Conv', 'Clip', 'DepthwiseConv', 'BiasAdd','Clip'],

    },
})
UNQUANT_FUSION_IMPLS = {k : v['cdlt'] for k, v in FUSION_OP_INFO.items() if k != 'single_layer_info'}