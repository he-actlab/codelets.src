import argparse
import polymath as pm
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from collections import OrderedDict
import io
from pathlib import Path
from onnxsim import simplify
from collections import namedtuple
from torchvision.ops._register_onnx_ops import _onnx_opset_version
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
import sys
import onnx
from onnx.tools import update_model_dims
from onnx.utils import Extractor
from onnx import version_converter, helper
import numpy as np
import csv

import onnxruntime as ort

CWD = Path(f"{__file__}").parent

Targets = namedtuple('Targets', ['boxes', 'masks', 'labels'])

LAYER_UTILS = {
    "gemm":
        {
         "fn": lambda params: nn.Linear(params['N'], params['P'], bias=True),
         "input_gen": lambda params: (torch.randn(params['M'], params['N']), {}),
         "fn_params": ['N', 'P'],
          "input_gen_params": ['M', 'N']
         },
    "gemm_no_bias":
        {
            "fn": lambda params: nn.Linear(params['N'], params['P'], bias=False),
            "input_gen": lambda params: (torch.randn(params['M'], params['N']), {}),
            "fn_params": ['N', 'P'],
            "input_gen_params": ['M', 'N']
        },
    "conv": {"fn": lambda params: nn.Conv2d(in_channels=params['IC'], out_channels=params['OC'],
                                     kernel_size=params['KH'], stride=params['stride'], padding=params['pad'],
                                     bias=False),
             "input_gen": lambda params: (torch.randn(params['N'], params['IC'], params['IH'], params['IW']), {}),
             "fn_params": ['IC', 'OC', 'KH', 'stride', 'pad'],
             "input_gen_params": ['N', 'IC', 'IH', 'IW']
             },
    "conv_bias": {"fn": lambda params: nn.Conv2d(in_channels=params['IC'], out_channels=params['OC'],
                                     kernel_size=params['KH'], stride=params['stride'], padding=params['pad'],
                                     bias=True),
             "input_gen": lambda params: (torch.randn(params['N'], params['IC'], params['IH'], params['IW']), {}),
             "fn_params": ['IC', 'OC', 'KH', 'stride', 'pad'],
             "input_gen_params": ['N', 'IC', 'IH', 'IW']
             },
    "clip": {"fn": lambda params: torch.clamp,
             "input_gen": lambda params: (torch.randn(params['N'], params['C'], params['H'], params['W']),
                            {"min": params['minval'], "max": params['maxval']}),
             "fn_params": [],
             "input_gen_params": (["N", "C", "H", "W"], {"min": "minval", "max": "maxval"})
             },
    "elem_clip": {"fn": lambda params: torch.clamp,
             "input_gen": lambda params: (torch.randn(params['N'], params['C'], params['H'], params['W']),
                            {"min": params['minval'], "max": params['maxval']}),
             "fn_params": [],
             "input_gen_params": (["N", "C", "H", "W"], {"min": "minval", "max": "maxval"})
             },
    "elem_add": {"fn": lambda params: torch.add,
                 "input_gen": lambda params: ((torch.randn(params['N'],params['C'], params['H'], params['W']), torch.randn(params['N'],params['C'], params['H'], params['W'])), {}),
                 "input_gen_params": ["N", "C", "H", "W"],
                 "fn_params": []},
    "elem_mul": {"fn": lambda params: torch.mul,
                 "input_gen": lambda params: ((torch.randn(params['N'],params['C'], params['H'], params['W']), torch.randn(params['N'],params['C'], params['H'], params['W'])), {}),
                 "input_gen_params": ["N", "C", "H", "W"],
                 "fn_params": []},
    "elem_sub": {"fn": lambda params: torch.sub,
                 "input_gen": lambda params: ((torch.randn(params['N'],params['C'], params['H'], params['W']), torch.randn(params['N'],params['C'], params['H'], params['W'])), {}),
                 "input_gen_params": ["N", "C", "H", "W"],
                 "fn_params": []},
    "elem_div": {"fn": lambda params: torch.div,
                 "input_gen": lambda params: ((torch.randn(params['N'],params['C'], params['H'], params['W']), torch.randn(params['N'],params['C'], params['H'], params['W'])), {}),
                 "input_gen_params": ["N", "C", "H", "W"],
                 "fn_params": []},
    "elem_less": {"fn": lambda params: torch.lt,
                  "input_gen": lambda params: ((torch.randn(params['N'], params['C'], params['H'], params['W']),
                                                torch.randn(params['N'], params['C'], params['H'], params['W'])), {}),
                  "input_gen_params": ["N", "C", "H", "W"],
                  "fn_params": []},
    "elem_equal": {"fn": lambda params: torch.eq,
                   "input_gen": lambda params: ((torch.randn(params['N'], params['C'], params['H'], params['W']),
                                                 torch.randn(params['N'], params['C'], params['H'], params['W'])), {}),
                   "input_gen_params": ["N", "C", "H", "W"],
                   "fn_params": []},
    "elem_exp": {"fn": lambda params: torch.exp, "input_gen": lambda params: (torch.randn(params['N'], params['C'], params['H'], params['W']), {}), "input_gen_params": ["N", "C", "H", "W"], "fn_params": []},
    "reduce_sum": {"fn": lambda params: torch.sum, "input_gen": lambda params: (torch.randn(params['N'], params['C']), {}), "input_gen_params": ["N", "C"], "fn_params": []},
    "relu": {"fn": lambda params: torch.nn.ReLU(), "input_gen": lambda params: (torch.randn(params['N'], params['C'], params['H'], params['W']), {}), "input_gen_params": ["N", "C", "H", "W"], "fn_params": []},
    "leaky_relu": {"fn": lambda params: torch.nn.LeakyReLU(), "input_gen": lambda params: (torch.randn(params['N'], params['C'], params['H'], params['W']), {}), "input_gen_params": ["N", "C", "H", "W"], "fn_params": []},
        "sigmoid": {"fn": lambda params: torch.sigmoid, "input_gen": lambda params: (torch.randn(params['N'], params['C'], params['H'], params['W']), {}), "input_gen_params": ["N", "C", "H", "W"], "fn_params": []},
    "elem_sigmoid": {"fn": lambda params: torch.sigmoid, "input_gen": lambda params: (torch.randn(params['N'], params['C'], params['H'], params['W']), {}), "input_gen_params": ["N", "C", "H", "W"], "fn_params": []},
       "elem_tanh": {"fn": lambda params: torch.tanh, "input_gen": lambda params: (torch.randn(params['N'], params['C'], params['H'], params['W']), {}), "input_gen_params": ["N", "C", "H", "W"], "fn_params": []},
    "batch_norm": {"fn": lambda params: nn.BatchNorm2d(params['C']), "input_gen": lambda params: (torch.randn(params['N'], params['C'], params['H'], params['W']), {}), "input_gen_params": ["N", "C", "H", "W"], "fn_params": ["C"]},
        "relu2d": {"fn": lambda params: torch.relu, "input_gen": lambda params: (torch.randn(params['N'], params['C']), {}), "fn_params": [], "input_gen_params": ["N", "C"]},
    "elem_tanh2d": {"fn": lambda params: torch.tanh, "input_gen": lambda params: (torch.randn(params['N'], params['C']), {}), "fn_params": [], "input_gen_params": ["N", "C"]},
    "elem_pow2d": {"fn": lambda params: torch.pow, "input_gen": lambda params: (torch.randn(params['N'], params['C']), {}), "fn_params": [], "input_gen_params": ["N", "C"]},
    "elem_ceil2d": {"fn": lambda params: torch.ceil, "input_gen": lambda params: (torch.randn(params['N'], params['C']), {}) , "fn_params": [], "input_gen_params": (["N", "C"], {})},
    "transpose2d": {"fn": lambda params: torch.transpose, "input_gen": lambda params: (torch.randn(params['N'], params['C']), {"const_inputs": params['axis']}) , "fn_params": [], "input_gen_params": (["N", "C"], {"const_inputs": "axis"})},
"tensor_transpose2d": {"fn": lambda params: torch.transpose, "input_gen": lambda params: (torch.randn(params['N'], params['C']), {"const_inputs": params['axis']}) , "fn_params": [], "input_gen_params": (["N", "C"], {"const_inputs": "axis"})},
    "reduce_mean2d": {"fn": lambda params: torch.mean, "input_gen": lambda params: (torch.randn(params['N'], params['C']), {"keepdim": params["keepdim"], "const_inputs": (params['axis'],)}) , "fn_params": [], "input_gen_params": (["N", "C"], {"keepdim": "keepdim", "const_inputs": "axis"})},
    "reduce_min2d": {"fn": lambda params: torch.min, "input_gen": lambda params: (torch.randn(params['N'], params['C']),  {"keepdim": params["keepdim"], "const_inputs": (params['axis'],), "extraction_args": {"input_names": ["input"],"output_names": ["output"]}}) , "fn_params": [], "input_gen_params": (["N", "C"], {"keepdim": "keepdim", "const_inputs": "axis"})},
    "max_pool": {"fn": lambda params: nn.MaxPool2d(params['KH'], stride=params['stride'], padding=params['pad']), "input_gen": lambda params: (torch.randn(params['N'], params['C'], params['IH'], params['IW']), {}),
                 "fn_params": ["KH", "stride", "pad"], "input_gen_params": ["N", "C", "IH", "IW"]},
    "maxpool": {"fn": lambda params: nn.MaxPool2d(params['KH'], stride=params['stride'], padding=params['pad']), "input_gen": lambda params: (torch.randn(params['N'], params['C'], params['IH'], params['IW']), {}),
                "fn_params": ["KH", "stride", "pad"], "input_gen_params": ["N", "C", "IH", "IW"]},
    "avg_pool": {"fn": lambda params: nn.AvgPool2d(params['KH'], stride=params['stride'], padding=params['pad']), "input_gen": lambda params: (torch.randn(params['N'], params['C'], params['IH'], params['IW']), {}),
                    "fn_params": ["KH", "stride", "pad"], "input_gen_params": ["N", "C", "IH", "IW"]},
    "global_avg_pool": {"fn": lambda params: nn.AdaptiveAvgPool2d((1, 1)), "input_gen": lambda params: (torch.randn(params['N'], params['C'], params['IH'], params['IW']), {}),
                    "fn_params": [], "input_gen_params": ["N", "C", "IH", "IW"]},
    "depthwise_conv":  {"fn": lambda params: nn.Conv2d(in_channels=params['C'], out_channels=params['C'],
                                               kernel_size=params['KH'], stride=params['stride'], padding=params['pad'],
                                               groups=params['C'], bias=False),
                    "input_gen": lambda params: (torch.randn(params['N'], params['C'], params['IH'], params['IW']), {}),
                    "fn_params": ['IC', 'OC', 'KH', 'stride', 'pad'],
                    "input_gen_params": ['N', 'IC', 'IH', 'IW']},
}


def get_image_from_url(url, size=None):
    import requests
    from PIL import Image
    from io import BytesIO
    from torchvision import transforms

    data = requests.get(url)
    image = Image.open(BytesIO(data.content)).convert("RGB")

    if size is None:
        size = (300, 200)
    image = image.resize(size, Image.BILINEAR)

    to_tensor = transforms.ToTensor()
    return to_tensor(image)


def get_test_images():
    image_url = "http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg"
    image = get_image_from_url(url=image_url, size=(100, 320))

    image_url2 = "https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png"
    image2 = get_image_from_url(url=image_url2, size=(250, 380))

    images = [image]
    test_images = [image2]
    return images, test_images

def contains_cl(args):
    for t in args:
        if isinstance(t, torch.Tensor):
            if t.is_contiguous(memory_format=torch.channels_last) and not t.is_contiguous():
                return True
        elif isinstance(t, list) or isinstance(t, tuple):
            if contains_cl(list(t)):
                return True
    return False


def print_inputs(args, indent=''):
    for t in args:
        if isinstance(t, torch.Tensor):
            print(indent, t.stride(), t.shape, t.device, t.dtype)
        elif isinstance(t, list) or isinstance(t, tuple):
            print(indent, type(t))
            print_inputs(list(t), indent=indent + '    ')
        else:
            print(indent, t)


def check_wrapper(fn):
    name = fn.__name__

    def check_cl(*args, **kwargs):
        was_cl = contains_cl(args)
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            print("`{}` inputs are:".format(name))
            print_inputs(args)
            print('-------------------')
            raise e
        failed = False
        if was_cl:
            if isinstance(result, torch.Tensor):
                if result.dim() == 4 and not result.is_contiguous(memory_format=torch.channels_last):
                    print("`{}` got channels_last input, but output is not channels_last:".format(name),
                          result.shape, result.stride(), result.device, result.dtype)
                    failed = True
        if failed and True:
            print("`{}` inputs are:".format(name))
            print_inputs(args)
            raise Exception(
                'Operator `{}` lost channels_last property'.format(name))
        return result
    return check_cl

old_attrs = dict()

def attribute(m):
    old_attrs[m] = dict()
    for i in dir(m):
        e = getattr(m, i)
        exclude_functions = ['is_cuda', 'has_names', 'numel',
                             'stride', 'Tensor', 'is_contiguous', '__class__']
        if i not in exclude_functions and not i.startswith('_') and '__call__' in dir(e):
            try:
                old_attrs[m][i] = e
                setattr(m, i, check_wrapper(e))
            except Exception as e:
                print(i)
                print(e)


def add_value_info_for_constants(model : onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)

    return add_const_value_infos_to_graph(model.graph)

def create_lenet(optimize_model, training_mode, convert_data_format, to_polymath):
    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                                   kernel_size=5, stride=1, padding=0, bias=True)
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                                   kernel_size=5, stride=1, padding=0, bias=True)
            self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                                   kernel_size=5, stride=1, padding=0, bias=True)
            self.linear1 = nn.Linear(120, 84)
            self.linear2 = nn.Linear(84, 10)
            self.tanh = nn.Tanh()
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        def forward(self, x):
            x = self.conv1(x)
            x = self.tanh(x)
            x = self.avgpool(x)
            x = self.conv2(x)
            x = self.tanh(x)
            x = self.avgpool(x)
            x = self.conv3(x)
            x = self.tanh(x)
            x = torch.flatten(x, 1)
            x = self.linear1(x)
            x = self.tanh(x)
            x = self.linear2(x)
            return x
    model = LeNet()
    input_var = torch.randn(1, 1, 32, 32)
    output = model(input_var)
    model.eval()
    convert_torch_model(input_var, model, "lenet", optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)


def create_custom_conv(optimize_model, training_mode, convert_data_format, to_polymath, input_shape, oc, ksize, stride, pad,
                       name=None):
    n, ic, h, w = input_shape
    class CustomConv(nn.Module):
        def __init__(self):
            super(CustomConv, self).__init__()

            self.conv = nn.Conv2d(in_channels=ic, out_channels=oc,
                                   kernel_size=ksize, stride=stride, padding=pad, bias=True)

        def forward(self, x):
            x = self.conv(x)
            return x
    model = CustomConv()
    input_var = torch.randn(n, ic, h, w)
    output = model(input_var)
    model.eval()
    if name is None:
        name = "custom_conv"
    convert_torch_model(input_var, model, name, optimize_model, training_mode, to_polymath,
                        convert_data_format=convert_data_format)


def create_custom_matmul(optimize_model, training_mode, convert_data_format, to_polymath, M, N, P, include_bias=False):
    class CustomMatmul(nn.Module):
        def __init__(self):
            super(CustomMatmul, self).__init__()

            self.mmul = nn.Linear(N,  P, bias=include_bias)

        def forward(self, x):
            x = self.mmul(x)
            return x
    model = CustomMatmul()
    input_var = torch.randn(M, N)
    output = model(input_var)
    model.eval()
    convert_torch_model(input_var, model, "custom_matmul", optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)

def create_custom_layer(layer_name, params, optimize_model, convert_data_format, training_mode, to_polymath, fname=None):

    class CustomLayer(nn.Module):
        def __init__(self, kwargs):
            if "const_inputs" in kwargs:
                assert isinstance(kwargs['const_inputs'], tuple)
                self.const_inputs = kwargs.pop('const_inputs')
            else:
                self.const_inputs = tuple([])
            self.kwargs = kwargs

            super(CustomLayer, self).__init__()
            self.layer = LAYER_UTILS[layer_name]['fn'](params)

        def forward(self, *args):
            x = self.layer(*(args + self.const_inputs), **self.kwargs)
            return x
    input_var, kwargs = LAYER_UTILS[layer_name]['input_gen'](params)

    if "opset" in kwargs:
        opset = kwargs.pop("opset")
    else:
        opset = _onnx_opset_version

    if "extraction_args" in kwargs:
        extraction_args = kwargs.pop("extraction_args")
    else:
        extraction_args = None

    model = CustomLayer(kwargs)


    if not isinstance(input_var, tuple):
        input_var = (input_var,)
    output = model(*input_var)
    model.eval()
    if fname is None:
        fname = f"custom_{layer_name}"
    convert_torch_model(input_var, model, fname, optimize_model, training_mode, to_polymath,
                        convert_data_format=convert_data_format,
                        opset=opset,
                        extraction_args=extraction_args)


def create_custom_multi_layer(layer_sequence, all_params, optimize_model, convert_data_format, training_mode, to_polymath, fname=None):
    from collections import defaultdict
    assert isinstance(all_params, dict)
    class CustomLayerSeq(nn.Module):
        def __init__(self, kwargs):
            self.const_inputs = {}
            layer_nums = defaultdict(int)
            for lname in layer_sequence:
                l = f"{lname}{layer_nums[lname]}"
                layer_nums[lname] += 1
                if "const_inputs" in kwargs[l]:
                    assert isinstance(kwargs[l]['const_inputs'], tuple)
                    self.const_inputs[l] = kwargs[l].pop('const_inputs')
                else:
                    self.const_inputs[l] = tuple([])
            self.kwargs = kwargs

            super(CustomLayerSeq, self).__init__()
            self.seq = torch.nn.Sequential()
            layer_nums = defaultdict(int)
            for lname in layer_sequence:
                l = f"{lname}{layer_nums[lname]}"
                layer_nums[lname] += 1
                if l == "gemm":
                    self.seq.add_module("flatten", nn.Flatten())
                self.seq.add_module(l, LAYER_UTILS[lname]['fn'](all_params[l]))

        def forward(self, *args):
            return self.seq(*args)

    # Only need the first layer args
    input_var = None
    all_kwargs = {}
    opset = _onnx_opset_version
    extraction_args = None

    layer_nums = defaultdict(int)
    for lname in layer_sequence:
        l = f"{lname}{layer_nums[lname]}"
        layer_nums[lname] += 1
        if input_var is None:
            input_var, all_kwargs[l] = LAYER_UTILS[lname]['input_gen'](all_params[l])
        else:
            _, all_kwargs[l] = LAYER_UTILS[lname]['input_gen'](all_params[l])
        if "opset" in all_kwargs[l]:
            opset = all_kwargs[l].pop("opset")

        if "extraction_args" in all_kwargs[l]:
            extraction_args = all_kwargs[l].pop("extraction_args")
        else:
            extraction_args = None

    model = CustomLayerSeq(all_kwargs)

    if not isinstance(input_var, tuple):
        input_var = (input_var,)
    output = model.seq(*input_var)
    model.eval()

    if fname is None:
        layer_seq_name = "_".join(layer_sequence)
        fname = f"custom_{layer_seq_name}"
    convert_torch_model(input_var, model, fname, optimize_model, training_mode, to_polymath,
                        convert_data_format=convert_data_format,
                        opset=opset,
                        extraction_args=extraction_args)


def create_custom_gemm(optimize_model, training_mode, convert_data_format, to_polymath, M, N, P, fname=None):

    class CustomGemm(nn.Module):
        def __init__(self):
            super(CustomGemm, self).__init__()
            self.gemm = nn.Linear(N,  P, bias=True)

        def forward(self, x):
            x = self.gemm(x)
            return x
    model = CustomGemm()
    input_var = torch.randn(M, N)
    output = model(input_var)
    model.eval()
    if fname is None:
        fname = "custom_gemm"
    convert_torch_model(input_var, model, fname, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)


def create_lenet_bn(optimize_model, training_mode, convert_data_format, to_polymath):
    class LeNetBN(nn.Module):
        def __init__(self):
            super(LeNetBN, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                                   kernel_size=5, stride=1, padding=0, bias=True)
            self.bn1 = nn.BatchNorm2d(6)
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                                   kernel_size=5, stride=1, padding=0, bias=True)
            self.bn2 = nn.BatchNorm2d(16)

            self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                                   kernel_size=5, stride=1, padding=0, bias=True)
            self.bn3 = nn.BatchNorm2d(120)
            self.linear1 = nn.Linear(120, 84)
            self.linear2 = nn.Linear(84, 10)
            self.relu = nn.ReLU()
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.max_pool(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.max_pool(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)

            x = torch.flatten(x, 1)
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    model = LeNetBN()
    input_var = torch.randn(3, 1, 32, 32)
    output = model(input_var)
    model.eval()
    convert_torch_model(input_var, model, "lenet_bn", optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)


def create_resnet18(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = models.resnet18(pretrained=not training_mode)
    input_var = torch.randn(batch_size, 3, 224, 224)
    model_name = "resnet18"
    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"
    if not training_mode:
        output = model(input_var)
        model.eval()
    else:
        model_name = f"{model_name}_train"
    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)


def create_vgg16(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = models.vgg16(pretrained=not training_mode)
    input_var = torch.randn(batch_size, 3, 224, 224)
    model_name = "vgg16"
    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"
    if not training_mode:
        model.eval()
        output = model(input_var)
    else:
        model_name = f"{model_name}_train"
    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)

def create_inception(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = models.inception_v3(pretrained=not training_mode)
    input_var = torch.randn(batch_size, 3, 299, 299)
    model_name = "inceptionv3"
    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"
    if not training_mode:
        model.eval()
        output = model(input_var)
    else:
        model_name = f"{model_name}_train"
    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)

def create_mobilenet(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = models.mobilenet_v2(pretrained=not training_mode)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    input_var = torch.randn(batch_size, 3, 224, 224)
    model_name = "mobilenetv2"
    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"
    if not training_mode:
        model.eval()
        output = model(input_var)
    else:
        model_name = f"{model_name}_train"
    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)

def create_alexnet(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = models.alexnet(pretrained=not training_mode)
    input_var = torch.randn(batch_size, 3, 224, 224)
    model_name = "alexnet"
    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"
    if not training_mode:
        model.eval()
        output = model(input_var)
    else:
        model_name = f"{model_name}_train"
    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)

def create_efficientnet(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = models.alexnet(pretrained=not training_mode)
    input_var = torch.randn(batch_size, 3, 224, 224)
    model_name = "efficientnet"
    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"
    if not training_mode:
        model.eval()
        output = model(input_var)
    else:
        model_name = f"{model_name}_train"
    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)

def create_resnet50(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = models.resnet50(pretrained=not training_mode)
    input_var = torch.randn(batch_size, 3, 224, 224)
    model_name = "resnet50"

    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"

    if not training_mode:
        output = model(input_var)
        model.eval()
    else:
        model_name = f"{model_name}_train"



    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)

def create_resnet101(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = models.resnet101(pretrained=not training_mode)
    input_var = torch.randn(batch_size, 3, 224, 224)
    model_name = "resnet101"

    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"

    if not training_mode:
        output = model(input_var)
        model.eval()
    else:
        model_name = f"{model_name}_train"



    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)

def _make_empty_samples(N, C, H, W, training=False):

    img, other = get_test_images()
    t = Targets(boxes=torch.rand(0, 4), labels=torch.tensor([]).to(dtype=torch.int64),
                masks=torch.rand(0, H, W))

    return img, [t._asdict()]

def _make_mrcnn_samples():
    img, other = get_test_images()
    dummy_image = [torch.ones(3, 100, 100) * 0.3]
    return img, other, dummy_image


def create_maskrcnn(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=not training_mode, min_size=200, max_size=300)

    N, C, H, W = 1, 1, 300, 300
    # inputs = _make_empty_samples(N, C, H, W, training=training_mode)
    images, test_images, dummy_image = _make_mrcnn_samples()
    model_name = "mask_rcnn_vision"


    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"

    if not training_mode:
        model.eval()
        model(images)
        input_var = [(images,), (test_images,), (dummy_image,)]
    else:
        model_name = f"{model_name}_train"
        model.train()
        input_var = [(images,)]

    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)


class RPNModule(torch.nn.Module):
    def __init__(self):
        super(RPNModule, self).__init__()

        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        out_channels = 256
        rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_pre_nms_top_n = dict(training=2000, testing=1000)
        rpn_post_nms_top_n = dict(training=2000, testing=1000)
        rpn_nms_thresh = 0.7
        rpn_score_thresh = 0.0

        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

    def forward(self, images, features):
        images = ImageList(images, [i.shape[-2:] for i in images])
        return self.rpn(images, features)

def create_maskrcnn_part(part_name, optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=not training_mode, min_size=200, max_size=300)

    images, test_images, dummy_image = _make_mrcnn_samples()
    model_name = f"mask_rcnn_vision_{part_name}"

    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"

    if not training_mode:
        model.eval()
        transformed_input, targets = model.transform(images, None)
        features = model.backbone(transformed_input.tensors)

        proposals, proposal_losses = model.rpn(transformed_input, features, targets)
        detections, detector_losses = model.roi_heads(features, proposals, transformed_input.image_sizes, targets)
    else:
        model.eval()
        transformed_input, targets = model.transform(images, None)
        features = model.backbone(transformed_input.tensors)
        proposals, proposal_losses = model.rpn(transformed_input, features, targets)
        detections, detector_losses = model.roi_heads(features, proposals, transformed_input.image_sizes, targets)
    f = io.BytesIO()
    store_path = f"{CWD}/full_dnns/{model_name}.onnx"

    if part_name == "backbone":
        inputs = (transformed_input.tensors,)
        torch.onnx.export(model.backbone,  # model being run
                          inputs,  # model input (or a tuple for multiple inputs)
                          f,  # where to save the model (can be a file or file-like object)
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=["images_tensors"],
                          output_names=["feature_pool", "feature0", "feature1", "feature2", "feature3"],
                          opset_version=_onnx_opset_version,
                          verbose=False,
                          )

    elif part_name == "rpn":
        rpn_model = RPNModule()
        rpn_model.eval()
        inputs = (transformed_input.tensors, features) + ({},)
        torch.onnx.export(rpn_model,  # model being run
                          inputs,  # model input (or a tuple for multiple inputs)
                          f,
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          opset_version=_onnx_opset_version,
                          verbose=False,
                          )
    elif part_name == "roi":
        inputs = (features, proposals, transformed_input.image_sizes)
        torch.onnx.export(model.roi_heads,  # model being run
                          inputs,  # model input (or a tuple for multiple inputs)
                          f,
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          opset_version=_onnx_opset_version,
                          verbose=False,
                          )
    model_proto = onnx.ModelProto.FromString(f.getvalue())

    add_value_info_for_constants(model_proto)
    model_proto = onnx.shape_inference.infer_shapes(model_proto)
    with open(store_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    # convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)
    #


def _print_nodes(graph):
    nodes = []
    for n in graph.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                print(f"Found graph attribute for {n.op_type} - {n.name}\n"
                      f"Attribute name: {a.name}")
                nodes += _print_nodes(a.g)
        nodes.append(n.op_type)
    return nodes

def print_nodes(model_proto):
    nodes = _print_nodes(model_proto.graph)
    num_unique_nodes = len(list(set(nodes)))
    num_nodes_total = len(list(nodes))
    all_node_names = list(set(nodes))

    print(f"All node names: {all_node_names}\n"
          f"Unique operations: {num_unique_nodes}\n"
          f"Total Operations: {num_nodes_total}")

def convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath,
                        convert_data_format=False, out_dir=None,
                        opset=_onnx_opset_version,
                        verbose=False,
                        extraction_args=None):
    f = io.BytesIO()
    mode = torch.onnx.TrainingMode.TRAINING if training_mode else torch.onnx.TrainingMode.EVAL
    filepath = f"{CWD}/models/{model_name}.onnx"

    if 'mask_rcnn' not in model_name:
        torch.onnx.export(model,  # model being run
                          input_var,  # model input (or a tuple for multiple inputs)
                          f,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          keep_initializers_as_inputs=True,
                          training=mode,
                          verbose=False,
                          input_names=['input'],  # the model's input names
                          output_names=['output'],
                          )
    else:
        model.eval()
        # input_var = [(input_var,)]
        if isinstance(input_var[0][-1], dict):
            input_var = input_var[0] + ({},)
        else:
            input_var = input_var[0]

        dynamic_axes = {"images_tensors": [0, 1, 2], "boxes": [0, 1], "labels": [0],
                                        "scores": [0], "masks": [0, 1, 2]}
        torch.onnx.export(model,  # model being run
                          input_var,  # model input (or a tuple for multiple inputs)
                          f,  # where to save the model (can be a file or file-like object)
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          # training=mode,
                          input_names=["images_tensors"],
                          output_names=["boxes", "labels", "scores", "masks"],
                          dynamic_axes=dynamic_axes,
                          opset_version=opset,
                          verbose=False,
                          # export_params=True,  # store the trained parameter weights inside the model file
                          # keep_initializers_as_inputs=True,
                          # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN,
                          )

    model_proto = onnx.ModelProto.FromString(f.getvalue())

    if extraction_args is not None:
        assert "input_names" in extraction_args and "output_names" in extraction_args
        input_names = extraction_args.pop("input_names")
        output_names = extraction_args.pop("output_names")
        with open(filepath, "wb") as f:
            f.write(model_proto.SerializeToString())
        onnx.utils.extract_model(filepath, filepath, input_names, output_names)
        model_proto = onnx.load(filepath)

    if verbose:
        print_nodes(model_proto)
    onnx.checker.check_model(model_proto)
    add_value_info_for_constants(model_proto)
    model_proto = onnx.shape_inference.infer_shapes(model_proto)

    if optimize_model:
        model_proto, check = simplify(model_proto)
        assert check
    model_proto = update_node_names(model_proto)
    model_proto = update_edge_names(model_proto)
    with open(filepath, "wb") as f:
        f.write(model_proto.SerializeToString())

    if to_polymath:
        graph = pm.from_onnx(filepath)
        pm.pb_store(graph, f"{CWD}/models/srdfg")



def optimize_bert_onnx(to_polymath, batch_size=1):
    from collections import defaultdict
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    load_path = f"{MODEL_DIR}/bertsquad-12-opt-trimmed.onnx"
    store_path = f"{MODEL_DIR}/bertsquad-12-opt-trimmed-xposed.onnx"
    model_init = onnx.load(load_path)
    added_xpose = 0
    for i in range(len(model_init.graph.node)):
        n = model_init.graph.node[i]
        if n.op_type == "ReduceMean":
            axes_val = get_attribute(n, 'axes')
            print(f"Here: {axes_val}")
            # xpose_node = helper.make_node(
            #     'Transpose',
            #     inputs=[n.input[0]],
            #     outputs=[f'added_expose{added_xpose}_'],
            #     perm =
            # )
            # added_xpose += 1

    # with open(store_path, "wb") as f:
    #     f.write(model_init.SerializeToString())
    # providers = ['CPUExecutionProvider']
    # inputs = {}
    # inputs['bert/embeddings/Reshape_1:0'] = torch.randn(1, 256, 768).numpy()
    # inputs['bert/embeddings/one_hot:0'] = torch.randn(256, 2).numpy()
    # inputs['bert/encoder/layer_0/attention/self/mul_1:0'] = torch.randn(1, 1, 256, 256).numpy()
    # sess = ort.InferenceSession(load_path, providers=ort.get_available_providers())
    # result = sess.run(None, inputs)[0]
    #
    # new_sess = ort.InferenceSession(load_path, providers=ort.get_available_providers())
    # new_result = new_sess.run(None, inputs)[0]
    # np.testing.assert_allclose(new_result, result)


    # inpt_shapes = {"unique_ids_raw_output___9:0": (batch_size,),
    #           "segment_ids:0": (batch_size,256),
    #           "input_mask:0": (batch_size,256),
    #           "input_ids:0": (batch_size,256)
    #           }
    # out_shapes = {"unstack:1": (batch_size,256),
    #               "unstack:0": (batch_size,256),
    #               "unique_ids:0": (batch_size,)
    #               }
    # optimize_onnx(load_path, store_path, inpt_shapes, out_shapes, to_polymath)

def set_bert_shapes(batch_size=1):
    bert_name = f"bert-base-case"
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    load_path = f"{MODEL_DIR}/{bert_name}.onnx"
    store_path = f"{MODEL_DIR}/{bert_name}-opt.onnx"
    inpt_shapes = {"bert/embeddings/Reshape_1:0": (batch_size, 256, 768),
              "bert/embeddings/one_hot:0": (256, 2),
              "bert/encoder/layer_0/attention/self/mul_1:0": (batch_size,1, 256, 256),
              }
    out_shapes = {"BiasAdd:0": (256, 2),
                  }
    optimize_onnx(load_path, store_path, inpt_shapes, out_shapes, False)



def optimize_yolo_onnx(to_polymath, batch_size=1):
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    load_path = f"{MODEL_DIR}/yolov3.onnx"
    store_path = f"{MODEL_DIR}/yolov3-opt2.onnx"
    # n_candidates = 80
    # n_box = 40
    n_candidates = 'n_candidates'
    n_box = 'n_box'
    inpt_shapes = {"input_1": (batch_size, 3, 416, 416),
              "image_shape": (batch_size, 2),
              }
    out_shapes = {
                "yolonms_layer_1/ExpandDims_1:0": (batch_size, n_candidates, 4),
                  "yolonms_layer_1/ExpandDims_3:0": (batch_size, 80, n_candidates),
                  "yolonms_layer_1/concat_2:0": (n_box, 3)
                  }
    optimize_onnx(load_path, store_path, inpt_shapes, out_shapes, to_polymath)

def extract_static_yolo():
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    load_path = f"{MODEL_DIR}/yolov3-opt-static.onnx"
    store_path = f"{MODEL_DIR}/yolov3-opt-static1.onnx"


    input_names = ["input_1"]
    output_names = ["convolution_output_", "convolution_output1_", "convolution_output2_"]

    onnx.utils.extract_model(load_path, store_path, input_names, output_names)

def extract_bert_layer():
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    load_path = f"{MODEL_DIR}/bertsquad-12-opt-trimmed.onnx"
    store_path = f"{MODEL_DIR}/bertsquad-12-opt-trimmed-fuse-tester.onnx"


    input_names = ["bert/encoder/layer_0/intermediate/dense/mul_3:0",
                   "bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1:0",
                   ]
    output_names = ["bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1:0"]

    onnx.utils.extract_model(load_path, store_path, input_names, output_names)

def extract_lenet():
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    load_path = f"{MODEL_DIR}/lenet-opt.onnx"
    store_path = f"{MODEL_DIR}/lenet-opt-trimmed.onnx"
    input_names = ["import/Placeholder:0"]
    output_names = ["import/conv3/Relu:0"]
    onnx.utils.extract_model(load_path, store_path, input_names, output_names, check_model=False)

def rename_yolo_ops():
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    load_path = f"{MODEL_DIR}/yolov3-opt-static.onnx"
    store_path = f"{MODEL_DIR}/yolov3-opt-static1.onnx"
    model = onnx.load(load_path)
    model = rename_out_edges(model, 'convolution_output')
    with open(store_path, "wb") as f:
        f.write(model.SerializeToString())


def remove_softmax_efficientnet():
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    load_path = f"{MODEL_DIR}/efficientnet-lite4-new-opt.onnx"
    store_path = f"{MODEL_DIR}/efficientnet-lite4-new-opt-no-softmax.onnx"
    input_names = ['efficientnet-lite4/model/stem/conv2d/Conv2D__5:0']

    output_names = ['efficientnet-lite4/model/head/dense/BiasAdd:0']
    onnx.utils.extract_model(load_path, store_path, input_names, output_names)


def trim_gpt2():
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    load_path = f"{MODEL_DIR}/gpt2-opt.onnx"
    store_path = f'{MODEL_DIR}/gpt2-trimmed-opt.onnx'
    input_names = [
        'onnx::Add_204'
    ]
    model_proto = onnx.load(load_path)
    output_names = [o.name for o in model_proto.graph.output]

    onnx.utils.extract_model(load_path, store_path, input_names, output_names)



def trim_bert():
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    load_path = f"{MODEL_DIR}/bertsquad-12-opt1.onnx"
    store_path = f"{MODEL_DIR}/bertsquad-12-opt2.onnx"
    store_path2 = f"{MODEL_DIR}/bertsquad-12-opt3.onnx"

    input_names = [
        # Result from reshape x2, gather, reshape
        'bert/embeddings/Reshape_1:0',
        # one-hot encoded segment ids
        'bert/embeddings/one_hot:0',
        # All of these are the same value, but redundant computation is removed
        'bert/encoder/layer_0/attention/self/mul_1:0',
        'bert/encoder/layer_1/attention/self/mul_1:0',
        'bert/encoder/layer_2/attention/self/mul_1:0',
        'bert/encoder/layer_3/attention/self/mul_1:0',
        'bert/encoder/layer_4/attention/self/mul_1:0',
        'bert/encoder/layer_5/attention/self/mul_1:0',
        'bert/encoder/layer_6/attention/self/mul_1:0',
        'bert/encoder/layer_7/attention/self/mul_1:0',
        'bert/encoder/layer_8/attention/self/mul_1:0',
        'bert/encoder/layer_9/attention/self/mul_1:0',
        'bert/encoder/layer_10/attention/self/mul_1:0',
        'bert/encoder/layer_11/attention/self/mul_1:0',
        #
    ]
    output_names = ['BiasAdd:0']
    # onnx.utils.extract_model(load_path, store_path, input_names, output_names)
    replacements = input_names[3:]
    model = onnx.load(store_path)
    for n in model.graph.node:
        for i in range(len(n.input)):
            inpt = n.input[i]
            if inpt in replacements:
                n.input[i] = input_names[2]

    onnx.checker.check_model(model)
    with open(store_path2, "wb") as f:
        f.write(model.SerializeToString())
    input_names = [
        # Result from reshape x2, gather, reshape
        'bert/embeddings/Reshape_1:0',
        # one-hot encoded segment ids
        'bert/embeddings/one_hot:0',
        # All of these are the same value, but redundant computation is removed
        'bert/encoder/layer_0/attention/self/mul_1:0',
    ]
    onnx.utils.extract_model(store_path2, store_path2, input_names, output_names)



def collect_unset_dims(model):
    dim_param_set = set()
    def init_dim_param_set(dim_param_set, value_infos) -> None:
        for info in value_infos:
            shape = info.type.tensor_type.shape
            for dim in shape.dim:
                if dim.HasField('dim_param'):
                    dim_param_set.add(dim.dim_param)  # type: ignore

    init_dim_param_set(dim_param_set, model.graph.input)  # type: ignore
    init_dim_param_set(dim_param_set, model.graph.output)  # type: ignore
    init_dim_param_set(dim_param_set, model.graph.value_info)  # type: ignore
    return list(dim_param_set)

def optimize_onnx(load_path, store_path, inpt_shapes, out_shapes, to_polymath, single_params=None):
    model = onnx.load(load_path)

    unset_params = collect_unset_dims(model)

    if len(unset_params) > 0 and inpt_shapes is None and out_shapes is None:
        assert single_params is not None and set(single_params.keys()) == set(unset_params)
        for i in model.graph.input:
            for dim in i.type.tensor_type.shape.dim:
                if dim.HasField('dim_param') and dim.dim_param in single_params:
                    if dim.HasField('dim_value') and dim.dim_value != single_params[dim.dim_param]:
                        raise RuntimeError(f"Invalid dim value")
                    dim.dim_value = single_params[dim.dim_param]

        for o in model.graph.output:
            for dim in o.type.tensor_type.shape.dim:
                if dim.HasField('dim_param') and dim.dim_param in single_params:
                    if dim.HasField('dim_value') and dim.dim_value != single_params[dim.dim_param]:
                        raise RuntimeError(f"Invalid dim value")
                    dim.dim_value = single_params[dim.dim_param]

    # print
    if inpt_shapes is not None and out_shapes is not None:
        model = update_model_dims.update_inputs_outputs_dims(model, inpt_shapes,
                                                         out_shapes)


    model = onnx.shape_inference.infer_shapes(model)

    model, check = simplify(model)

    # assert check
    # model = update_node_names(model)
    # model = update_edge_names(model)
    with open(store_path, "wb") as f:
        f.write(model.SerializeToString())

    if to_polymath:
        graph = pm.from_onnx(store_path)
        pm.pb_store(graph, f"{CWD}/full_dnns/")

def create_bert(optimize_model, training_mode, convert_data_format, to_polymath, batch_size=1):
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")

    input_keys = ["unique_ids_raw_output___9:0",
                  "segment_ids:0",
                  "input_mask:0",
                  "input_ids:0"]
    model = models.resnet18(pretrained=not training_mode)
    input_var = torch.randn(batch_size, 3, 224, 224)
    model_name = "resnet18"
    if batch_size != 1:
        model_name = f"{model_name}_batch{batch_size}"
    if not training_mode:
        output = model(input_var)
        model.eval()
    else:
        model_name = f"{model_name}_train"
    if convert_data_format:
        input_var = input_var.contiguous(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        out = model(input_var)
    convert_torch_model(input_var, model, model_name, optimize_model, training_mode, to_polymath, convert_data_format=convert_data_format)

    # output_keys = ["unstack:1","unstack:0","unique_ids:0"]
    # assert all([k in input_shapes for k in input_keys])
    # assert all([k in out_shapes for k in output_keys])
    #
    # load_path = f"{MODEL_DIR}/bertsquad-12.onnx"
    # store_path = f"{MODEL_DIR}/bertsquad-12-opt1.onnx"
    #
    # model = onnx.load(load_path)
    # static_shapes = update_model_dims.update_inputs_outputs_dims(model, input_shapes,
    #                                                          out_shapes)
    # model = onnx.shape_inference.infer_shapes(static_shapes)
    #
    # if optimize_model:
    #     model, check = simplify(model)
    #     assert check
    # model = update_node_names(model)
    # model = update_edge_names(model)
    # with open(store_path, "wb") as f:
    #     f.write(model.SerializeToString())
    #
    # if to_polymath:
    #     graph = pm.from_onnx(store_path)
    #     pm.pb_store(graph, f"{CWD}/full_dnns/")



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def fix_original_onnx_model(batch_size):
    from pathlib import Path
    CWD = Path(f"{__file__}").parent

    input_path = f"{CWD}/full_dnns/mask_rcnn_zoo_original.onnx"
    output_path = f"{CWD}/full_dnns/mask_rcnn_zoo_original_updated.onnx"

    model_proto = onnx.load(input_path)
    new_start_idx = -1
    target_idx = -1
    for idx, n in enumerate(model_proto.graph.node):
        if n.name == '0':
            assert n.op_type == 'Unsqueeze'
            target_idx = idx
        elif n.name == '2':
            new_start_idx = idx
        elif new_start_idx != -1 and target_idx != -1:
            break

    assert target_idx != -1 and new_start_idx != -1
    target_shape = (batch_size, 3, 800, 800)
    dummy_tensor = onnx.helper.make_tensor_value_info("dummy", 1, target_shape)
    model_proto.graph.input[0].type.tensor_type.shape.CopyFrom(dummy_tensor.type.tensor_type.shape)
    model_proto.graph.node[new_start_idx].input[0] = model_proto.graph.input[0].name
    del model_proto.graph.node[target_idx]

    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    onnx.checker.check_model(output_path)

def update_node_names(model_proto):
    non_digit_nodes = []
    for n in model_proto.graph.node:
        if not n.name.isdigit():
            non_digit_nodes.append(n.name)
    for n in model_proto.graph.node:
        if n.name.isdigit():
            new_name = f"{n.op_type}{n.name}"
            assert new_name not in non_digit_nodes
            n.name = new_name
    return model_proto

def update_edge_names(model_proto):
    node_name_map = {}
    INPUT_NAMES = ['A', 'B', 'D', 'X', 'W']
    OUTPUT_NAMES = ['Y', 'Z', 'C', 'H', 'P']

    for n in model_proto.graph.node:
        for idx, i in enumerate(n.input):
            if i not in node_name_map:
                if i.isdigit():
                    assert idx < len(INPUT_NAMES)
                    new_name = f"{n.name.lower()}_{i}{INPUT_NAMES[idx]}"
                else:
                    new_name = i
                node_name_map[i] = new_name

        for idx, o in enumerate(n.output):
            if o not in node_name_map:
                if o.isdigit():
                    assert idx < len(OUTPUT_NAMES)
                    new_name = f"{n.name.lower()}_{o}{OUTPUT_NAMES[idx]}"
                else:
                    new_name = o
                node_name_map[o] = new_name

    for v in model_proto.graph.value_info:
        assert v.name in node_name_map
        v.name = node_name_map[v.name]

    for i in model_proto.graph.initializer:
        assert i.name in node_name_map
        i.name = node_name_map[i.name]

    for n in model_proto.graph.node:
        n.input[:] = [node_name_map[i] for i in n.input]
        n.output[:] = [node_name_map[o] for o in n.output]

    for i in model_proto.graph.input:
        i.name = node_name_map[i.name]

    for o in model_proto.graph.output:
        o.name = node_name_map[o.name]

    return model_proto

def rename_out_edges(model_proto, base_string):
    node_name_map = {}

    for n in model_proto.graph.node:
        for idx, i in enumerate(n.input):
            if i not in node_name_map and base_string in i:
                new_name = f"{i}_"
                node_name_map[i] = new_name

        for idx, o in enumerate(n.output):
            if o not in node_name_map and base_string in o:
                new_name = f"{o}_"

                node_name_map[o] = new_name

    for v in model_proto.graph.value_info:
        if v.name in node_name_map:
            assert v.name in node_name_map
            v.name = node_name_map[v.name]

    for i in model_proto.graph.initializer:
        if i.name in node_name_map:
            i.name = node_name_map[i.name]

    for n in model_proto.graph.node:
        for idx in range(len(n.input)):
            if n.input[idx] in node_name_map:
                n.input[idx] = node_name_map[n.input[idx]]

        for idx in range(len(n.output)):
            if n.output[idx] in node_name_map:
                n.output[idx] = node_name_map[n.output[idx]]

    for i in model_proto.graph.input:
        if i.name in node_name_map:
            i.name = node_name_map[i.name]

    for o in model_proto.graph.output:
        if o.name in node_name_map:
            o.name = node_name_map[o.name]

    return model_proto

def simplify_mrcnn_zoo(batch_size=1):
    from pathlib import Path
    CWD = Path(f"{__file__}").parent
    initial_path = f"{CWD}/full_dnns/mask_rcnn_zoo_original_updated.onnx"
    filepath = f"{CWD}/full_dnns/mask_rcnn_zoo_test.onnx"
    model_proto = onnx.load(initial_path)
    model_proto = update_node_names(model_proto)
    model_proto = update_edge_names(model_proto)
    # onnx.checker.check_model(model_proto)
    # model_proto, check = simplify(model_proto)
    # assert check
    add_value_info_for_constants(model_proto)
    model_proto = onnx.shape_inference.infer_shapes(model_proto)
    print_nodes(model_proto)
    # #
    with open(filepath, "wb") as f:
        f.write(model_proto.SerializeToString())

def get_info(info_proto):
    """Extract the shape from a ValueInfoProto."""
    shape = []
    shape_name = []
    for dim in info_proto.type.tensor_type.shape.dim:
        name = dim.dim_param
        value = dim.dim_value
        if value is None or value == 0:
            shape_name.append(name)
        else:
            shape_name.append(value)
        shape.append(value)

    name = info_proto.name
    return name, shape, shape_name

def get_numpy(tensor_proto):
    """Grab data in TensorProto and convert to numpy array."""
    try:
        from onnx.numpy_helper import to_array
    except ImportError as e:
        raise ImportError("Unable to import onnx which is required {}".format(e))
    return to_array(tensor_proto)

def parse_array(tensor_proto):
    np_array = get_numpy(tensor_proto).reshape(tuple(tensor_proto.dims))
    return np_array

def collect_value_info(graph):
    node_info = {}

    for vi in graph.value_info:
        name, shape, shape_name = get_info(vi)
        node_info[name] = shape

    for init in graph.initializer:
        node_info[init.name] = parse_array(init).shape

    for inp in graph.input:
        name, shape, shape_name = get_info(inp)
        node_info[name] = shape

    for outp in graph.output:
        name, shape, shape_name = get_info(outp)
        node_info[name] = shape

    return node_info

def get_backbone_outputs(graph, node_info, node_output_map):
    num_max_pool = 0
    backbone_output_shape = [0, 256, 0, 0]
    backbone_kernel_shape = [256, 256, 3, 3]
    output_names = []
    for n in graph.node:
        if n.op_type == "MaxPool":
            if num_max_pool == 1:
                output_names.append(n.output[0])
                output_names.append(n.input[0])
            else:
                num_max_pool += 1
        elif n.op_type == "Conv" and node_output_map[n.input[0]][1] == 'Add':
            assert n.output[0] in node_info
            output_names.append(n.output[0])
    assert len(output_names) == 5
    return output_names

def get_rpn_outputs(graph, node_info, node_output_map):
    rpn_output_shape = [0, 4]
    output_names = []
    for n in graph.node:
        if n.op_type == "Gather":

            if n.input[0] in node_output_map and node_output_map[n.input[0]][1] == "Concat" and \
                    n.input[1] in node_output_map and node_output_map[n.input[1]][1] == "TopK" and \
                    node_info[n.output[0]] == rpn_output_shape:
                output_names = [n.output[0]]
                break
    assert len(output_names) == 1
    return output_names

def split_mrcnn(model_name, split_part):
    from pathlib import Path
    initial_path = f"{CWD}/full_dnns/{model_name}.onnx"
    filepath = f"{CWD}/full_dnns/{model_name}_simplified.onnx"
    node_output_map = {}
    model_proto = onnx.load(initial_path)
    # model_proto, _ = simplify(model_proto)

    # add_value_info_for_constants(model_proto)
    # model_proto = onnx.shape_inference.infer_shapes(model_proto)
    node_info = collect_value_info(model_proto.graph)
    for n in model_proto.graph.node:
        for o in n.output:
            node_output_map[o] = (n.name, n.op_type)
    node_output_map[model_proto.graph.input[0].name] = (None, None)

    if split_part == "backbone":
        input_path = initial_path
        output_path = f"{CWD}/full_dnns/{model_name}_{split_part}.onnx"
        input_names = ['image']
        output_names = get_backbone_outputs(model_proto.graph, node_info, node_output_map)
        # extract_model(input_path, output_path, input_names, output_names)
    elif split_part == "rpn":
        input_path = initial_path
        output_path = f"{CWD}/full_dnns/{model_name}_{split_part}.onnx"
        input_names = ['image']
        output_names = get_rpn_outputs(model_proto.graph, node_info, node_output_map)
        input_names += get_backbone_outputs(model_proto.graph, node_info, node_output_map)
        # extract_model(input_path, output_path, input_names, output_names)

def load_specs():
    specs = []
    SPEC_FILENAME = f"{CWD}/benchmark_specs_v2.csv"
    with open(SPEC_FILENAME) as f:
        spec_file = csv.DictReader(f, delimiter=",")
        for row in spec_file:
            specs.append(row)
    return specs

def run_covenant_benchmarks():
    bench_specs = load_specs()
    names = []
    for row in bench_specs:
        M = int(row['N'])
        N = int(row['M'])
        P = int(row['K'])
        onnx_name = f"{row['Model'].lower().replace(' ', '_').replace('-','_')}_{row['Layer'].lower().replace(' ', '_').replace('-','_')}"
        assert onnx_name not in names
        names.append(onnx_name)
        create_custom_gemm(True, True, False, False, M, N, P, onnx_name)
    print(names)

def main():
    # run_covenant_benchmarks()
    n = 1
    ic = 128
    oc = 128
    h = 30
    w = 30
    ksize = 3
    stride = 3
    pad = 0
    # assert (w + 2 * pad - ksize) % stride == 0, 'width does not work'
    input_shape = (n, ic, h, w)
    create_custom_conv(True, True, False, False, input_shape, oc, ksize, stride, pad, "cc_layer2")

    # # input_var = torch.randn(*input_shape)
    # # l = torch.nn.Conv2d(ic, oc, ksize, stride=stride, padding=pad)
    # # out = l(input_var)
    # M = 128
    # N = 1024
    # P = 2048
    # size_limit1 = 518750
    # size_limit = 4190000
    # total_size = (M*N) + (N*P) + (M*P)*4 + P*4
    # assert total_size <= size_limit, f"Total size {total_size} is greater than limit {size_limit}"
    # # optimize_model, training_mode, convert_data_format, to_polymath, input_shape, oc, ksize, stride, pad
    # create_custom_matmul(True, True, False, False, M, N, P, include_bias=True)
    # print(out.shape)
    # output = F.conv2d(input_var)
    # oc, ksize, stride, pad
    # create_lenet_bn(True, True, False, False)
    # create_lenet(True, True, False, False)
    # benchmark = "mask_rcnn_zoo_original_updated_simplified"
    # training_mode = False
    # data_format_convert = False
    # to_polymath = False
    # optimize_model = True
    # batch_size = 1
    # split_part = "backbone"
    # # split_mrcnn(benchmark, split_part)
    # create_maskrcnn_part(split_part, optimize_model, training_mode, data_format_convert, to_polymath)

def get_attribute(node, attr_name):
    for a in node.attribute:
        if a.name == attr_name:
            return onnx.helper.get_attribute_value(a)
    return None


def print_unique_model_layers(model_name):
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")

    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    model = onnx.load_model(model_path)
    layer_info = collect_value_info(model.graph)
    layers = {}

    def is_dw_conv(node):
        inpt_name = node.input[0]
        assert inpt_name in layer_info
        inpt_shape = layer_info[inpt_name]
        ic = inpt_shape[1]
        groups = get_attribute(node, "group")
        if groups is not None and groups == ic:
            return True
        return False
    dw_nodes = []
    for n in model.graph.node:
        if n.op_type.lower() == "conv" and is_dw_conv(n):
            lname = "DepthwiseConv"
            dw_nodes.append(n)
        else:
            lname = n.op_type

        if lname not in layers:
            layers[lname] = 1
        else:
            layers[lname] += 1
    csv_res = "\n".join(
        f"{op}, {num}" for op, num in layers.items()
    )
    print(f"Operation, Count")
    print(f"{csv_res}")


def optimize_graph(model_name, single_params=None):
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    load_path = f"{MODEL_DIR}/{model_name}.onnx"
    store_path = f"{MODEL_DIR}/{model_name}-opt.onnx"
    optimize_onnx(load_path, store_path, None, None, False, single_params=single_params)
    model = onnx.load(store_path)
    model = version_converter.convert_version(model,_onnx_opset_version)
    with open(store_path, "wb") as f:
        f.write(model.SerializeToString())
    return f"{model_name}-opt"

def is_dw_conv(node, layer_info):
    inpt_name = node.input[0]
    assert inpt_name in layer_info
    inpt_shape = layer_info[inpt_name]
    ic = inpt_shape[1]
    groups = get_attribute(node, "group")
    if groups is not None and groups == ic:
        return True
    return False

def get_layer_name(node, layer_info):
    if node.op_type.lower() == "conv" and is_dw_conv(node, layer_info):
        lname = "DepthwiseConv"
    else:
        lname = node.op_type
    return lname

def get_fusable_layer(graph, layer_name, input_node, layer_info):
    if isinstance(layer_name, list):
        out_layers = []
        outputs = []
        for l in layer_name:
            for n in graph.node:
                lname = get_layer_name(n, layer_info)
                if lname == l and input_node in n.input and n.output[0] not in outputs:
                    assert hasattr(n, "output") and len(n.output) == 1
                    out_layers.append({'output': n.output[0], 'layer': n})
                    outputs.append(n.output[0])
        if len(out_layers) == len(layer_name):
            return out_layers
    else:
        for n in graph.node:
            lname = get_layer_name(n, layer_info)
            if lname == layer_name and input_node in n.input:
                assert hasattr(n, "output") and len(n.output) == 1
                return [{'output': n.output[0], 'layer': n}]
    return None

def get_fused_nodes(graph, sequence, initial_layer, layer_info):
    # TODO: Make sure the output isnt used in multiple places
    assert len(initial_layer.output) == 1
    tgt_input = initial_layer.output[0]
    fdescriptors = []
    fdescriptors.append([{
        'layer': initial_layer,
        'output': tgt_input
    }])
    for l in sequence[1:]:
        fl = get_fusable_layer(graph, l, tgt_input, layer_info)
        if fl is None:
            return None
        else:
            assert isinstance(fl, list)
            tgt_input = fl[0]['output']
            fdescriptors.append(fl)
    return fdescriptors

def fuse_layers(model_name,
                all_fused_nodes,
                fusion_instances,
                layers,
                fusion_ops,
                layer_info, test_run=False):
    intermediate_nodes = []
    for layer_list in layers[:-1]:
        for l in layer_list:
            intermediate_nodes.append(l['output'])

    # intermediate_nodes = [l['output'] for l in layers[:-1]]
    fused_templates = []
    for layer_list in layers:
        for l in layer_list:
            fused_templates.append(l['layer'])
    # fused_templates = [l['layer'] for l in layers]
    layer_inputs = []
    for layer_list in layers:
        for l in layer_list:
            for i in l['layer'].input:
                if i not in intermediate_nodes:
                    layer_inputs.append(i)

    result = layers[-1][0]['output']
    all_fused_nodes['intermediate'] += intermediate_nodes
    all_fused_nodes['layers'] += fused_templates
    all_fused_nodes['fusion_inputs'] += layer_inputs
    all_fused_nodes['fusion_outputs'].append(result)
    flattened_ops = flatten_seq(fusion_ops)
    fusion_name = "_".join(flattened_ops)
    instance_name = f"{fusion_name}{fusion_instances[fusion_name]}"
    fusion_instances[fusion_name] += 1
    if not test_run:
        MODEL_DIR = Path(f"{Path(__file__).parent}/models")
        src_path = f"{MODEL_DIR}/{model_name}.onnx"
        model = onnx.load(src_path)
        dst_path = f"{MODEL_DIR}/{model_name}_{instance_name}.onnx"
        layer_inputs = [l for l in layer_inputs if l not in layer_info['initializer_names']]
        onnx.utils.extract_model(src_path, dst_path, layer_inputs, [result])

    return all_fused_nodes, fusion_instances

def flatten_seq(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten_seq(list_of_lists[0]) + flatten_seq(list_of_lists[1:])
    return list_of_lists[:1] + flatten_seq(list_of_lists[1:])

def fusion_generator(src_model, fusion_sequences, test_run=False):
    from collections import defaultdict
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    model_path = f"{MODEL_DIR}/{src_model}.onnx"
    model = onnx.load_model(model_path)

    node_list = list(model.graph.node)
    nidx = 0
    all_fused_nodes = {'layers': [],
                       'fusion_inputs': [],
                       'fusion_outputs': [],
                       'intermediate': []
                       }

    fusion_sequences = sorted(fusion_sequences, key=lambda x: len(x), reverse=True)
    fusion_starts = [s[0] for s in fusion_sequences]
    fusion_instances = defaultdict(int)

    layer_info = collect_value_info(model.graph)
    layer_info['initializer_names'] = [init.name for init in model.graph.initializer]


    while nidx < len(model.graph.node):
        n = node_list[nidx]

        if n in all_fused_nodes['layers'] or any([o in all_fused_nodes['fusion_inputs'] for o in n.output]):
            nidx += 1
            continue

        lname = get_layer_name(n, layer_info)
        if lname in fusion_starts:

            possible_fusions = [s for s in fusion_sequences if s[0] == lname]
            for pf in possible_fusions:
                fused_nodes = get_fused_nodes(model.graph, pf, n, layer_info)
                if fused_nodes is not None:
                    all_fused_nodes, fusion_instances = fuse_layers(src_model,
                                                                    all_fused_nodes,
                                                                    fusion_instances,
                                                                    fused_nodes,
                                                                    pf,
                                                                    layer_info, test_run=test_run)
                    break
        nidx += 1
    print(f"Fusion summary:")
    for k, v in fusion_instances.items():
        print(f"{k} - {v}")

def quantize_model(model_name):
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    load_path = f"{MODEL_DIR}/{model_name}.onnx"
    store_path = f"{MODEL_DIR}/{model_name}-int8.onnx"

def optimize_mobilenet():
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    load_path = f"{MODEL_DIR}/mobilenetv2-12.onnx"
    store_path = f"{MODEL_DIR}/mobilenetv2-12-opt.onnx"

if __name__ == "__main__":
    if sys.stdin and sys.stdin.isatty():


        argparser = argparse.ArgumentParser(description='ONNX Benchmark Generator')
        argparser.add_argument('-b', '--benchmark', required=True,
                               help='Name of the benchmark to create. One of "resnet18", "lenet')

        argparser.add_argument('-o', '--optimize_model', type=str2bool, nargs='?', default=True,
                               const=True, help='Optimize the model')

        argparser.add_argument('-t', '--training_mode', type=str2bool, nargs='?', default=False,
                               const=True, help='Whether or not the model is in training mode')

        argparser.add_argument('-bs', '--batch_size', type=int, default=1, help='The batch size for the model')

        argparser.add_argument('-df', '--data_format_convert', type=str2bool, nargs='?', default=False,
                               const=True, help='Whether or not the model is in training mode')


        argparser.add_argument('-pm', '--to_polymath', type=str2bool, nargs='?', default=False,
                               const=True, help='Whether or not the model should be converted to PolyMath')
        args = argparser.parse_args()
        if args.benchmark == "lenet":
            create_lenet(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath)
        elif args.benchmark == "lenetbn":
            create_lenet_bn(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath)
        elif args.benchmark == "bert":
            create_bert(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath)
        elif args.benchmark == "resnet18":
            create_resnet18(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath,
                            batch_size=args.batch_size)
        elif args.benchmark == "resnet50":
            create_resnet50(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath,
                            batch_size=args.batch_size)
        elif args.benchmark == "vgg16":
            create_vgg16(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath,
                            batch_size=args.batch_size)
        elif args.benchmark == "efficientnet":
            create_efficientnet(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath,
                            batch_size=args.batch_size)
        elif args.benchmark == "alexnet":
            create_alexnet(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath,
                            batch_size=args.batch_size)
        elif args.benchmark == "inception":
            create_inception(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath,
                            batch_size=args.batch_size)
        elif args.benchmark == "mobilenet":
            create_mobilenet(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath,
                            batch_size=args.batch_size)
        elif args.benchmark == "maskrcnn":
            create_maskrcnn(args.optimize_model, args.training_mode, args.data_format_convert, args.to_polymath,
                            batch_size=args.batch_size)
        elif args.benchmark == "maskrcnn_simplify":
            simplify_mrcnn_zoo(batch_size=args.batch_size)
        else:
            raise RuntimeError(f"Invalid benchmark supplied. Options are one of:\n"
                               f"\"lenet\", \"resnet18\".")
    else:
        # MODEL_DIR = Path(f"{Path(__file__).parent}/models")
        # load_path = f"{MODEL_DIR}/efficientnet-lite4-new-opt.onnx"
        # optimize_bert_onnx(False)
        # model = onnx.load(load_path)
        # print(dir(model))
        # print(model.producer_version)
        # remove_softmax_efficientnet()
        # extract_bert_layer()
        #
        # sequences = [['Conv', 'Relu'],
        #              ['Conv', 'Relu', 'MaxPool'],
        #              ['Conv', 'Add', 'Relu', 'GlobalAveragePool'],
        #              ['Conv', 'Add', 'Relu']]
        # # sequences = [['Conv', 'Relu', 'MaxPool'], ]
        # # sequences = [['Conv', 'Add'],
        # #              ['Conv', 'Clip', 'AveragePool'],
        # #              ['Conv', 'Clip', 'DepthwiseConv',],
        # #              ['Conv', 'Clip', 'DepthwiseConv', 'Clip',], ]
        # optimize_graph('efficientnet-lite4-new')
        # create_mobilenet(True, False, False, False,
        #                  batch_size=1)
        # optimize_graph('mobilenetv2-12', single_params={'batch_size': 1})
        # quantize_model('bertsquad-12-opt-trimmed')
        # optimize_bert_onnx(False)
        # set_bert_shapes()
        # name = "bertsquad-12-opt-trimmed"
        # # name = "bertsquad-12-opt-trimmed-fuse-tester"
        # sequences = [
        #                 ["MatMul", "Reshape", "Add", "Add",
        #                  "ReduceMean",
        #                  "Sub",
        #                  "Mul",
        #                  "ReduceMean",
        #                  "Add",
        #                  "Sqrt",
        #                  "Reciprocal",
        #                  "Mul", ["Mul", "Mul"], "Sub", "Add"],
        #              ["Gemm", "Add", "ReduceMean", "Sub", "Mul", "ReduceMean",
        #                                                    "Add", "Sqrt", "Reciprocal", "Mul", ["Mul", "Mul"],
        #                                                    "Sub", "Add"],
        #              ["MatMul", "Mul", "Add", "Softmax"],
        #              ["Gemm", "Reshape", "Transpose"], ["MatMul", "Transpose"],
        #              ["Gemm","Pow","Mul","Add", "Mul", "Tanh", "Add", "Mul", "Mul",]
        # ]
        # seq = [
        #         ["Gemm",
        #          "Add",
        #          "ReduceMean",
        #          "Sub",
        #          "Mul",
        #          "ReduceMean",
        #          "Add",
        #          "Sqrt",
        #          "Reciprocal",
        #          "Mul",
        #          ["Mul", "Mul",],
        #           "Sub",
        #          "Add"]
        # ]
        # model = "lenet"
        # optimize_graph(model)
        # extract_lenet()
        # name = "mobilenetv2-opt"
        # sequences = [['Conv', 'Add'],
        #              # ['Conv', 'Clip', 'AveragePool'],
        #              ['Conv', 'Clip', 'DepthwiseConv',],
        #              ['Conv', 'Clip', 'DepthwiseConv', 'Clip',], ]
        # fusion_generator(name, sequences, test_run=True)
        trim_gpt2()
        # model = "gpt2-opt"
        # create_custom_gemm(True, False, False, False, 8, 128, 128)
#