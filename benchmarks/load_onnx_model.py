import argparse
from onnx import ModelProto, GraphProto
import onnx
from onnx import helper
from pathlib import Path
import polymath as pm
import numpy as np

MODEL_DIR = Path(f"{Path(__file__).parent}/models")
LAYER_DIR = Path(f"{Path(__file__).parent}/layers")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_onnx_model(model_name):
    onnx_model = onnx.load(model_name)
    return onnx_model

def store_layer(layer_name, model_name, model: ModelProto = None):
    layer_path = f"{LAYER_DIR}/{layer_name}.onnx"
    model_path = f"{MODEL_DIR}/{model_name}.onnx"

    inputs = []
    outputs = []
    if model is None:
        model = onnx.load_model(model_path)

    for n in model.graph.node:
        if n.name == layer_name:
            inputs = n.input
            outputs = n.output
            break

    onnx.utils.extract_model(model_path, layer_path, inputs, outputs)


def convert_model_to_polymath(model_path):
    graph = pm.from_onnx(model_path, verbose=True)
    root_path = Path(model_path).parent
    pm.pb_store(graph, f"{root_path}/srdfg/")


def store_unique_model_layers(model_name, store_as_polymath=False, name_mapping=None):
    name_mapping = name_mapping or {}
    layers = {}
    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    model = onnx.load_model(model_path)
    tensor_dict = {i.name: i for i in model.graph.input}
    tensor_dict.update({o.name: o for o in model.graph.output})
    tensor_dict.update({v.name: v for v in model.graph.value_info})

    for n in model.graph.node:
        if n.op_type not in layers:
            inputs = n.input
            outputs = n.output
            if n.op_type == 'BatchNormalization':
                outputs = [n.output[0]]

            op_name = n.op_type.lower()
            if op_name in name_mapping:
                op_name = name_mapping[op_name]
            if n.op_type == "Conv":
                is_dw = False
                inp_shape = get_onnx_shape(tensor_dict, n.input[0])
                for a in n.attribute:
                    if a.name == "group" and helper.get_attribute_value(a) == inp_shape[1]:
                        is_dw = True
                        break

                if is_dw:
                    op_name = f"depthwise_{op_name}"

            layer_path = f"{LAYER_DIR}/{model_name}_{op_name}.onnx"
            onnx.utils.extract_model(model_path, layer_path, inputs, outputs)
            if store_as_polymath:
                convert_model_to_polymath(layer_path)
            layers[n.op_type] = 1
        else:
            layers[n.op_type] += 1

def print_unique_model_layers(model_name, store_as_polymath=False):
    layers = {}
    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    new_model_path = f"{MODEL_DIR}/{model_name}_shapes.onnx"
    model = onnx.load_model(model_path)
    # model = onnx.shape_inference.infer_shapes(model)
    with open(new_model_path, "wb") as f:
        f.write(model.SerializeToString())
    for n in model.graph.node:
        if n.op_type not in layers:
            layers[n.op_type] = 1
        else:
            layers[n.op_type] += 1

    for k, v in layers.items():
        print(f"{k}, {v}")



def get_onnx_shape(tensor_dict, val_name):
    assert val_name in tensor_dict
    value = tensor_dict[val_name]
    shape = [d.dim_value for d in value.type.tensor_type.shape.dim]
    return tuple(shape)

def store_target_model_layer(model_name, layer_name, store_name=None, store_as_polymath=False,
                             store_min=False, tgt_layer_name=None):
    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    model = onnx.load_model(model_path)
    found = False
    layers = []
    op_name = layer_name.lower() if store_name is None else store_name
    layer_path = f"{LAYER_DIR}/{model_name}_{op_name}.onnx"
    for n in model.graph.node:

        if n.op_type == layer_name:
            if tgt_layer_name and tgt_layer_name != n.name:
                continue
            outputs = n.output
            if n.op_type == 'BatchNormalization':
                outputs = [n.output[0]]
            layers.append({'inputs': n.input, 'outputs': outputs})

            found = True
            if not store_min:
                break
    if not found:
        raise RuntimeError(f"Unable to find layer {layer_name} in model")

    if not store_min:
        layer = layers[0]
    else:
        tensor_dict = {i.name: i for i in model.graph.input}
        tensor_dict.update({o.name: o for o in model.graph.output})
        tensor_dict.update({v.name: v for v in model.graph.value_info})
        min_size = float('inf')
        min_layer = None
        for l in layers:
            lsize = np.sum([np.prod(get_onnx_shape(tensor_dict, i)) for i in l['inputs']])
            lsize += np.sum([np.prod(get_onnx_shape(tensor_dict, i)) for i in l['outputs']])
            if lsize < min_size:
                min_size = lsize
                min_layer = l
        assert min_layer is not None
        layer = min_layer
    onnx.utils.extract_model(model_path, layer_path, layer['inputs'], layer['outputs'])
    if store_as_polymath:
        convert_model_to_polymath(layer_path)



if __name__ == "__main__":
    # argparser = argparse.ArgumentParser(description='ONNX Benchmark Generator')
    # argparser.add_argument('-b', '--benchmark', required=True,
    #                        help='Name of the benchmark to create. One of "resnet18", "lenet')
    #
    #
    # argparser.add_argument('-t', '--training_mode', type=str2bool, nargs='?', default=False,
    #                        const=True, help='Whether or not the model is in training mode')
    #
    #
    # argparser.add_argument('-pm', '--to_polymath', type=str2bool, nargs='?', default=False,
    #                        const=True, help='Whether or not the model should be converted to PolyMath')
    # args = argparser.parse_args()
    # model_name = 'lenetbn'
    model_name = 'vit-pad256-transpose-ort'
    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    #
    # convert_model_to_polymath(model_path)
    # store_unique_model_layers(model_name, store_as_polymath=True)
    print_unique_model_layers(model_name, store_as_polymath=False)
    # store_target_model_layer(model_name, "Conv", store_name="conv_large", store_as_polymath=True, store_min=False, tgt_layer_name='Conv_114')
    # model_names = ['reference_fc1', 'resnet_50_v2_fc1', 'resnet_50_v2_c1', 'resnet_50_v2_c2', 'vgg_16_fc1', 'vgg_16_c2',
    #                'inceptionv3_fc1', 'inceptionv3_c1', 'squeezenet_c1', 'squeezenet_c2', 'mobilenet_v3_large_c1',
    #                'mobilenet_v3_large_c2', 'googlenet_fc1', 'bert_large_ffn_fc1', 'bert_large_ffn_fc2',
    #                'bert_large_self_attn_kqv_gen', 'bert_large_self_attn_qk', 'bert_large_self_attn_vqk',
    #                'bert_large_self_attn_zw', 'dlrm_mlp_top_1', 'dlrm_mlp_top_2', 'dlrm_mlp_top_3', 'dlrm_mlp_top_4']
    # for model_name in model_names:
    #     model_path = f"{MODEL_DIR}/{model_name}.onnx"
    #     store_unique_model_layers(model_name, store_as_polymath=True)
