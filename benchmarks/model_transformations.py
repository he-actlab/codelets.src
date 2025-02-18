from pathlib import Path

import polymath as pm
import onnx
from onnxsim import simplify
from onnx.tools import update_model_dims
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
import onnxruntime as ort
from collections import defaultdict
import pprint
from onnxruntime.transformers.onnx_model import OnnxModel
from benchmarks.transpose_layer_norm import LayerNormTranspose, SoftmaxTranspose
from onnx import AttributeProto, mapping

MODEL_DIR = Path(f"{Path(__file__).parent}/models")
CWD = Path(f"{__file__}").parent
DICT_SIZE = 10000
import numpy as np

from onnx import helper, TensorProto, numpy_helper, shape_inference
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

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

def optimize_graph(model_name, single_params=None):
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    load_path = f"{MODEL_DIR}/{model_name}.onnx"
    store_path = f"{MODEL_DIR}/{model_name}-opt.onnx"
    optimize_onnx(load_path, store_path, None, None, False, single_params=single_params)
    return f"{model_name}-opt"


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
    assert check
    model = update_node_names(model)
    model = update_edge_names(model)
    with open(store_path, "wb") as f:
        f.write(model.SerializeToString())

    if to_polymath:
        graph = pm.from_onnx(store_path)
        pm.pb_store(graph, f"{CWD}/full_dnns/")

def quantize_model(model_name):
    load_path = f"{MODEL_DIR}/{model_name}.onnx"
    store_path = f"{MODEL_DIR}/{model_name}_quantized.onnx"
    model = onnx.load(load_path)
    for i in range(len(model.graph.node)):
        node = model.graph.node[i]
        if node.op_type == "Clip":
            inpt = node.input[0]
            outpt = node.output[0]
            relu_node = onnx.helper.make_node("Relu", [inpt], [outpt], node.name)
            model.graph.node[i] = relu_node

def set_bert_shapes(model_name, batch_size=1, max_seq_length = 128):
    bert_name = model_name
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    load_path = f"{MODEL_DIR}/{bert_name}.onnx"
    store_path = f"{MODEL_DIR}/{bert_name}-opt.onnx"
    inpt_shapes = {"input_ids": (batch_size, max_seq_length),
              "attention_mask": (batch_size, max_seq_length),
              "token_type_ids": (batch_size, max_seq_length),
              }
    out_shapes = {"output_0": (batch_size, max_seq_length, 28996),
                  # "end": (batch_size, max_seq_length)
                  }
    optimize_onnx(load_path, store_path, inpt_shapes, out_shapes, False)


def validate_transformation(init_model_name, opt_model_name, batch_size=1, seq_length=128):
    init_model_path = f"{MODEL_DIR}/{init_model_name}.onnx"
    opt_model_path = f"{MODEL_DIR}/{opt_model_name}.onnx"

    # Turn off optimizations
    init_options = ort.SessionOptions()
    init_options.optimized_model_filepath = f"{MODEL_DIR}/{init_model_name}-valid.onnx"
    init_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    init_options.use_deterministic_compute = True

    opt_options = ort.SessionOptions()
    opt_options.optimized_model_filepath = f"{MODEL_DIR}/{opt_model_name}-valid.onnx"
    opt_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opt_options.use_deterministic_compute = True

    inputs = {}
    inputs['input_ids'] = np.random.randint(DICT_SIZE, size=(batch_size, seq_length), dtype=np.int64)
    inputs['attention_mask'] = np.ones((batch_size, seq_length), dtype=np.int64)
    inputs['token_type_ids'] = np.zeros((batch_size, seq_length), dtype=np.int64)
    sess = ort.InferenceSession(init_model_path, providers=ort.get_available_providers(),
                                sess_options=init_options)
    result = sess.run(None, inputs)[0]

    new_sess = ort.InferenceSession(opt_model_path, providers=ort.get_available_providers(),
                                    sess_options=opt_options)
    new_result = new_sess.run(None, inputs)[0]
    np.testing.assert_allclose(new_result, result)

    init_model = onnx.load(init_model_path)
    opt_model = onnx.load(opt_model_path)

    init_op_counts = defaultdict(int)
    opt_op_counts = defaultdict(int)


    for n in init_model.graph.node:
        init_op_counts[n.op_type] += 1
        init_op_counts['total'] += 1

    for n in opt_model.graph.node:
        opt_op_counts[n.op_type] += 1
        opt_op_counts['total'] += 1

    print(f"{init_model_name} op counts: {init_op_counts}")
    print(f"{opt_model_name} op counts: {opt_op_counts}")


def get_reduction_axis(node):
    for attr in node.attribute:
        if attr.name == "axes":
            assert attr.type == AttributeProto.AttributeType.INTS, f"Invalid type for attribute: {helper.printable_type(attr.type)}"
            return attr.ints.pop()

def transpose_reduction_ops(model_name, validate_transpose=False):
    load_path = f"{MODEL_DIR}/{model_name}.onnx"
    store_path = f"{MODEL_DIR}/{model_name}-transpose.onnx"
    model = onnx.load(load_path)
    if "bert" in model_name:
        model_type = "bert"
    elif "vit" in model_name:
        model_type = "vit"
    else:
        print(f"Unsupported model for transpose: {model_name}")
    ln_model_fuser = LayerNormTranspose(model, model_type)
    ln_model_fuser.apply()
    ln_model_fuser.model.save_model_to_file(store_path)

    softmax_model_fuser = SoftmaxTranspose(ln_model_fuser.model, model_type)
    softmax_model_fuser.apply()
    softmax_model_fuser.model.save_model_to_file(store_path)
    onnx.checker.check_model(softmax_model_fuser.model.model)

    if validate_transpose:
        model = softmax_model_fuser.model
        for n in model.model.graph.node:
            if n.op_type == "ReduceMean":
                reduce_axis = get_reduction_axis(n)
                if reduce_axis == -1 or reduce_axis == 2:
                    raise RuntimeError("Found invalid reducemean on last axis")
            elif n.op_type == "Softmax":
                reduce_axis = get_reduction_axis(n)
                if reduce_axis == -1 or reduce_axis == 3:
                    raise RuntimeError("Found invalid reducemean on last axis")

def apply_pad256(model_name, pad_amt=256):
    load_path = f"{MODEL_DIR}/{model_name}.onnx"
    store_name = f"{model_name}-pad{pad_amt}"
    store_path = f"{MODEL_DIR}/{store_name}.onnx"
    model_proto = onnx.load(load_path)

    model = OnnxModel(model_proto)


    # In order to apply padding, we need to:
    ## Create a padding initializer of zeros with the correct shape and add it to the graph
    ## Set the initializer as the input to the padding operation
    ## Replace all uses of the previous initializer with the output of the padding node

    def pad_tensor(tensor, np_dtype, new_size):
        if not isinstance(tensor, onnx.TensorProto):
            raise ValueError("Expected input type is an ONNX TensorProto but got %s" % type(tensor))

        if len(tensor.dims) != 3:
            raise ValueError("Only 3-D tensors can be transposed")

        if tensor.raw_data:
            float32_data = np.reshape(np.frombuffer(tensor.raw_data, dtype=np_dtype), tensor.dims)
            pad_val = new_size - float32_data.shape[1]
            assert pad_val > 0
            float32_padded_data = np.pad(float32_data, ((0,0), (0, pad_val), (0, 0)), 'constant')
            new_tensor = numpy_helper.from_array(float32_padded_data, tensor.name)
            return new_tensor
        else:
            raise ValueError("only raw buffer supported")

    # There are only three changes necessary:
    ## First, `vit.embeddings.position_embeddings` needs to be updatd from [1, 197, 768] to [1, 256, 768]

    pos_embeddings = model.get_initializer('vit.embeddings.position_embeddings')
    pe_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[pos_embeddings.data_type]
    pe_idx = list(model.model.graph.initializer).index(pos_embeddings)
    new_pos_embeddings = pad_tensor(pos_embeddings, pe_dtype, 256)
    model.model.graph.initializer[pe_idx].CopyFrom(new_pos_embeddings)

    ## Second, `vit.embeddings.cls_token` needs to be updaetd from [1, 1, 768] to [1, 60, 768]

    cls_token = model.get_initializer('vit.embeddings.cls_token')
    ct_idx = list(model.model.graph.initializer).index(cls_token)
    ct_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[cls_token.data_type]
    new_cls_token = pad_tensor(cls_token, ct_dtype, 60)
    model.model.graph.initializer[ct_idx].CopyFrom(new_cls_token)


    # Third, we need to make sure that all "Reshape" operations are changing accordingly
    reshapes = model.get_nodes_by_op_type("Reshape")

    for n in reshapes:
        idx, shape_arg = model.get_constant_input(n)
        if idx is not None and 197 in shape_arg:
            shape_name = n.input[idx]
            shape_init = model.get_initializer(shape_name)
            assert shape_init is not None
            new_np_shape = shape_arg.copy()
            for i in range(shape_arg.shape[0]):
                if shape_arg[i] == 197:
                    new_np_shape[i] = 256
            new_shape = numpy_helper.from_array(new_np_shape, shape_name)
            shape_idx = list(model.model.graph.initializer).index(shape_init)
            model.model.graph.initializer[shape_idx].CopyFrom(new_shape)


    model.save_model_to_file(store_path)
    # Now that these are updated, should be good to go?





    return store_name



def bert_use_ort_optimizer(bert_name, set_shapes=True,
                           apply_transpose=False,
                           apply_ort_opt=True, trim_gather=False):
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    load_name = store_name = bert_name
    if apply_transpose:
        print(f"Loading init model from {store_name}")

        store_name = f"{load_name}-transpose"
        transpose_reduction_ops(load_name)
        print(f"Storing transposed model to {store_name}")


    if set_shapes:
        print(f"Loading init from {store_name}")
        set_bert_shapes(f"{store_name}")
        load_name = store_name
        store_name = f"{store_name}-opt"
        print(f"Storing shaped model to {store_name}")

    if trim_gather:
        load_path = f"{MODEL_DIR}/{store_name}.onnx"
        store_path = f"{MODEL_DIR}/{store_name}-trimmed.onnx"
        store_name = f"{store_name}-trimmed"
        input_names = ["gather_15_225Y", "gather_16_226Y", "cast_2_208Y"]
        output_names = ["output_0"]

        onnx.utils.extract_model(load_path, store_path, input_names, output_names)

    if apply_ort_opt:
        load_path = f"{MODEL_DIR}/{store_name}.onnx"
        store_path = f"{MODEL_DIR}/{store_name}-ort.onnx"
        store_name = f"{store_name}-ort"
        opt_options = FusionOptions('bert')
        opt_options.enable_embed_layer_norm = False
        opt_options.enable_gelu = True
        opt_options.enable_layer_norm = False
        opt_options.enable_attention = False
        opt_options.enable_skip_layer_norm = False
        opt_options.enable_embed_layer_norm = False
        opt_options.enable_bias_skip_layer_norm = False
        opt_options.enable_bias_gelu = False
        opt_options.enable_gelu_approximation = False
        print(f"Loading init from {store_name}")
        opt_model = optimizer.optimize_model(
            load_path,
            'bert',
            num_heads=12,
            hidden_size=768,
            optimization_options=opt_options,
            opt_level=0 if apply_transpose else None
        )
        opt_model.save_model_to_file(store_path)
        print(f"Storing model to {store_path}")


def vit_use_ort_optimizer(vit_name,
                          apply_transpose=False,
                          apply_ort_opt=True, pad256=False):
    MODEL_DIR = Path(f"{Path(__file__).parent}/models")
    name = vit_name

    if pad256:
        print(f"Padding {name}")
        name = apply_pad256(name)
        MODEL_DIR = Path(f"{Path(__file__).parent}/models")
        model_path = f"{MODEL_DIR}/{name}.onnx"

        optimize_onnx(model_path, model_path, None, None, False)

    if apply_transpose:
        print(f"Transposing {name}")

        transpose_reduction_ops(name, True)
        name = f"{name}-transpose"

        MODEL_DIR = Path(f"{Path(__file__).parent}/models")
        model_path = f"{MODEL_DIR}/{name}.onnx"

        # if not pad256:
        # optimize_onnx(model_path, model_path, None, None, False)


    if apply_ort_opt:
        print(f"Applying ORT optimization to {name}")

        load_path = f"{MODEL_DIR}/{name}.onnx"
        name = f"{name}-ort"
        store_path = f"{MODEL_DIR}/{name}.onnx"
        opt_options = FusionOptions('bert')
        opt_options.enable_embed_layer_norm = False
        opt_options.enable_gelu = True
        opt_options.enable_layer_norm = False
        opt_options.enable_attention = False
        opt_options.enable_skip_layer_norm = False
        opt_options.enable_embed_layer_norm = False
        opt_options.enable_bias_skip_layer_norm = False
        opt_options.enable_bias_gelu = False
        opt_options.enable_gelu_approximation = False
        opt_model = optimizer.optimize_model(
            load_path,
            'bert',
            num_heads=12,
            hidden_size=768,
            optimization_options=opt_options,
            opt_level=0 if apply_transpose else None
        )
        opt_model.save_model_to_file(store_path)
        print(f"Storing model to {store_path}")




def layer_norm_gelu(model_name):
    load_path = f"{MODEL_DIR}/{model_name}.onnx"
    store_path = f"{MODEL_DIR}/{model_name}-lnorm-gelu.onnx"
    opt_options = FusionOptions('bert')
    opt_options.enable_embed_layer_norm = False
    opt_options.enable_gelu = True
    opt_options.enable_layer_norm = True
    opt_options.enable_attention = False
    opt_options.enable_skip_layer_norm = False
    opt_options.enable_embed_layer_norm = False
    opt_options.enable_bias_skip_layer_norm = False
    opt_options.enable_bias_gelu = False
    opt_options.enable_gelu_approximation = False

    opt_model = optimizer.optimize_model(
        load_path,
        'bert',
        num_heads=12,
        hidden_size=768,
        optimization_options=opt_options
    )
    opt_model.save_model_to_file(store_path)

    op_counts = defaultdict(int)
    for n in opt_model.model.graph.node:
        op_counts[n.op_type] += 1
        op_counts['total'] += 1

    print("Op Counts:")
    pprint.pprint(op_counts)


if __name__ == "__main__":
    # model_name = "bert-base-cased"
    model_name = "vit"
    vit_use_ort_optimizer(model_name,
                           apply_ort_opt=True,
                           apply_transpose=True, pad256=True)
    # transpose_reduce_mean("bert-base-cased1-gelu")
    # set_bert_shapes()
    # layer_norm_gelu('bert-base-cased-opt')
    # validate_transformation('bert-base-cased-opt-ort', 'bert-base-cased-transpose-opt-ort')
    # bert_use_ort_optimizer(model_name)
    # insert_transposed_reductions(model_name)