from codelets.adl import Codelet, Capability, Operand, ArchitectureNode, ComputeNode
from codelets.adl.util import tile_perms, make_operand_from_node
import numpy as np
from collections import namedtuple
SIMD_INPUT_COMPONENTS = ["OBUF", "VMEM", "IMM", "EXTMEM"]
SIMD_OUTPUT_COMPONENTS = ["IBUF", "VMEM", "EXTMEM", "IMM"]

SYSTOLIC_ARRAY_INPUT_COMPONENTS = ["IBUF", "WBUF", "BBUF"]
SYSTOLIC_ARRAY_OUTPUT_COMPONENTS = ["OBUF"]

def conv2d_systolic_array(node, hag):
    cap = Codelet("conv2d")
    cap.input_dimension_names = [["n", "ic", "h", "w"], ["oc", "ic", "kh", "kw"]]
    cap.output_dimension_names = ["n", "oc", "oh", "ow"]
    cap.input_dtypes = ["fxp32", "fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SYSTOLIC_ARRAY_INPUT_COMPONENTS
    cap.output_components = SYSTOLIC_ARRAY_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    op_params = {"stride": node.args[-2], "pad": node.args[-1]}
    loop_order = ["oc", "kh", "kw", "ic", "n", "h", "w"]
    op = Capability(cap, "systolic_array", inputs, outputs, loop_order, 1, **op_params)
    tilings = tile_perms(list(op.input_dims), hag)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    op.tiling = tiling_dict
    return op


def conv2d_bias_systolic_array(node, hag):
    cap = Codelet("conv2d_bias")
    cap.input_dimension_names = [["n", "ic", "h", "w"], ["oc", "ic", "kh", "kw"], ["oc"]]
    cap.output_dimension_names = ["n", "oc", "oh", "ow"]
    cap.input_dtypes = ["fxp32", "fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SYSTOLIC_ARRAY_INPUT_COMPONENTS
    cap.output_components = SYSTOLIC_ARRAY_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    op_params = {"stride": node.args[-2], "pad": node.args[-1]}
    loop_order = ["oc", "kh", "kw", "ic", "n", "h", "w"]
    op = Capability(cap, "systolic_array", inputs, outputs, loop_order, 1, **op_params)

    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0

    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    op.tiling = tiling_dict
    return op


def gemm_systolic_array(node, hag):
    # TODO: Need to handle transpose
    cap = Codelet("gemm")
    if "transB" in node.kwargs and node.kwargs["transB"]:
        cap.input_dimension_names = [["M", "N"], ["P", "N"], ["P"]]
        cap.output_dimension_names = ["M", "P"]
    else:
        cap.input_dimension_names = [["M", "N"], ["N", "P"], ["N"]]
        cap.output_dimension_names = ["M", "P"]
    cap.input_dtypes = ["fxp32", "fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SYSTOLIC_ARRAY_INPUT_COMPONENTS
    cap.output_components = SYSTOLIC_ARRAY_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = ["M", "N", "P"]
    op_params = {}
    op = Capability(cap, "systolic_array", inputs, outputs, loop_order, 1, **op_params)
    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    op.tiling = tiling_dict
    return op


def tanh_simd(node, hag: ComputeNode):
    cap = Codelet("tanh")

    cap.input_dtypes = ["fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    # TODO: Fix exec num
    executions = 1
    loop_order = [f"d{i}" for i in range(len(outputs[0].dimensions))]
    cap.input_dimension_names = [loop_order]
    cap.output_dimension_names = loop_order
    op_params = {}
    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)
    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]
    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    op.tiling = tiling_dict
    target_component = hag.get_subgraph_node("SIMD")
    return op

def average_pool_simd(node, hag):
    cap = Codelet("avg_pool")
    cap.input_dimension_names = [["n", "c", "ih", "iw"]]
    cap.output_dimension_names = ["n", "c", "oh", "ow"]
    cap.input_dtypes = ["fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = ["n", "c", "ih", "iw"]
    op_params = {"stride": node.args[4], "kh": node.args[2], "kw": node.args[3],
                 "pad": node.args[5]}
    executions = 1
    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)
    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    op.tiling = tiling_dict

    return op

def softmax_simd(node, hag):
    cap = Codelet("softmax")
    cap.input_dimension_names = [["n", "c"]]
    cap.output_dimension_names = ["n", "c"]
    cap.input_dtypes = ["fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = ["n", "c"]
    op_params = {}
    executions = 1
    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)

    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    op.tiling = tiling_dict
    return op

def add_simd(node, hag):
    cap = Codelet("add")

    cap.input_dtypes = ["fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = [f"d{i}" for i in range(len(outputs[0].dimensions))]
    cap.input_dimension_names = [loop_order, loop_order]
    cap.output_dimension_names = loop_order

    # TODO: Fix executions
    executions = 1
    op_params = {}
    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)

    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1

    return op



def batch_normalization_simd(node, hag):
    cap = Codelet("batchnorm")
    cap.input_dimension_names = [["n", "ic", "h", "w"], ["ic"], ["ic"], ["ic"], ["ic"]]
    cap.output_dimension_names = ["n", "ic", "h", "w"]
    cap.input_dtypes = ["fxp32", "fxp32", "fxp32", "fxp32", "fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = cap.output_dimension_names
    # TODO: Fix executions
    executions = 1
    op_params = {}

    if "eps" in node.kwargs:
        op_params["eps"] = node.kwargs["eps"]
    else:
        op_params["eps"] = 1e-05

    if "momentum" in node.kwargs:
        op_params["momentum"] = node.kwargs["momentum"]
    else:
        op_params["momentum"] = 0.9

    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)

    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1

    return op

def relu_simd(node, hag):
    cap = Codelet("relu")

    cap.input_dtypes = ["fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    # TODO: Fix exec num
    executions = 1
    loop_order = [f"d{i}" for i in range(len(outputs[0].dimensions))]
    cap.input_dimension_names = [loop_order]
    cap.output_dimension_names = loop_order
    op_params = {}
    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)
    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]
    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    op.tiling = tiling_dict
    target_component = hag.get_subgraph_node("SIMD")
    return op

def fully_connected_systolic_array(node, hag):
    cap = Codelet("fc")
    cap.input_dimension_names = [["M", "N"], ["N", "P"]]
    cap.output_dimension_names = ["M", "P"]
    cap.input_dtypes = ["fxp32", "fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SYSTOLIC_ARRAY_INPUT_COMPONENTS
    cap.output_components = SYSTOLIC_ARRAY_OUTPUT_COMPONENTS
    return cap

def get_binary_simd(node, hag):
    cap = Codelet(f"{node.op_name}")
    cap.input_dtypes = ["fxp32", "fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS
    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = [f"d{i}" for i in range(len(outputs[0].dimensions))]
    cap.input_dimension_names = [loop_order, loop_order]
    cap.output_dimension_names = loop_order
    # TODO: Fix executions
    executions = 1
    op_params = {}
    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)

    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1

    return op

def maxpool_simd(node, hag):
    cap = Codelet("maxpool")
    cap.input_dimension_names = [["n", "ic", "h", "w"]]
    cap.output_dimension_names = ["n", "ic", "oh", "ow"]
    cap.input_dtypes = ["fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = cap.input_dimension_names[0]
    # TODO: Fix executions
    executions = 1
    op_params = {"stride": node.args[4], "kernel_size": (node.args[2], node.args[3]), "pad": node.args[4]}

    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)

    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    return op

def flatten(node, hag):
    cap = Codelet("flatten")
    cap.input_dtypes = ["fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS
    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = [f"d{i}" for i in range(len(inputs[0].dimensions))]

    cap.input_dimension_names = [loop_order]
    cap.output_dimension_names = ["M", "N"]
    op_params = {"axis": 1}

    if "axis" in node.kwargs:
        op_params["axis"] = node.kwargs["axis"]

    executions = 1
    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)

    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    return op


def global_avg_pool_simd(node, hag):
    cap = Codelet("global_avg_pool")
    cap.input_dimension_names = [["n", "ic", "h", "w"]]
    cap.output_dimension_names = ["n", "ic", "oh", "ow"]
    cap.input_dtypes = ["fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS
    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = cap.input_dimension_names[0]
    # TODO: Fix executions
    executions = 1
    op_params = {}
    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)

    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    return op

def macc_simd(node, hag):
    pass

def max_simd(node, hag):
    pass

def min_simd(node, hag):
    pass

def rshift_simd(node, hag):
    pass

def move_simd(node, hag):
    pass

def cond_mv_true_simd(node, hag):
    pass

def cond_mv_false_simd(node, hag):
    pass

def not_simd(node, hag):
    pass

def and_simd(node, hag):
    pass

def or_simd(node, hag):
    pass

def leaky_relu_simd(node, hag):
    pass

def sigmoid_simd(node, hag):
    pass

def exp_simd(node, hag):
    pass

def ln_simd(node, hag):
    pass

def sqrt_simd(node, hag):
    pass

def inv_sqrt_simd(node, hag):
    pass

def log2_simd(node, hag):
    pass

def eq_simd(node, hag):
    pass

def neq_simd(node, hag):
    pass

def gt_simd(node, hag):
    pass

def gte_simd(node, hag):
    pass

def lt_simd(node, hag):
    pass

def lte_simd(node, hag):
    pass


GENESYS_CAPS = {
    "conv": conv2d_systolic_array,
    "conv_bias": conv2d_bias_systolic_array,
    "tanh": tanh_simd,
    "elem_tanh": tanh_simd,
    "gemm": gemm_systolic_array,
    "avg_pool": average_pool_simd,
    "softmax": softmax_simd,
    "elem_add": get_binary_simd,
    "elem_mul": get_binary_simd,
    "elem_sub": get_binary_simd,
    "elem_div": get_binary_simd,
    "batch_norm": batch_normalization_simd,
    "max_pool": maxpool_simd,
    "global_avg_pool": global_avg_pool_simd,
    "relu": relu_simd,
    "coarse_flatten": flatten
}