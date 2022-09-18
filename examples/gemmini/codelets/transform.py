from codelets.adl.graph import ArchitectureNode
from codelets.templates.codelet_template import CodeletTemplate
from examples.genesys import OP_DTYPES


def tensor_reshape4d2d(hag: ArchitectureNode):

    # TODO: Right now, shapes are fixed. Need to enable different dimension combinations
    with CodeletTemplate("tensor_reshape4d2d") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
    return cdlt


def tensor_reshape4d3d(hag: ArchitectureNode):

    # TODO: Right now, shapes are fixed. Need to enable different dimension combinations
    with CodeletTemplate("tensor_reshape4d3d") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
    return cdlt


def tensor_reshape3d4d(hag: ArchitectureNode):

    # TODO: Right now, shapes are fixed. Need to enable different dimension combinations
    with CodeletTemplate("tensor_reshape3d4d") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.outputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.outputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.outputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.outputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
    return cdlt

def tensor_squeeze(hag: ArchitectureNode):

    # TODO: Right now, shapes are fixed. Need to enable different dimension combinations
    with CodeletTemplate("tensor_squeeze") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
    return cdlt


def tensor_resize(hag: ArchitectureNode):

    # TODO: Right now, shapes are fixed. Need to enable different dimension combinations
    with CodeletTemplate("resize") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H1 = cdlt.dummy_op("H1", cdlt.node.inputs[0].shape[2])
        W1 = cdlt.dummy_op("W1", cdlt.node.inputs[0].shape[3])

        H2 = cdlt.dummy_op("H2", cdlt.node.outputs[0].shape[2])
        W2 = cdlt.dummy_op("W2", cdlt.node.outputs[0].shape[3])
        DIMS = cdlt.dummy_op('DIMS', cdlt.node.inputs[1].shape[0])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, C, H1, W1], default_dtype=OP_DTYPES[2])
        scale = cdlt.create_operand_template("scale", OP_DTYPES, [DIMS], default_dtype=OP_DTYPES[2])
        # op2 = cdlt.create_operand_template("op2", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H2, W2], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1, scale])
        cdlt.set_outputs([out])
    return cdlt

def tensor_pad(hag: ArchitectureNode):

    with CodeletTemplate("tensor_pad") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        cdlt.configure("end", "SIMD")
    return cdlt


def tensor_flip(hag: ArchitectureNode):

    with CodeletTemplate("tensor_flip") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        C = cdlt.dummy_op("C", cdlt.node.inputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        data = cdlt.create_operand_template("data", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, C, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([data])
        cdlt.set_outputs([out])
        cdlt.configure("end", "SIMD")

    return cdlt


def concat(hag: ArchitectureNode):

    with CodeletTemplate("concat") as cdlt:

        N = cdlt.dummy_op("N", cdlt.node.inputs[0].shape[0])
        IC1 = cdlt.dummy_op("IC1", cdlt.node.inputs[0].shape[1])
        IC2 = cdlt.dummy_op("IC2", cdlt.node.inputs[1].shape[1])
        OC = cdlt.dummy_op("OC", cdlt.node.outputs[0].shape[1])
        H = cdlt.dummy_op("H", cdlt.node.inputs[0].shape[2])
        W = cdlt.dummy_op("W", cdlt.node.inputs[0].shape[3])

        op1 = cdlt.create_operand_template("op1", OP_DTYPES, [N, IC1, H, W], default_dtype=OP_DTYPES[2])
        op2 = cdlt.create_operand_template("op2", OP_DTYPES, [N, IC2, H, W], default_dtype=OP_DTYPES[2])
        out = cdlt.create_operand_template("out", OP_DTYPES, [N, OC, H, W], default_dtype=OP_DTYPES[2])
        cdlt.set_inputs([op1, op2])
        cdlt.set_outputs([out])

    return cdlt

def load_transform_cdlts(cfg):

    TRANSFORM_CDLTS = {
        'tensor_reshape4d2d': tensor_reshape4d2d,
        'tensor_reshape4d3d': tensor_reshape4d3d,
        'tensor_reshape3d4d': tensor_reshape3d4d,
        # 'tensor_flip': tensor_flip,
        # 'tensor_pad': tensor_pad,
        # 'concat': concat,
        'tensor_squeeze': tensor_squeeze,
        # 'resize': tensor_resize
    }
    return TRANSFORM_CDLTS
