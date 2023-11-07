import argparse
import onnx
from collections import defaultdict
from onnx import shape_inference

def get_tensor_shape(tensor_name, graph):
    # Helper function to get the shape of a tensor by its name
    all_tensors_info = list(graph.value_info) + list(graph.input) + list(graph.output) + list(graph.initializer)
    for tensor_info in all_tensors_info:
        if tensor_info.name == tensor_name:
            # Initializer tensors have a slightly different structure
            if hasattr(tensor_info, 'dims'):
                return tuple(tensor_info.dims)
            else:
                return tuple(dim.dim_value if (dim.dim_value > 0 and dim.dim_value is not None) else '?'
                             for dim in tensor_info.type.tensor_type.shape.dim)
    return tuple()  # Return an empty tuple if shape not found


def format_operation_name(node, graph):
    # Generate formatted operation name with dimensions
    input_shapes = [get_tensor_shape(inp, graph) for inp in node.input if get_tensor_shape(inp, graph)]
    op_name_with_dims = node.op_type + ('' if not input_shapes else "".join(map(lambda d: str(len(d)) + "d", input_shapes)))
    return op_name_with_dims, input_shapes

def main(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Apply shape inference on the model
    inferred_model = shape_inference.infer_shapes(model)

    # Get the graph from the inferred model
    graph = inferred_model.graph

    # Dictionary to hold operation details
    operations_details = defaultdict(lambda: defaultdict(list))

    # Collect operations with dimensions and shapes in the graph
    for node in graph.node:
        op_name_with_dims, input_shapes = format_operation_name(node, graph)
        operations_details[op_name_with_dims][tuple(input_shapes)].append(node)

    # Print the report
    maximum_operation_name_length = max(len(op) for op in operations_details.keys())
    print(f"operation_name{' ' * (maximum_operation_name_length - 14)} | number_of_instances")
    width_of_first_column = max(maximum_operation_name_length, 14)
    for op, shapes_dict in operations_details.items():
        instance_count = sum(len(nodes) for nodes in shapes_dict.values())
        print(f"{op}{' ' * (width_of_first_column - len(str(op)))} | {instance_count}")
        print(f"{' ' * (width_of_first_column + 2)}\\")
        for shapes, nodes in shapes_dict.items():
            if shapes:  # Only print if there are shapes available
                shapes_str = ', '.join('x'.join(str(dim) for dim in shape) for shape in shapes)
                print(f"{' ' * (width_of_first_column + 3)}| {shapes_str}")
        print(f"{' ' * (width_of_first_column + 2)}/")

if __name__ == "__main__":
    # Parse the command line arguments for the ONNX model path
    parser = argparse.ArgumentParser(description='Process an ONNX model.')
    parser.add_argument('model_path', type=str, help='Path to the ONNX model.')
    args = parser.parse_args()
    
    # Run main function
    main(args.model_path)
