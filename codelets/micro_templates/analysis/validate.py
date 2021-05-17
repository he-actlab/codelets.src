from codelets.codelet_template import CodeletTemplate
from codelets.adl.graph import ArchitectureNode

def validate_operations(cdlt: CodeletTemplate, hag: ArchitectureNode):

    for op in cdlt.ops:
        if op.op_type == "compute":
            source_locations = []
            for s in op.sources:
                assert s.location == op.target
                source_locations.append(cdlt.get_source_location(s))
            dest = cdlt.get_operation_output(op)
            op_str = f"{op.op_name}: {', '.join(source_locations)} --> {dest}"
            print(f"{op_str}")
        if op.op_type == "transfer":
            source_location = op.src_op.location
            dest_location = op.dst_op.location
            if not hag.has_edge(source_location, dest_location):
                raise RuntimeError(f"Invalid transfer between operands for {op.op_str}:\n"
                      f"Source: {source_location}\n"
                      f"Dest: {dest_location}")

def find_compute_paths(cdlt: CodeletTemplate, hag: ArchitectureNode):
    input_compute_paths = {}
    # First do outputs
    for i in cdlt.inputs:
        input_compute_paths[i.name] = []
        target_operand = i

        for o in cdlt.ops:
            if o.op_type == "transfer" and o.src_op == target_operand:
                target_operand = o.dst_op
                input_compute_paths[i.name].append((o.op_str, o.dst_op.name, o.dst_op.location))
            elif o.op_type == "compute" and target_operand in o.sources:
                input_compute_paths[i.name].append((o.op_str, target_operand.name))

    output_compute_paths = {}
    for o in cdlt.outputs:
        output_compute_paths[o.name] = []
        target_operand = o
        for op in reversed(cdlt.ops):
            if op.op_type == "transfer" and op.dst_op == target_operand:
                target_operand = op.src_op
                output_compute_paths[o.name].append((op.op_str, op.src_op.name, op.src_op.location))
            elif op.op_type == "compute" and target_operand == cdlt.get_output_operand(op.op_str):
                output_compute_paths[o.name].append((op.op_str, target_operand.name))

    return input_compute_paths, output_compute_paths