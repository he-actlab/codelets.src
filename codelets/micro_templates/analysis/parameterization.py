from codelets.codelet_template import CodeletTemplate
from codelets.micro_templates import OperandTemplate, IndexOperandTemplate
from codelets.adl.graph import ArchitectureNode
from codelets.adl.graph.graph_algorithms import get_shortest_paths


def set_targets(cdlt: CodeletTemplate, hag: ArchitectureNode):
    for o in cdlt.ops:
        if o.op_type == "compute":
            # Search which target supports the current op
            compute_ops = hag.operation_mappings['compute']
            for target in compute_ops.keys():
                if o.op_name in compute_ops[target]:
                    o.target = target
                    break
            if not o.is_target_set():
                raise RuntimeError(f"Unable to find target supporting op {o.op_name} in model")


def add_transfers(cdlt: CodeletTemplate, hag: ArchitectureNode):
    operand_map = {}
    transferred_sources = {}
    for i, o in enumerate(cdlt.ops):
        if o.op_type == "compute":
            # If target for compute op is not set, first identify where op can be executed in HAG
            if not o.is_target_set():
                set_targets(cdlt, hag)
            storage_subgraph_nodes = hag.get_subgraph_nodes_of_type('StorageNode')
            storage_subgraph_nodes[o.target] = hag.get_subgraph_node(o.target)
            # Get all possible paths via storage nodes from DRAM to the target node
            paths = get_shortest_paths(storage_subgraph_nodes, 'DRAM', o.target)
            number_of_paths = len(paths)
            path_index = 0
            transfer_operands = []
            new_sources = []
            # For each source of the compute op, add transfers
            # Round robin between paths if multiple paths available
            for source in o.sources:
                source_in_inputs = False
                # The source of compute op can be an OperandTemplate or IndexOperandTemplate
                if isinstance(source, OperandTemplate):
                    source_in_inputs = source in cdlt.inputs
                elif isinstance(source, IndexOperandTemplate):
                    source_in_inputs = source.operand in cdlt.inputs
                if source.name not in transferred_sources:
                    if source_in_inputs:
                        path = paths[path_index]
                        source.location = 'DRAM'
                        result = source
                        for storage_node in path[1:]:
                            result, operand = cdlt.transfer(result, storage_node, add_op=False)
                            result.location = storage_node
                            # Store the loop information, same as compute op, in the new operand from the transfer
                            operand.loop_id = o.loop_id
                            operand.loop_level = o.loop_level
                            transfer_operands.append(operand)
                    else:
                        # Allocate storage for this source which is an intermediate value
                        # In the paths, last element is the compute node and elements before that are storage nodes
                        source.location = paths[path_index][-2]
                        result, operand = cdlt.transfer(source, o.target, add_op=False)
                        result.location = o.target
                        # Store the loop information, same as compute op, in the new operand from the transfer
                        operand.loop_id = o.loop_id
                        operand.loop_level = o.loop_level
                        transfer_operands.append(operand)
                    path_index = (path_index + 1) % number_of_paths
                    new_sources.append(result)
                    # Mark the source as transferred
                    transferred_sources[source.name] = 1
            # Set the location of output operand of compute operation as target
            o.output_operand.location = o.target
            # Store the index in the list of codelet ops where the new transfers should be inserted
            if len(transfer_operands) > 0:
                operand_map[i] = transfer_operands
            # Update the sources with the results of final transfers
            o.sources = new_sources
        if o.op_type == 'transfer' and o.src_op.location is not None:
            # Filter and get only storage nodes from subgraph nodes
            storage_subgraph_nodes = hag.get_subgraph_nodes_of_type('StorageNode')
            # Add the node where the source operand of transfer is executed.
            # This will be the root for the path search
            storage_subgraph_nodes[o.src_op.location] = hag.get_subgraph_node(o.src_op.location)
            # Get all possible paths via storage nodes from node to DRAM
            paths = get_shortest_paths(storage_subgraph_nodes, o.src_op.location, 'DRAM')
            number_of_paths = len(paths)
            path_index = 0
            transfer_operands = []
            # For each source of the compute op, add transfers
            # Round robin between paths if multiple paths available
            source = o.src_op
            if source.name not in transferred_sources and o.dst_op in cdlt.outputs:
                path = paths[path_index]
                result = source
                o.dst_op.location = 'DRAM'
                for storage_node in path[1:-1]:
                    result, operand = cdlt.transfer(result, storage_node, add_op=False)
                    operand.loop_id = o.loop_id
                    operand.loop_level = o.loop_level
                    transfer_operands.append(operand)
                # Mark the source as transferred
                transferred_sources[source.name] = 1
                o.src_op = result
            else:
                continue
            # Store the index in the list of codelet ops where the new transfers should be inserted
            if len(transfer_operands) > 0:
                operand_map[i] = transfer_operands

    new_ops = []
    # Insert the new transfer ops at the appropriate index in the codelet ops
    for i, o in enumerate(cdlt.ops):
        if i in operand_map:
            new_ops.extend(operand_map[i])
        new_ops.append(o)
    cdlt.ops = new_ops
