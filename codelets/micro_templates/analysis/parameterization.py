from codelets.codelet_template import CodeletTemplate
from codelets.micro_templates import OperandTemplate
from codelets.adl.graph import ArchitectureNode
from codelets.adl.graph.graph_algorithms import compute_node_levels, get_shortest_paths


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
    for i, o in enumerate(cdlt.ops):
        if o.op_type == "compute":
            # If target for compute op is not set, first identify where op can be executed in HAG
            if not o.is_target_set():
                set_targets(cdlt, hag)
            storage_subgraph_nodes = {}
            # Filter and get only storage nodes from subgraph nodes
            for name, node in hag.all_subgraph_nodes.items():
                if node.get_type() == 'StorageNode' or node.name == o.target:
                    storage_subgraph_nodes[name] = node
            # Get all possible paths via storage nodes from DRAM to the target node
            paths = get_shortest_paths(storage_subgraph_nodes, 'DRAM', o.target)
            number_of_paths = len(paths)
            path_index = 0
            transfer_operands = []
            new_sources = []
            # For each source of the compute op, add transfers
            # Round robin between paths if multiple paths available
            for source in o.sources:
                path = paths[path_index]
                result = source
                for storage_node in path[1:]:
                    result, operand = cdlt.transfer(result, storage_node, add_op=False)
                    transfer_operands.append(operand)
                path_index = (path_index + 1) % number_of_paths
                new_sources.append(result)
            # Store the index in the list of codelet ops where the new transfers should be inserted
            if len(transfer_operands) > 0:
                operand_map[i] = transfer_operands
            # Update the sources with the results of final transfers
            o.sources = new_sources
    new_ops = []
    # Insert the new transfer ops at the appropriate index in the codelet ops
    for i, o in enumerate(cdlt.ops):
        if i in operand_map:
            new_ops.extend(operand_map[i])
        new_ops.append(o)
    cdlt.ops = new_ops
