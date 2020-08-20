from codelets.adl import ArchitectureGraph, ComputeNode, CommunicationNode, StorageNode

PE_CAPABILITIES = ["add", "sub", "mul"]
def generate_tabla(num_pus, pes_per_pu, ni_size, nw_size, nd_size):
    mem_sizes = {"ni": ni_size, "nw": nw_size, "nd": nd_size}
    tabla_graph = ArchitectureGraph()
    pu_map = []
    start_pos = (0, 0)
    for i in range(num_pus):
        pu_map.append(generate_pu(tabla_graph, i, pes_per_pu, mem_sizes, start_pos))

    return tabla_graph

def generate_pe(adl_graph, pe_id, mem_sizes, pu, position):
    pe_node = ComputeNode(f"PE{pe_id}")
    pe_node.set_attr("outer_graph", f"PU{pu.index}")
    pe_node.set_attr("pos", f"{position[0]}, {position[1]}")
    adl_graph._add_node(pe_node)

    nw_node = StorageNode(f"NW{pe_id}")
    nw_node.set_attr("size", mem_sizes['nw'])
    nw_node.set_attr("outer_graph", f"PU{pu.index}")

    adl_graph._add_node(nw_node)
    adl_graph._add_edge(nw_node, pe_node)
    adl_graph._add_edge(pe_node, nw_node)


    ni_node = StorageNode(f"NI{pe_id}")
    ni_node.set_attr("size", mem_sizes['ni'])
    ni_node.set_attr("outer_graph", f"PU{pu.index}")

    adl_graph._add_node(ni_node)
    adl_graph._add_edge(ni_node, pe_node)
    adl_graph._add_edge(pe_node, ni_node)

    nd_node = StorageNode(f"ND{pe_id}")
    nd_node.set_attr("size", mem_sizes['nd'])
    nd_node.set_attr("outer_graph", f"PU{pu.index}")

    adl_graph._add_node(nd_node)
    adl_graph._add_edge(nd_node, pe_node)

    return pe_node

def generate_pu(adl_graph, pu_id, pes_per_pu, mem_sizes, start_pos):
    pu_node = ComputeNode(f"PU{pu_id}")
    pu_node.set_attr("has_subgraph", True)
    adl_graph._add_node(pu_node)
    pes = []
    for p in range(pes_per_pu):
        pes.append(generate_pe(adl_graph, p, mem_sizes, pu_node, start_pos))

    for pe_idx in range(pes_per_pu - 1):
        adl_graph._add_edge(pes[pe_idx], pes[pe_idx + 1])
    adl_graph._add_edge(pes[pes_per_pu - 1], pes[0])
    return pu_node, pes[0]
