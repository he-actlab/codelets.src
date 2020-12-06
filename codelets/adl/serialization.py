import json
from codelets.adl.architecture_graph import ArchitectureGraph
from typing import Union
from codelets.adl.architecture_node import ArchitectureNode
from codelets.adl.communication_node import CommunicationNode
from codelets.adl.compute_node import ComputeNode
from codelets.adl.storage_node import StorageNode
from codelets.adl.codelet import Codelet
from jsonschema import validate
from pathlib import Path

BaseArchNode = Union[ComputeNode, CommunicationNode, StorageNode]
CWD = Path(f"{__file__}").parent
JSON_SCHEMA_PATH = f"{CWD}/adl_schema.json"

def get_compute_blob(node):
    blob = {"name": node.name, "dimensions": node.dimensions}
    capabilities = []
    for name, c in node.capabilities.items():
        cap_blob = {}
        cap_blob['name'] = c.name
        cap_blob['input_dimension_names'] = c.input_dimension_names
        cap_blob['output_dimension_names'] = c.output_dimension_names
        cap_blob['input_dtypes'] = c.input_dtypes
        cap_blob['input_components'] = c.input_components
        cap_blob['output_dtypes'] = c.output_dtypes
        cap_blob['output_components'] = c.output_components
        cap_blob['op_params'] = c.op_params
        cap_blob['latency'] = c.latency
        cap_blob['subcapabilities'] = []
        # TODO: Add subcapabilities
        # for sc in c.subcapabilities:
        #     cap_blob['subcapabilities'].append()
        capabilities.append(cap_blob)
    blob['capabilities'] = capabilities
    return blob

def get_communication_blob(node):
    blob = {"name": node.name, "comm_type": node.get_comm_type(), "latency": node.get_latency()}
    blob['bandwidth'] = node.get_bw()
    return blob

def get_storage_blob(node):
    blob = {"name": node.name, "read_bw": node.read_bw, "write_bw": node.write_bw}
    blob["access_type"] = node.access_type
    blob["size"] = node.size
    blob["input_ports"] = node.input_ports
    blob["output_ports"] = node.output_ports
    blob["buffering_scheme"] = node.buffering_scheme
    blob["latency"] = node.latency
    return blob

def generate_hw_cfg(graph, save_path):
    json_graph = {}
    json_graph[graph.name] = {"compute": [], "storage": [], "communication": []}
    compute_blobs = []
    storage_blobs = []
    comm_blobs = []
    for name, node in graph._all_subgraph_nodes.items():
        if isinstance(node, ComputeNode):
            compute_blobs.append(get_compute_blob(node))
        elif isinstance(node, StorageNode):
            storage_blobs.append(get_storage_blob(node))
        elif isinstance(node, CommunicationNode):
            comm_blobs.append(get_communication_blob(node))
    json_graph[graph.name]["compute"] = compute_blobs
    json_graph[graph.name]["storage"] = storage_blobs
    json_graph[graph.name]["communication"] = comm_blobs

    with open(f"{save_path}", "w") as file:
        json.dump(json_graph, file, indent=4)

def serialize_graph(graph, save_path, validate_graph=False):
    json_graph = {}
    json_graph['graph'] = _serialize_node(graph)
    if validate_graph:
        validate_schema(json_graph)
    with open(f"{save_path}", "w") as file:
        json.dump(json_graph, file, indent=4)

def get_compute_node_attr(node: ComputeNode):
    attr = {}
    key = 'compute_node'
    cap_names = node.get_capabilities()
    cap_dict = {cname: node.get_capability(cname) for cname in cap_names}
    attr['capabilities'] = get_json_capabilities(cap_dict)
    attr['dimensions'] = node.dimensions
    return key, attr

def get_json_capabilities(capabilities):
    caps = []
    for cname, c in capabilities.items():
        cap = {'op_name': cname}
        cap['latency'] = c.latency
        cap['is_atomic'] = c.is_atomic
        cap['input_dimension_names'] = c.input_dimension_names
        cap['output_dimension_names'] = c.output_dimension_names
        cap['input_components'] = c.input_components
        cap['output_components'] = c.output_components
        cap['input_dtypes'] = c.input_dtypes
        cap['output_dtypes'] = c.output_dtypes
        cap_dict = {k.get_name(): k for k in c.subcapabilities}
        cap['subcapabilities'] = get_json_capabilities(cap_dict)
        caps.append(cap)
    return caps

def get_capability_operands(capability):
    inputs = capability.input_components
    outputs = capability.output_components


    return inputs, outputs

def get_storage_node_attr(node: StorageNode):
    key = 'storage_node'
    attr = {}
    attr['capacity'] = node.get_size()
    attr['read_bw'] = node.get_read_bw()
    attr['write_bw'] = node.get_write_bw()
    attr['access_type'] = node.get_access_type()
    return key, attr

def get_communication_node_attr(node: CommunicationNode):
    attr = {}
    key = 'communication_node'
    attr['comm_type'] = node.get_comm_type()
    attr['bw'] = node.get_bw()
    attr['latency'] = node.get_latency()
    return key, attr

def _serialize_node(node: Union[StorageNode, CommunicationNode, ComputeNode]):
    json_node = {}
    json_node['node_id'] = node.index
    json_node['name'] = node.name
    json_node['node_type'] = node.get_type()
    type_key, type_attr = NODE_TYPE_ATTR_FN[node.get_type()](node)
    json_node[type_key] = type_attr
    json_node['attributes'] = node.get_all_attributes()
    json_node['subgraph'] = {"nodes": [], "edges": []}
    for n in node.get_subgraph_nodes():
        sub_node = _serialize_node(n)
        json_node['subgraph']['nodes'].append(sub_node)

    for e in node.get_subgraph_edges():
        e_attr = [{'key': list(k.keys())[0], 'value': list(k.values())[0]} for k in e.attributes]
        sub_edge = {'src': e.src, 'dst': e.dst, 'attributes': e_attr}
        json_node['subgraph']['edges'].append(sub_edge)

    return json_node

def deserialize_graph(filepath, validate_load=False):
    with open(filepath, "r") as graph_path:
        json_graph = json.load(graph_path)

    if validate_load:
        validate_schema(json_graph)
    graph = _deserialize_node(json_graph['graph'])

    return graph

def _deserialize_capability(json_cap):
    # TODO: Fix this once the proper data structures are in place
    name = json_cap.pop("op_name")
    cap = Codelet(name, **json_cap)
    return cap

def _deserialize_compute_attr(json_attr_):
    key = 'compute_node'
    json_attr = json_attr_[key]
    capabilities = []
    for cap in json_attr['capabilities']:
        capabilities.append(_deserialize_capability(cap))
    return (json_attr['dimensions'], capabilities)

def _deserialize_storage_attr(json_attr_):
    key = 'storage_node'
    json_attr = json_attr_[key]
    return (json_attr['read_bw'], json_attr['write_bw'], json_attr['access_type'], json_attr['capacity'])

def _deserialize_communication_attr(json_attr_):
    key = 'communication_node'
    json_attr = json_attr_[key]
    return (json_attr['comm_type'], json_attr['latency'], json_attr['bw'])

def _deserialize_node(json_node):
    node_name = json_node['name']
    node_id = json_node['node_id']
    node_type = json_node['node_type']
    init_args = DESER_NODE_ATTR_FN[node_type](json_node)
    node = NODE_INITIALIZERS[node_type](node_name, *init_args, index=node_id)
    _set_node_attributes_from_json(node, json_node)

    subgraph_nodes = []

    for n in json_node['subgraph']['nodes']:
        sub_node = _deserialize_node(n)
        subgraph_nodes.append(sub_node)
    for sg_n in subgraph_nodes:
        node.add_subgraph_node(sg_n)

    for e in json_node['subgraph']['edges']:
        edge_attr = _get_edge_attributes_from_json(e)
        src = node.subgraph.get_node_by_index(e['src'])
        dst = node.subgraph.get_node_by_index(e['dst'])
        if dst.name == "IBUF":
            print(dst.get_preds())
        node.add_subgraph_edge(src, dst, attributes=edge_attr)

    return node


def _get_edge_attributes_from_json(json_edge):
    edge_attr = {}
    for attr in json_edge['attributes']:
        key = attr['key']
        value = attr['value']
        edge_attr[key] = value

    return edge_attr

def _set_node_attributes_from_json(node, json_node):
    for attr in json_node['attributes']:
        key = attr['key']
        value = attr['value']
        if key in ["out_degree", "in_degree"]:
            continue

        node.set_attr(key, value)

def validate_schema(json_graph):
    with open(JSON_SCHEMA_PATH, "r") as schema_file:
        schema = json.load(schema_file)
    validate(json_graph, schema=schema)

def compare_graphs(graph1: BaseArchNode, graph2: BaseArchNode):
    if graph1.name != graph2.name:
        print(f"Unequal node ids for {graph1.name}:\n"
              f"Graph1: {graph1.name}\n"
              f"Graph2: {graph2.name}\n")
        return False
    elif graph1.get_graph_node_count() != graph2.get_graph_node_count():
        print(f"Unequal node counts for {graph1.name}:\n"
              f"Graph1: {graph1.get_graph_node_count()}\n"
              f"Graph2: {graph2.get_graph_node_count()}\n")
        return False
    elif graph1.get_graph_edge_count() != graph2.get_graph_edge_count():
        print(f"Unequal edge counts for {graph1.name}:\n"
              f"Graph1: {graph1.get_graph_edge_count()}\n"
              f"Graph2: {graph2.get_graph_edge_count()}\n")
        return False
    elif graph1.index != graph2.index:
        print(f"Unequal node ids for {graph1.name}:\n"
              f"Graph1: {graph1.index}\n"
              f"Graph2: {graph2.index}\n")
        return False
    elif graph1.get_type() != graph2.get_type():
        print(f"Unequal node times for {graph1.name}:\n"
              f"Graph1: {graph1.get_type()}\n"
              f"Graph2: {graph2.get_type()}\n")
        return False
    elif graph1._has_parent != graph2._has_parent:
        raise RuntimeError(f"Unequal parents for {graph1.name}:\n"
              f"Graph1: {graph1._has_parent}\n"
              f"Graph2: {graph2._has_parent}")
        # return False
    elif not CMP_NODE_ATTR_FN[graph1.get_type()](graph1, graph2):
        raise RuntimeError(f"Uequal type specific attributes for {graph1.name} ({graph1.get_type()})")
        # return False
    else:
        graph1_attr = {attr['key']: attr['value'] for attr in graph1.get_all_attributes()}
        graph2_attr = {attr['key']: attr['value'] for attr in graph2.get_all_attributes()}
        if not compare_attributes(graph1_attr, graph2_attr):

            raise RuntimeError(f"Unequal attributes for {graph2.name}:\n"
                  f"Graph1: {json.dumps(graph1_attr, indent=4)}\n\n"
                  f"Graph2: {json.dumps(graph2_attr, indent=4)}")
    g1_edges = {(e.src, e.dst): e.attributes for e in graph1.get_subgraph_edges()}
    g2_edges = {(e.src, e.dst): e.attributes for e in graph2.get_subgraph_edges()}
    if len(g1_edges) != len(g2_edges) or set(g1_edges.keys()) != set(g2_edges.keys()):
        raise RuntimeError(f"Unequal edges for {graph1.name}")
        # return False

    for k,v in g1_edges.items():
        e1_attr = {list(k.keys())[0]: list(k.values())[0] for k in v}
        e2_attr = {list(k.keys())[0]: list(k.values())[0] for k in g2_edges[k]}
        edge_eq = compare_attributes(e1_attr, e2_attr)
        if not edge_eq:
            raise RuntimeError(f"Unequal attributes for {graph2.name} in edge {k}:\n"
                  f"Graph1: {json.dumps(v, indent=4)}\n\n"
                  f"Graph2: {json.dumps(g2_edges[k], indent=4)}")
            # return False

    node_names = [n.name for n in graph1.get_subgraph_nodes()]
    for nid in node_names:
        is_eq = compare_graphs(graph1.get_subgraph_node(nid), graph2.get_subgraph_node(nid))
        if not is_eq:
            return is_eq

    return True

def compare_cap_inputs(inputs1, inputs2):
    for i, inp1 in enumerate(inputs1):
        inp2 = inputs2[i]
        if inp1['src'] != inp2['src']:
            print(inp1['src'])
            print(inp2['src'])
            return False
        elif inp1['dims'] != inp2['dims']:
            return False
    return True

def compare_cap_outputs(ouputs1, outputs2):
    for i, out1 in enumerate(outputs2):
        out2 = outputs2[i]
        if out1['dst'] != out2['dst']:
            return False
        elif out1['dims'] != out2['dims']:
            return False
    return True

def compare_capabilities(caps1, caps2):
    for cap_name, c1 in caps1.items():
        if cap_name not in caps2:
            return False
        c2 = caps2[cap_name]
        if c1.latency != c2.latency:
            return False
        if c1.input_components != c2.input_components:
            return False
        elif c1.output_components != c2.output_components:
            return False
        sub_caps1 = {sc1.get_name(): sc1 for sc1 in c1.subcapabilities}
        sub_caps2 = {sc1.get_name(): sc1 for sc1 in c1.subcapabilities}
        if not compare_capabilities(sub_caps1, sub_caps2):
            return False
    return True

def compare_compute_attr(node1: ComputeNode, node2: ComputeNode):
    cap1_names = node1.get_capabilities()
    caps1 = {cname: node1.get_capability(cname) for cname in cap1_names}
    cap2_names = node2.get_capabilities()
    caps2 = {cname: node2.get_capability(cname) for cname in cap2_names}
    if node1.name == "pe_array":
        print(f"{caps1.keys()}")
        print(f"{caps2.keys()}")
    if not compare_capabilities(caps1, caps2):
        return False
    return True


def compare_storage_attr(node1: StorageNode, node2: StorageNode):
    if node1.get_read_bw() != node2.get_read_bw():
        return False
    elif node1.get_write_bw() != node2.get_write_bw():
        return False
    elif node1.get_access_type() != node2.get_access_type():
        return False
    elif node1.get_size() != node2.get_size():
        return False
    return True

def compare_communication_attr(node1: CommunicationNode, node2: CommunicationNode):
    if node1.get_comm_type() != node2.get_comm_type():
        return False
    elif node1.get_bw() != node2.get_bw():
        return False
    return True

def compare_attributes(attr1, attr2):
    if len(attr1) != len(attr2) or set(attr1.keys()) != set(attr2.keys()):
        return False
    for k, v in attr1.items():
        if k not in attr2.keys():
            return False
        elif attr2[k] != v:
            return False
    return True

NODE_INITIALIZERS = {
    "ComputeNode": ComputeNode,
    "StorageNode": StorageNode,
    "CommunicationNode": CommunicationNode
}

NODE_TYPE_ATTR_FN = {
    "ComputeNode": get_compute_node_attr,
    "StorageNode": get_storage_node_attr,
    "CommunicationNode": get_communication_node_attr
}

DESER_NODE_ATTR_FN = {
    "ComputeNode": _deserialize_compute_attr,
    "StorageNode": _deserialize_storage_attr,
    "CommunicationNode": _deserialize_communication_attr
}

CMP_NODE_ATTR_FN = {
    "ComputeNode": compare_compute_attr,
    "StorageNode": compare_storage_attr,
    "CommunicationNode": compare_communication_attr
}