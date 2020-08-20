import json
from codelets.adl.architecture_graph import ArchitectureGraph
from typing import Union
from codelets.adl.architecture_node import ArchitectureNode
from codelets.adl.communication_node import CommunicationNode
from codelets.adl.compute_node import ComputeNode
from codelets.adl.storage_node import StorageNode
from jsonschema import validate
from pathlib import Path

BaseArchNode = Union[ComputeNode, CommunicationNode, StorageNode]
CWD = Path(f"{__file__}").parent
JSON_SCHEMA_PATH = f"{CWD}/adl_schema.json"

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
    attr['capabilities'] = get_json_capabilities(node.get_capabilities())
    return key, attr

def get_json_capabilities(capability_list):
    caps = []
    for cname, c in capability_list.items():
        cap = {'op_name': cname}
        cap['latency'] = c['latency']
        cap['input_nodes'], cap['output_nodes'] = get_capability_operands(c)
        cap['subcapabilities'] = get_json_capabilities(c['subcapabilities'])
        caps.append(cap)
    return caps

def get_capability_operands(capability):
    inputs = []
    for inode in capability['inputs']:
        json_operand = {}
        json_operand['node_id'] = inode['node_id']
        json_operand['dtype'] = inode['dtype']
        json_operand['dimensions'] = inode['dimensions']
        inputs.append(json_operand)

    outputs = []
    for inode in capability['outputs']:
        json_operand = {}
        json_operand['node_id'] = inode['node_id']
        json_operand['dtype'] = inode['dtype']
        json_operand['dimensions'] = inode['dimensions']
        outputs.append(json_operand)

    return inputs, outputs

def get_storage_node_attr(node: StorageNode):
    key = 'storage_node'
    attr = {}
    attr['capacity'] = node.get_capacity()
    attr['read_bw'] = node.get_read_bw()
    attr['write_bw'] = node.get_write_bw()
    attr['access_type'] = node.get_access_type()
    return key, attr

def get_communication_node_attr(node: CommunicationNode):
    attr = {}
    key = 'communication_node'
    attr['comm_type'] = node.get_comm_type()
    attr['bandwidth'] = node.get_bandwidth()

    return key, attr

def _serialize_node(node: Union[StorageNode, CommunicationNode, ComputeNode]):
    json_node = {}
    json_node['node_id'] = node.index
    json_node['name'] = node.name
    json_node['node_type'] = node.get_type()
    # json_node['type_attributes'] = NODE_TYPE_ATTR_FN[node.get_type()](node)
    type_key, type_attr = NODE_TYPE_ATTR_FN[node.get_type()](node)
    json_node[type_key] = type_attr
    json_node['attributes'] = node.get_all_attributes()
    json_node['subgraph'] = {"nodes": [], "edges": []}
    for n in node.get_subgraph_nodes():
        sub_node = _serialize_node(n)
        json_node['subgraph']['nodes'].append(sub_node)

    for e in node.get_subgraph_edges():
        sub_edge = {'src': e.src, 'dst': e.dst, 'attributes': e.attributes}
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
    return json_cap

def _deserialize_compute_attr(json_attr_):
    key = 'compute_node'
    json_attr = json_attr_[key]
    capabilities = {}
    for cap_name, cap in json_attr['capabilities']:
        capabilities[cap_name] = _deserialize_capability(cap)

    return (capabilities,)

def _deserialize_storage_attr(json_attr_):
    key = 'storage_node'
    json_attr = json_attr_[key]
    return (json_attr['read_bw'], json_attr['write_bw'], json_attr['access_type'], json_attr['capacity'])

def _deserialize_communication_attr(json_attr_):
    key = 'communication_node'
    json_attr = json_attr_[key]
    return (json_attr['comm_type'], json_attr['bandwidth'])

def _deserialize_node(json_node):
    node_name = json_node['name']
    node_id = json_node['node_id']
    node_type = json_node['node_type']
    # init_args = DESER_NODE_ATTR_FN[node_type](json_node['type_attributes'])
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
        if key == "out_degree":
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
        print(f"Unequal parents for {graph1.name}:\n"
              f"Graph1: {graph1._has_parent}\n"
              f"Graph2: {graph2._has_parent}")
        return False
    elif not CMP_NODE_ATTR_FN[graph1.get_type()](graph1, graph2):
        print(f"Uequal type specific attributes for {graph1.name} ({graph1.get_type()})")
        return False
    else:
        graph1_attr = {attr['key']: attr['value'] for attr in graph1.get_all_attributes()}
        graph2_attr = {attr['key']: attr['value'] for attr in graph2.get_all_attributes()}
        if not compare_attributes(graph1_attr, graph2_attr):
            print(f"Unequal attributes for {graph2.name}:\n"
                  f"Graph1: {json.dumps(graph2_attr, indent=4)}\n\n"
                  f"Graph2: {json.dumps(graph1_attr, indent=4)}")
    g1_edges = {(e.src, e.dst): e.attributes for e in graph1.get_subgraph_edges()}
    g2_edges = {(e.src, e.dst): e.attributes for e in graph2.get_subgraph_edges()}
    if len(g1_edges) != len(g2_edges) or set(g1_edges.keys()) != set(g2_edges.keys()):
        print(f"Unequal edges for {graph1.name}")
        return False

    for k,v in g1_edges.items():
        e1_attr = {attr['key']: attr['value'] for attr in v}
        e2_attr = {attr['key']: attr['value'] for attr in g2_edges[k]}
        edge_eq = compare_attributes(e1_attr, e2_attr)
        if not edge_eq:
            print(f"Unequal attributes for {graph2.name} in edge {k}:\n"
                  f"Graph1: {json.dumps(v, indent=4)}\n\n"
                  f"Graph2: {json.dumps(g2_edges[k], indent=4)}")
            return False

    node_names = [n.name for n in graph1.get_subgraph_nodes()]
    for nid in node_names:
        is_eq = compare_graphs(graph1.get_subgraph_node(nid), graph2.get_subgraph_node(nid))
        if not is_eq:
            return is_eq

    return True

def compare_compute_attr(node1: ComputeNode, node2: ComputeNode):
    cap1 = node1.get_capabilities()
    cap2 = node2.get_capabilities()
    # TODO: Update this with proper capability comparison
    if set(cap1.keys()) != set(cap2.keys()):
        return False
    return True

def compare_storage_attr(node1: StorageNode, node2: StorageNode):
    if node1.get_read_bw() != node2.get_read_bw():
        return False
    elif node1.get_write_bw() != node2.get_write_bw():
        return False
    elif node1.get_access_type() != node2.get_access_type():
        return False
    elif node1.get_capacity() != node2.get_capacity():
        return False
    return True

def compare_communication_attr(node1: CommunicationNode, node2: CommunicationNode):
    if node1.get_comm_type() != node2.get_comm_type():
        return False
    elif node1.get_bandwidth() != node2.get_bandwidth():
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