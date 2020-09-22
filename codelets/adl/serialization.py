import json
from codelets.adl.architecture_graph import ArchitectureGraph
from typing import Union
from codelets.adl.architecture_node import ArchitectureNode
from codelets.adl.communication_node import CommunicationNode
from codelets.adl.compute_node import ComputeNode
from codelets.adl.storage_node import StorageNode
from codelets.adl.capability import Capability
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
    cap_names = node.get_capabilities()
    cap_dict = {cname: node.get_capability(cname) for cname in cap_names}
    attr['capabilities'] = get_json_capabilities(cap_dict)
    return key, attr

def get_json_capabilities(capabilities):
    caps = []
    for cname, c in capabilities.items():
        cap = {'op_name': cname}
        cap['latency'] = c.get_latency()
        cap['input_nodes'], cap['output_nodes'] = get_capability_operands(c)
        cap_dict = {k.get_name(): k for k in c.get_sub_capabilities()}
        cap['subcapabilities'] = get_json_capabilities(cap_dict)
        caps.append(cap)
    return caps

def get_capability_operands(capability):
    inputs = []
    for inode in capability.get_inputs():
        inp = capability.get_input(inode)
        json_operand = {}
        json_operand['name'] = inode
        json_operand['src'] = inp['src']
        #TODO: Change to create dims
        json_operand['dimensions'] = list(inp['dims'])
        inputs.append(json_operand)

    outputs = []
    for onode in capability.get_outputs():
        out = capability.get_output(onode)
        json_operand = {}
        json_operand['name'] = onode
        json_operand['dst'] = out['dst']

        #TODO: Change to create dims
        json_operand['dimensions'] = list(out['dims'])
        outputs.append(json_operand)

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
    cap = Capability(json_cap['op_name'])
    cap.set_latency(json_cap['latency'])
    for i in json_cap['input_nodes']:
        cap.add_input(name=i['name'], src=i['src'], dims=i['dimensions'])

    for o in json_cap['output_nodes']:
        cap.add_output(name=o['name'], dst=o['dst'], dims=o['dimensions'])
    return cap

def _deserialize_compute_attr(json_attr_):
    key = 'compute_node'
    json_attr = json_attr_[key]
    capabilities = []
    for cap in json_attr['capabilities']:
        capabilities.append(_deserialize_capability(cap))
    return (capabilities,)

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
        if c1.get_latency() != c2.get_latency():
            return False
        c1_inputs = [c1.get_input(in_name) for in_name in c1.get_inputs()]
        c2_inputs = [c2.get_input(in_name) for in_name in c2.get_inputs()]

        c1_outputs = [c1.get_output(out_name) for out_name in c1.get_outputs()]
        c2_outputs = [c2.get_output(out_name) for out_name in c2.get_outputs()]
        if not compare_cap_inputs(c1_inputs, c2_inputs):
            return False
        elif not compare_cap_outputs(c1_outputs, c2_outputs):
            return False
        sub_caps1 = {sc1.get_name(): sc1 for sc1 in c1.get_sub_capabilities()}
        sub_caps2 = {sc1.get_name(): sc1 for sc1 in c1.get_sub_capabilities()}
        if not compare_capabilities(sub_caps1, sub_caps2):
            return False

    return True

def compare_compute_attr(node1: ComputeNode, node2: ComputeNode):
    cap1_names = node1.get_capabilities()
    caps1 = {cname: node1.get_capability(cname) for cname in cap1_names}
    cap2_names = node2.get_capabilities()
    caps2 = {cname: node2.get_capability(cname) for cname in cap2_names}
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