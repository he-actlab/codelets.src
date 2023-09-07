import abc
from typing import Any, Dict, Final, Optional, Union
from dataclasses import dataclass, field
from collections import deque
import polymath as pm
import numpy as np
from codelets.adl.graph import StorageNode
from codelets.codelet_impl.codelet import Codelet, Operand
from math import ceil

@dataclass
class Fragment:
    offset_id: Union[int, str]
    size: int
    start: int
    end: int

    def __post_init__(self):
        assert (self.end - self.start) >= self.size

# TODO: Add datatype
@dataclass
class Relocation:
    offset_type: str
    bases: Dict[Union[int, str], Fragment] = field(default_factory=dict)

    def __getitem__(self, item):
        return self.bases[item]

    def get_fragment_str(self, item):
        return f"$({self.offset_type}[{self.bases[item]}])"

    def get_absolute_address(self, item):
        return self.bases[item]

    def item_names(self):
        return list(self.bases.keys())

    def total_length(self):
        if len(self.bases.keys()) == 0:
            return 0
        else:
            return max([v.end for v in self.bases.values()])


class RelocationTable(abc.ABC):
    _DEFAULT_ADDR_ALIGNMENT: Final[int] = 1

    _top_level_node: Any
    _storage_node: StorageNode
    _mem_layout: list[str]
    _mem_ns_mapping: dict[str, str]
    _relocatables: dict[str, Relocation]
    _addr_alignment: int

    def __init__(self, storage_node: StorageNode, mem_layout: list[str], mem_ns_mapping: dict[str, str], addr_alignment: Optional[int] = None) -> None:
        super().__init__()
        self._top_level_node = storage_node.name
        self._storage_node = storage_node
        self._mem_layout = mem_layout.copy()
        self._mem_ns_mapping = mem_ns_mapping.copy()

        assert isinstance(self.mem_layout, list)
        self._relocatables = {}
        for ml in self.mem_layout:
            self._relocatables[ml] = Relocation(ml)

        self._addr_alignment = addr_alignment or RelocationTable._DEFAULT_ADDR_ALIGNMENT

    @property
    def storage_node(self) -> StorageNode:
        return self._storage_node
    
    @property
    def mem_layout(self) -> list[str]:
        return self._mem_layout.copy()
    
    @property
    def mem_ns_mapping(self) -> dict[str, str]:
        return self._mem_ns_mapping.copy()
    
    @property
    def relocatables(self) -> dict[str, Relocation]:
        return self._relocatables.copy()
    
    @property
    def addr_alignment(self) -> int:
        return self._addr_alignment
    
    def get_aligned_sized(self, size: int, as_bytes: bool = True) -> int:
        if as_bytes:
            alignment: int = self.addr_alignment // 8
        else:
            alignment: int = self.addr_alignment
        return alignment * ceil(size / alignment)
    
    def get_relocation_base(self, operand: Operand) -> int:
        namespace = self.get_operand_namespace(operand)
        object_offset = self.get_relocation(namespace, operand.node_name).start

        assert object_offset % self.storage_node.width == 0, f"Invalid offset for address retrieval:\n" \
                                                           f"Storage width: {self.storage_node.width}\n" \
                                                           f"Namespace {namespace} offset: {object_offset}\n" \
                                                           f"Object {operand.node_name} offset: {object_offset}"

        return self.storage_node.address_from_bits(object_offset)
    
    def get_location_start_addr(self, ns, item_name) -> int:
        reloc: Relocation = self.relocatables[ns]
        assert item_name in reloc.bases
        return reloc.bases[item_name].start
    
    def get_relocation(self, namespace: str, item_id: Union[str, int]) -> Fragment:
        return self.relocatables[namespace][item_id]
    
    def get_relocation_by_name(self, name: str) -> Fragment:
        for _, v in self.relocatables.items():
            if name in v.item_names():
                return v[name]
        raise KeyError(f"Unable to find relocation for {name}")
    
    def get_namespace_by_name(self, name: str) -> str:
        for k, v in self.relocatables.items():
            if name in v.item_names():
                return k
        raise KeyError(f"Unable to find relocation for {name}")
    
    def get_operand_namespace(self, operand: Operand) -> str:
        for mem_loc in operand.data_path:
            if mem_loc in self.mem_ns_mapping:
                return self.mem_ns_mapping[mem_loc]
        raise RuntimeError(f"Unable to find namespace for operand {operand.node_name} with path {operand.data_path}")
    
    def get_namespace_size(self, namespace: str) -> int:
        reloc: Relocation = self.relocatables[namespace]
        return reloc.total_length()
    
    @abc.abstractmethod
    def add_data_relocation(self, node: pm.Node, cdlt: Codelet) -> None:
        ...
    
    @abc.abstractmethod
    def update_relocation_offset(self, offset_type: str, offset_id: Union[int, str], size: int, **kwargs: Any) -> None:
        ...
    
    @abc.abstractmethod
    def get_input_namespace_size(self) -> int:
        ...


class DebugRelocationTable(RelocationTable):
    MEM_NS_MAPPING = {'VMEM1': 'VMEM', 'VMEM2': 'VMEM', 'IBUF': 'SA_INPUTS', 'WBUF': 'SA_INPUTS',
                      'BBUF': 'SA_INPUTS', 'OBUF': 'SA_OUTPUTS', 'INSTR_MEM': 'INSTR_MEM'}
    MEM_LAYOUT = ['SA_INPUTS', 'INSTR_MEM', 'SA_OUTPUTS', 'VMEM']

    def __init__(self, storage_node: StorageNode, mem_layout: Optional[list[str]] = None, offsets=None, addr_alignment: Optional[int] = None) -> None:
        super().__init__(storage_node, mem_layout or DebugRelocationTable.MEM_LAYOUT, DebugRelocationTable.MEM_NS_MAPPING, addr_alignment=addr_alignment)
        if offsets:
            self._offset_type = "static"
            assert list(offsets.keys()) == self.mem_layout
            self._namespace_offsets = {ml: off for ml, off in offsets.items()}
        else:
            self._offset_type = "dynamic"
            self._namespace_offsets = None

    def __repr__(self):
        return str([(k, list(v.item_names())) for k, v in self.relocatables.items()])    

    @property
    def namespace_offsets(self):
        return self._namespace_offsets

    @property
    def offset_type(self):
        return self._offset_type

    @property
    def is_empty(self):
        return all([self.relocatables[ml].total_length() == 0 for ml in self.mem_layout]) 
    
    def get_input_namespace_size(self) -> int:
        return self.get_namespace_size('SA_INPUTS')

    def print_layout(self):
        base = 0
        for ns in self.mem_layout:
            reloc = self.relocatables[ns]

            aligned_size = self.get_aligned_sized(reloc.total_length()/8)
            aligned_end = aligned_size + base
            ref_end = self.get_namespace_offset(ns) / 8
            assert ref_end == base, f"Unequal offset and computed offset for {ns}:" \
                                            f"\n{ref_end} != {base}"
            if len(reloc.bases.keys()) == 0:
                continue
            print(f"{ns}: {base} --> {aligned_end}=============================")
            for key, item in reloc.bases.items():
                byte_start, byte_end = item.start / 8, item.end / 8
                print(f"\t{key}[size={item.size/8}]: {byte_start} --> {byte_end}")
            print(f"====================================================")
            base += aligned_size 

    def update_namespace_offsets(self):
        offset = 0

        if self.offset_type == "dynamic":
            return
            # for ns in self.mem_layout:
            #     self.namespace_offsets[ns] = offset
            #     offset += self.relocatables[ns].total_length()
        else:
            for i, ns in enumerate(self.mem_layout):
                if offset > self.namespace_offsets[ns]:
                    assert i > 0, f"Something is broken"
                    raise RuntimeError(f"Storage in {self.mem_layout[i - 1]} overruns {ns} memory.")
                offset += self.relocatables[ns].total_length()


    def get_namespace_offset(self, namespace: str):
        if self.offset_type == "static":
            assert namespace in self.mem_layout and namespace in self.namespace_offsets
            return self.namespace_offsets[namespace]
        else:
            assert namespace in self.relocatables
            off = 0
            for ns in self.mem_layout:
                if ns == namespace:
                    return off
                else:
                    reloc = self.relocatables[ns]
                    off += reloc.total_length()
            raise RuntimeError(f"Unable to find {namespace} in relocatables")

    # Mem notes:
    # instr_mem --> first
    # inputs: input, weight, bias
    #

    def update_relocation_offset(self, offset_type: str, offset_id: Union[int, str], size: int, **kwargs: Any) -> None:
        aligned_size = self.get_aligned_sized(size, as_bytes=False)
        current_offset = self.relocatables[offset_type].total_length()
        relocatable = self.relocatables[offset_type]
        if offset_id not in self.relocatables[offset_type].bases:
            relocatable.bases[offset_id] = Fragment(offset_id, size, current_offset, current_offset + aligned_size)
        elif offset_type in ['INPUTS', 'OUTPUTS']:
            # return
            raise RuntimeError(f"Existing relocation offset for offset type {offset_type} for operand {offset_id} found")
        else:
            # TODO: Need to add back in error handling here for the same data with different datatypes

            stored_size = relocatable.bases[offset_id].end - relocatable.bases[offset_id].start
            if stored_size < aligned_size:
                prev_fragment = relocatable.bases[offset_id]
                relocatable.bases[offset_id] = Fragment(offset_id, size, prev_fragment.start, prev_fragment.start + aligned_size)
        self.update_namespace_offsets()

    def add_data_relocation(self, node: pm.Node, cdlt: Codelet) -> None:

        for idx, operand in enumerate(cdlt.inputs):
            i = node.inputs[idx]
            data_size = np.prod(operand.shape)*operand.dtype.bits()
            ns = self.get_operand_namespace(operand)
            self.update_relocation_offset(ns, i.name, data_size)

        for idx, operand in enumerate(cdlt.outputs):
            o = node.outputs[idx]
            data_size = np.prod(operand.shape)*operand.dtype.bits()
            ns = self.get_operand_namespace(operand)
            self.update_relocation_offset(ns, o.name, data_size)

        self.update_namespace_offsets()


class _DataflowGraphNode:
    _next_id: int = 0

    _id: int
    _name: str

    def __init__(self, name: str) -> None:
        self._id = _DataflowGraphNode._next_id
        _DataflowGraphNode._next_id += 1
        self._name = name
    
    @property
    def id(self) -> int:
        return self._id
    
    @property
    def name(self) -> str:
        return self._name
    
    def __hash__(self) -> int:
        return self._id
    
    def __eq__(self, o: object) -> bool:
        if isinstance(o, _DataflowGraphNode):
            return self._id == o._id
        return False
    
    def __str__(self) -> str:
        return f"Node {self._id}: {self._name}"
    
    def __repr__(self) -> str:
        return f"_DataflowGraphNode(id={self.id}, name={self.name}))"


class _DataflowGraphEdge:
    _next_id: int = 0

    _id: int
    _destination_node: Optional[_DataflowGraphNode]
    _operand_name: str
    _operand: Operand

    def __init__(self, operand_name: str, operand: Operand, destination_node: Optional[_DataflowGraphNode]) -> None:
        self._id = _DataflowGraphEdge._next_id
        _DataflowGraphEdge._next_id += 1
        self._destination_node = destination_node
        self._operand_name = operand_name
        self._operand = operand
    
    @property
    def id(self) -> int:
        return self._id
    
    @property
    def operand_name(self) -> str:
        return self._operand_name
    
    @property
    def operand(self) -> Operand:
        return self._operand
    
    @property
    def destination_node(self) -> Optional[_DataflowGraphNode]:
        return self._destination_node
    
    def __repr__(self) -> str:
        return f"_DataflowGraphEdge(id={self.id}, operand_name={self.operand_name}, operand={self.operand.name}, destination_node={repr(self.destination_node)})"


class _DataflowGraph:
    _inputs: list[_DataflowGraphEdge]
    _graph: dict[_DataflowGraphNode, list[_DataflowGraphEdge]]

    def __init__(self) -> None:
        self._inputs = []
        self._graph = {}
    
    def _get_operands_stored_at_each_layer(self) -> dict[_DataflowGraphNode, list[str]]:
        operands_stored_at_each_layer: dict[_DataflowGraphNode, list[str]] = {}

        topological_order: list[_DataflowGraphNode] = self.topological_sort()
        incoming_edges: list[list[_DataflowGraphEdge]] = []
        are_incoming_edges_from_previous_node: list[list[bool]] = []
        outgoing_edges: list[list[_DataflowGraphEdge]] = []
        for i, node in enumerate(topological_order):
            if i == 0:
                incoming_edges.append([input_edge for input_edge in self._inputs if input_edge.destination_node is node])
                are_incoming_edges_from_previous_node.append([True for _ in incoming_edges[-1]])
            else:
                current_node_incoming_edges: list[_DataflowGraphEdge] = []
                current_node_are_incoming_edges_from_previous_node: list[bool] = []

                current_node_input_incoming_edges: list[_DataflowGraphEdge]= [input_edge for input_edge in self._inputs if input_edge.destination_node is node]
                current_node_are_input_incoming_edges_from_previous_node: list[bool] = [False for _ in current_node_input_incoming_edges]

                current_node_previous_non_neighbor_incoming_edges: list[_DataflowGraphEdge] = []
                current_node_are_previous_non_neightbor_incoming_edges_from_previous_node: list[bool] = []
                for previous_node in topological_order[:i - 1]:
                    current_node_previous_non_neighbor_incoming_edges = [edge for edge in self._graph[previous_node] if edge.destination_node is node]
                    current_node_are_previous_non_neightbor_incoming_edges_from_previous_node = [False for _ in current_node_previous_non_neighbor_incoming_edges]

                current_node_previous_incoming_edges: list[_DataflowGraphEdge] = [edge for edge in self._graph[topological_order[i - 1]] if edge.destination_node is node]
                current_node_are_previous_incoming_edges_from_previous_node: list[bool] = [True for _ in current_node_previous_incoming_edges]

                current_node_incoming_edges.extend(current_node_input_incoming_edges)
                current_node_incoming_edges.extend(current_node_previous_non_neighbor_incoming_edges)
                current_node_incoming_edges.extend(current_node_previous_incoming_edges)

                current_node_are_incoming_edges_from_previous_node.extend(current_node_are_input_incoming_edges_from_previous_node)
                current_node_are_incoming_edges_from_previous_node.extend(current_node_are_previous_non_neightbor_incoming_edges_from_previous_node)
                current_node_are_incoming_edges_from_previous_node.extend(current_node_are_previous_incoming_edges_from_previous_node)

                incoming_edges.append(current_node_incoming_edges)
                are_incoming_edges_from_previous_node.append(current_node_are_incoming_edges_from_previous_node)
            outgoing_edges.append(self._graph[node].copy())
        assert len(incoming_edges) == len(are_incoming_edges_from_previous_node)
        assert all(len(incoming_edges[i]) == len(are_incoming_edges_from_previous_node[i]) for i in range(len(incoming_edges)))
        assert len(incoming_edges) == len(outgoing_edges)
        assert len(incoming_edges) == len(topological_order)
 
        current_residual_operands: set[str] = set()
        for node, ((node_incoming_edges, is_incoming_edge_from_previous_node), node_outgoing_edges) in zip(reversed(topological_order), zip(zip(reversed(incoming_edges), reversed(are_incoming_edges_from_previous_node)), reversed(outgoing_edges))):
            operands_stored_at_each_layer[node] = list()
            for edge, is_edge_from_previous_node in zip(node_incoming_edges, is_incoming_edge_from_previous_node):
                if not is_edge_from_previous_node:
                    current_residual_operands.add(edge.operand_name)
                operands_stored_at_each_layer[node].append(edge.operand_name)
            for edge in node_outgoing_edges:
                if edge.operand_name in current_residual_operands:
                    current_residual_operands.remove(edge.operand_name)
                operands_stored_at_each_layer[node].append(edge.operand_name)
            operands_stored_at_each_layer[node].extend(current_residual_operands)
        
        return operands_stored_at_each_layer

    def topological_sort(self) -> list[_DataflowGraphNode]:
        indegree: dict[_DataflowGraphNode, int] = self.find_indegrees()
        queue = deque([node for node, indeg in indegree.items() if indeg == 0])
        top_order: list[_DataflowGraphNode] = []
        
        while queue:
            node = queue.popleft()
            top_order.append(node)
            for neighbor in self._graph.get(node, []):
                if neighbor.destination_node is not None:
                    indegree[neighbor.destination_node] -= 1
                    if indegree[neighbor.destination_node] == 0:
                        queue.append(neighbor.destination_node)
        return top_order
    
    def find_indegrees(self) -> dict[_DataflowGraphNode, int]:
        indegree: dict[_DataflowGraphNode, int] = {node: 0 for node in self._graph}
        for edges in self._graph.values():
            for edge in edges:
                if edge.destination_node is not None:
                    indegree[edge.destination_node] += 1
        return indegree

    def append_node(self, node: _DataflowGraphNode, inputs: list[tuple[str, Operand]], outputs: list[tuple[str, Operand]]) -> None:
        self.add_node(node)
        for input_name, input_operand in inputs:
            source_node: Optional[_DataflowGraphNode] = self.get_node_operand_is_output_of(input_name)
            if source_node is None:
                self.add_input_edge(_DataflowGraphEdge(input_name, input_operand, node))
            else:
                for edge in self._graph[source_node]:
                    if edge.operand_name == input_name:
                        edge._destination_node = node
                        break
        
        for output_name, output_operand in outputs:
            self.add_edge(node, _DataflowGraphEdge(output_name, output_operand, None))
    
    def add_node(self, node: _DataflowGraphNode) -> None:
        self._graph[node] = []

    def add_input_edge(self, edge: _DataflowGraphEdge) -> None:
        self._inputs.append(edge)
    
    def add_edge(self, source_node: _DataflowGraphNode, edge: _DataflowGraphEdge) -> None:
        self._graph[source_node].append(edge)
    
    def get_edges(self, source_node: _DataflowGraphNode) -> list[_DataflowGraphEdge]:
        return self._graph[source_node]
    
    def get_nodes(self) -> list[_DataflowGraphNode]:
        return list(self._graph.keys())
    
    def get_input_edges(self) -> list[_DataflowGraphEdge]:
        return self._inputs.copy()
    
    def get_output_edges(self) -> list[_DataflowGraphEdge]:
        return [edge for edges in self._graph.values() for edge in edges if edge.destination_node is None]
    
    def get_node_operand_is_output_of(self, operand_name: str) -> Optional[_DataflowGraphNode]:
        for node, edges in self._graph.items():
            for edge in edges:
                if edge.operand_name is operand_name:
                    return node
        return None
    
    def __str__(self) -> str:
        ret: str = ""
        topological_order: list[_DataflowGraphNode] = self.topological_sort()
        visited = set()
        
        def format_node_string(node, indent: int = 0):
            if node in visited:
                return ""
            node_string: str = ""
            visited.add(node)
            node_string += "  " * indent + str(node) + "\n"
            for neighbor in self._graph.get(node, []):
                node_string += "  " * (indent + 1) + str(node) + " -(" + neighbor.operand_name + ")-> " + str(neighbor.destination_node) + "\n"
                node_string += format_node_string(neighbor.destination_node, indent + 2)
            
            return node_string
        
        for node in topological_order:
            ret += format_node_string(node)
        
        return ret


class EndToEndRelocationTable(RelocationTable):
    MEM_NS_MAPPING: dict[str, str] = {'VMEM1': 'ACTIVATION', 'VMEM2': 'ACTIVATION', 'IBUF': 'ACTIVATION', 'WBUF': 'WEIGHT_AND_BIAS',
                      'BBUF': 'WEIGHT_AND_BIAS', 'OBUF': 'ACTIVATION', 'INSTR_MEM': 'INSTR_MEM'}
    MEM_LAYOUT: list[str] = ['ACTIVATION', 'WEIGHT_AND_BIAS', 'INSTR_MEM']

    _dataflow_graph: _DataflowGraph
    _operand_name_to_operand_map: dict[str, Operand]
    _maximum_activation_buffer_size: int

    _current_aligned_offset_for_activation_buffer: int = 0

    def __init__(self, storage_node: StorageNode, mem_layout: Optional[list[str]] = None, offsets=None, addr_alignment=1) -> None:
        super().__init__(storage_node, mem_layout or EndToEndRelocationTable.MEM_LAYOUT, EndToEndRelocationTable.MEM_NS_MAPPING)
        self._dataflow_graph = _DataflowGraph()
        self._operand_name_to_operand_map = {}
        self._maximum_activation_buffer_size = 0

        self._current_aligned_offset_for_activation_buffer = 0

    @property
    def maximum_activation_buffer_size(self) -> int:
        return self._maximum_activation_buffer_size
    
    def print_layout(self) -> None:
        print("====================================================")
        print("INFORMATION:")
        print("\t- Activation buffer size: " + str(self.maximum_activation_buffer_size // 8) + " bytes")
        print("====================================================")

        operands_stored_at_each_layer: dict[_DataflowGraphNode, str] = self._dataflow_graph._get_operands_stored_at_each_layer()
        for node in self._dataflow_graph.get_nodes():
            print("====================================================")
            print(f"{node.name}: {node.id}")
            print("====================================================")
            for ns in self.mem_layout:
                reloc: Relocation = self.relocatables[ns]
                sorted_fragments: list[tuple[Union[int, str], Fragment]] = sorted(reloc.bases.items(), key=lambda x: x[1].start)

                if ns == "ACTIVATION":
                    print("ACTIVATION_LOWER:") 
                    for operand_id, memory_fragment in sorted_fragments:
                        if operand_id in operands_stored_at_each_layer[node] and memory_fragment.start < (self.maximum_activation_buffer_size // 2):
                            byte_start, byte_end = memory_fragment.start // 8, memory_fragment.end // 8
                            print(f"\t{operand_id}[size={memory_fragment.size // 8}]: {byte_start} --> {byte_end}")
                    print("ACTIVATION_UPPER:") 
                    for operand_id, memory_fragment in sorted_fragments:
                        if operand_id in operands_stored_at_each_layer[node] and memory_fragment.start >= (self.maximum_activation_buffer_size // 2):
                            byte_start, byte_end = memory_fragment.start // 8, memory_fragment.end // 8
                            print(f"\t{operand_id}[size={memory_fragment.size // 8}]: {byte_start} --> {byte_end}")
                else:
                    print(f"{ns}:") 
                    for operand_id, memory_fragment in sorted_fragments:
                        if operand_id in operands_stored_at_each_layer[node]:
                            byte_start, byte_end = memory_fragment.start // 8, memory_fragment.end // 8
                            print(f"\t{operand_id}[size={memory_fragment.size // 8}]: {byte_start} --> {byte_end}")
    
    def get_input_namespace_size(self) -> int:
        return self.get_namespace_size('ACTIVATION')

    def add_data_relocation(self, node: pm.Node, cdlt: Codelet) -> None:
        operation_node = _DataflowGraphNode(node.name)
        self._dataflow_graph.append_node(operation_node, [(i.name, inp) for i, inp in zip(node.inputs, cdlt.inputs)], [(o.name, out) for o, out in zip(node.outputs, cdlt.outputs)])
        for node_input, cdlt_input in zip(node.inputs, cdlt.inputs):
            self._operand_name_to_operand_map[node_input.name] = cdlt_input
        for node_output, cdlt_output in zip(node.outputs, cdlt.outputs):
            self._operand_name_to_operand_map[node_output.name] = cdlt_output
        self._update_maximum_activation_buffer_size()
        self._update_relocations()

        # print("\n\nNew operation added to graph...\n\n")
        # self.print_layout()
    
    def _update_maximum_activation_buffer_size(self) -> None:
        maximum_activation_buffer_size: int = 0
        operands_stored_at_each_layer: dict[_DataflowGraphNode, list[str]] = self._dataflow_graph._get_operands_stored_at_each_layer()
        for node in self._dataflow_graph.get_nodes():
            current_node_activation_buffer_size: int = 0
            operands_stored_at_node_layer: list[str] = operands_stored_at_each_layer[node]
            for operand_name in operands_stored_at_node_layer:
                operand: Operand = self._operand_name_to_operand_map[operand_name]
                operand_location: str = self.get_operand_namespace(operand)
                data_size: int = np.prod(operand.shape) * operand.dtype.bits() 
                if operand_location == "ACTIVATION":
                    current_node_activation_buffer_size += data_size
            maximum_activation_buffer_size = max(maximum_activation_buffer_size, current_node_activation_buffer_size)
        self._maximum_activation_buffer_size = 2 * self.get_aligned_sized(maximum_activation_buffer_size, as_bytes=False)

    def _update_relocations(self) -> None:
        is_first_half_of_activation_buffer: bool = True
        operands_stored_at_each_layer: dict[_DataflowGraphNode, set[str]] = self._dataflow_graph._get_operands_stored_at_each_layer()
        for node in self._dataflow_graph.get_nodes():
            self._reset_current_aligned_offset_for_activation_buffer()
            operands_stored_at_node_layer: set[str] = operands_stored_at_each_layer[node]
            for operand_name in operands_stored_at_node_layer:
                operand: Operand = self._operand_name_to_operand_map[operand_name]
                operand_location: str = self.get_operand_namespace(operand)
                data_size: int = np.prod(operand.shape) * operand.dtype.bits()
                self.update_relocation_offset(operand_location, operand_name, data_size, is_first_half_of_activation_buffer=is_first_half_of_activation_buffer)
            is_first_half_of_activation_buffer = not is_first_half_of_activation_buffer

    def update_relocation_offset(self, offset_type: str, offset_id: Union[int, str], size: int, **kwargs: Any) -> None:
        aligned_size: int = self.get_aligned_sized(size, as_bytes=False)

        if offset_type == "ACTIVATION":
            assert "is_first_half_of_activation_buffer" in kwargs
            is_first_half_of_activation_buffer: bool = kwargs["is_first_half_of_activation_buffer"]
            if is_first_half_of_activation_buffer:
                current_offset: int = self._get_current_aligned_offset_for_activation_buffer()
            else:
                current_offset: int = self.maximum_activation_buffer_size // 2 + self._get_current_aligned_offset_for_activation_buffer()
        else:
            current_offset: int = self.relocatables[offset_type].total_length()

        relocatable: Relocation = self.relocatables[offset_type]
        if offset_id not in self.relocatables[offset_type].bases:
            relocatable.bases[offset_id] = Fragment(offset_id, size, current_offset, current_offset + aligned_size)
            if offset_type == "ACTIVATION":
                self._increment_current_aligned_offset_for_activation_buffer(aligned_size) 

    def _reset_current_aligned_offset_for_activation_buffer(self) -> None:
        self._current_aligned_offset_for_activation_buffer = 0
    
    def _get_current_aligned_offset_for_activation_buffer(self) -> int:
        return self._current_aligned_offset_for_activation_buffer
    
    def _increment_current_aligned_offset_for_activation_buffer(self, increment: int) -> None:
        self._current_aligned_offset_for_activation_buffer += increment
