from typing import Dict, Union
from dataclasses import dataclass, field
import polymath as pm
import numpy as np
from codelets.adl.graph import StorageNode
from codelets.codelet_impl.codelet import Codelet

@dataclass
class Fragment:
    offset_id: Union[int, str]
    start: int
    end: int

# TODO: Add datatype
@dataclass
class Relocation:
    offset_type: str
    bases: Dict[Union[int, str], Fragment] = field(default_factory=dict)
    total_length: int = field(default=0)

    def __getitem__(self, item):
        return self.bases[item]

    def get_fragment_str(self, item):
        return f"$({self.offset_type}[{self.bases[item]}])"

    def get_absolute_address(self, item):
        return self.bases[item]

    def item_names(self):
        return list(self.bases.keys())



class RelocationTable(object):
    MEM_LAYOUT = ['INSTR_MEM', 'STATE', 'INTERMEDIATE']
    def __init__(self, storage_node: StorageNode, mem_layout=None):
        self._mem_layout = mem_layout or RelocationTable.MEM_LAYOUT
        assert isinstance(self.mem_layout, list)
        self._namespace_offsets = {}
        self._relocatables = {}
        self._storage_node = storage_node

        for ml in self.mem_layout:
            self._relocatables[ml] = Relocation(ml)
            self._namespace_offsets[ml] = 0

    def __repr__(self):
        return str([(k, list(v.item_names())) for k, v in self.relocatables.items()])

    @property
    def mem_layout(self):
        return self._mem_layout

    @property
    def relocatables(self):
        return self._relocatables

    @property
    def namespace_offsets(self):
        return self._namespace_offsets

    @property
    def storage_node(self):
        return self._storage_node

    def get_relocation_by_name(self, name: str):
        for k, v in self.relocatables.items():
            if name in v.item_names():
                return v[name]
        raise KeyError(f"Unable to find relocation for {name}")

    def get_namespace_by_name(self, name: str):
        for k, v in self.relocatables.items():
            if name in v.item_names():
                return k
        raise KeyError(f"Unable to find relocation for {name}")

    def get_base_by_name(self, name: str):
        namespace = self.get_namespace_by_name(name)
        return self.get_relocation_base(namespace, name)

    def get_relocation(self, namespace: str, item_id: Union[str, int]):
        return self.relocatables[namespace][item_id]

    def get_namespace_offset(self, namespace: str):
        assert namespace in self.mem_layout
        offset = 0
        for ns in self.mem_layout:
            if ns == namespace:
                return offset
            offset += self.relocatables[ns].total_length
        raise RuntimeError(f"Invalid namespace name: {namespace}")

    def get_relocation_base(self, namespace: str, item_id: Union[str, int]):
        namespace_offset = self.get_namespace_offset(namespace)
        object_offset = self.get_relocation(namespace, item_id).start
        total_bit_offset = (namespace_offset + object_offset)

        assert total_bit_offset % self.storage_node.width == 0, f"Invalid offset for address retrieval:\n" \
                                                           f"Storage width: {self.storage_node.width}\n" \
                                                           f"Namespace {namespace} offset: {namespace_offset}\n" \
                                                           f"Object {item_id} offset: {object_offset}"
        return (namespace_offset + object_offset) // self.storage_node.width

    def update_relocation_offset(self, offset_type, offset_id, size):
        current_offset = self.relocatables[offset_type].total_length
        relocatable = self.relocatables[offset_type]
        if offset_id not in self.relocatables[offset_type].bases:
            relocatable.bases[offset_id] = Fragment(offset_id, current_offset, current_offset + size)
            relocatable.total_length += size
        else:
            # TODO: Need to add back in error handling here for the same data with different datatypes
            stored_size = relocatable.bases[offset_id].end - relocatable.bases[offset_id].start
            if stored_size < size:
                prev_fragment = relocatable.bases[offset_id]
                relocatable.bases[offset_id] = Fragment(offset_id, prev_fragment.start, prev_fragment.start + size)
                relocatable.total_length += (size - stored_size)
            # assert stored_size == size

    def add_data_relocation(self, node: pm.Node, cdlt: Codelet):
        for idx, operand in enumerate(cdlt.inputs):
            i = node.inputs[idx]
            data_size = np.prod(operand.shape)*operand.dtype.bits()
            if isinstance(i, pm.state):
                offset_type = 'STATE'
            else:
                offset_type = 'INTERMEDIATE'
            self.update_relocation_offset(offset_type, i.name, data_size)

        for idx, operand in enumerate(cdlt.outputs):
            o = node.outputs[idx]
            data_size = np.prod(operand.shape)*operand.dtype.bits()
            offset_type = 'INTERMEDIATE'
            self.update_relocation_offset(offset_type, o.name, data_size)