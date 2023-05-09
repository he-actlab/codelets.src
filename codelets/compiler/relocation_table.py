from typing import Dict, Union
from dataclasses import dataclass, field
import polymath as pm
import numpy as np
from codelets.adl.graph import StorageNode
from codelets.codelet_impl.codelet import Codelet
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


class RelocationTable(object):
    MEM_NS_MAPPING = {'VMEM1': 'VMEM', 'VMEM2': 'VMEM', 'IBUF': 'SA_INPUTS', 'WBUF': 'SA_INPUTS',
                      'BBUF': 'SA_INPUTS', 'OBUF': 'SA_OUTPUTS', 'INSTR_MEM': 'INSTR_MEM'}
    MEM_LAYOUT = ['SA_INPUTS', 'INSTR_MEM', 'SA_OUTPUTS', 'VMEM']

    # MEM_LAYOUT = ['INSTR_MEM',  'OUTPUTS', 'STATE', 'INTERMEDIATE', 'INPUTS']
    def __init__(self, storage_node: StorageNode, mem_layout=None, offsets=None, addr_alignment=1):
        self._top_level_node = storage_node.name
        self._mem_layout = mem_layout or RelocationTable.MEM_LAYOUT
        if offsets:
            self._offset_type = "static"
            assert list(offsets.keys()) == self.mem_layout
            self._namespace_offsets = {ml: off for ml, off in offsets.items()}
        else:
            self._offset_type = "dynamic"
            self._namespace_offsets = None

        self.addr_alignment = addr_alignment

        assert isinstance(self.mem_layout, list)
        self._relocatables = {}
        self._storage_node = storage_node
        for ml in self.mem_layout:
            self._relocatables[ml] = Relocation(ml)


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

    @property
    def offset_type(self):
        return self._offset_type

    @property
    def is_empty(self):
        return all([self.relocatables[ml].total_length() == 0 for ml in self.mem_layout])

    def get_relocation_by_name(self, name: str):
        for k, v in self.relocatables.items():
            if name in v.item_names():
                return v[name]
        raise KeyError(f"Unable to find relocation for {name}")

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

    def get_location_start_addr(self, ns, item_name):
        reloc = self.relocatables[ns]
        assert item_name in reloc.bases
        return reloc.bases[item_name].start

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

    def get_namespace_size(self, namespace: str):
        reloc = self.relocatables[namespace]
        return reloc.total_length()


    def get_relocation_base(self, operand: "Operand"):
        namespace = self.get_operand_namespace(operand)
        object_offset = self.get_relocation(namespace, operand.node_name).start

        assert object_offset % self.storage_node.width == 0, f"Invalid offset for address retrieval:\n" \
                                                           f"Storage width: {self.storage_node.width}\n" \
                                                           f"Namespace {namespace} offset: {object_offset}\n" \
                                                           f"Object {operand.node_name} offset: {object_offset}"

        return self.storage_node.address_from_bits(object_offset)

    def get_aligned_sized(self, size, as_bytes=True):
        if as_bytes:
            alignment = self.addr_alignment/8
        else:
            alignment = self.addr_alignment
        return alignment*ceil(size/alignment)

    # Mem notes:
    # instr_mem --> first
    # inputs: input, weight, bias
    #

    def update_relocation_offset(self, offset_type, offset_id, size):
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

    def get_operand_namespace(self, operand):
        for mem_loc in operand.data_path:
            if mem_loc in RelocationTable.MEM_NS_MAPPING:
                return RelocationTable.MEM_NS_MAPPING[mem_loc]
        raise RuntimeError(f"Unable to find namespace for operand {operand.node_name} with path {operand.data_path}")

    def add_data_relocation(self, node: pm.Node, cdlt: Codelet):

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
