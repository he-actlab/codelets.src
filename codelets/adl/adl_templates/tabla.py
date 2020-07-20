from codelets.adl.base import ArchitectureDescription

def add_pu(arch, pes, ni_size, nw_size, nd_size):
    pass

def create_tabla(pus, pes, ni_size, nw_size, nd_size):
    tabla_adl = ArchitectureDescription("tabla")
    pes_per_pu = int(pus / pes)
    for i in range(pus):
        add_pu(tabla_adl, pes_per_pu, ni_size, nw_size, nd_size)


TABLA_DELAYS = {
    "PE": {"PEGB": 2, "PENB": 1, "NI": 1, "NW": 1, "ND": 1, "NM": 1, "PUGB": 2, "PUNB": 1, "PE": 1},
    "PEGB": {"PE": 2},
    "PENB": {"PE": 0},
    "PUGB": {"PE": 2},
    "PUNB": {"PE": 0},
    "NI": {"PE": 0},
    "NW": {"PE": 0},
    "ND": {"PE": 0},
    "NM": {"PE": 0},
}