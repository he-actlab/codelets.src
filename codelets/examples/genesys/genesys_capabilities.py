from codelets.adl.operand import Operand
from codelets.adl.operation.compute_op import Compute
from codelets.adl.operation.loop_op import Loop
from codelets.adl.operation.memory_op import Comm
from . import SIMD_NS, SIMD_OPCODE_BITWIDTH, OP_DTYPES, \
    OP_LOCATIONS, NS_BITWIDTH, NS_IDX_BITWIDTH

def sa_mvmul():
    w = Operand("W", "storage", OP_DTYPES, 0, value_names=["WBUF"], components=["WBUF"], data_dims=[32, 32])
    x = Operand("X", "storage", OP_DTYPES, 0, value_names=["IBUF"], components=["IBUF"], data_dims=[32])
    o = Operand("O", "storage", OP_DTYPES, 0, value_names=["OBUF"], components=["OBUF"], data_dims=[32])
    mvmul = Compute("mvmul", "systolic_array", [w, x], [o], [])

    return mvmul

def sa_mem():
    sources = ["DRAM"]
    # load_op = Comm("LD", )

def conv2d(node, hag):
    size = 32
    with Loop("oc", 0, "OC") as oc_t:
        with Loop("n", 0, "N") as n_t:
            with Loop("ic", 0, "IC") as ic_t:
                with Loop("kh", 0, "KH") as kh_t:
                    with Loop("kw", 0, "KW") as kw_t:

                        with Loop("y", 0, "OH") as y_t:
                            with Loop("x", 0, "OW") as x_t:
                                Comm("DRAM", "WBUF", size, [oc_t, ic_t, kh_t, kw_t], 0)

                                with Loop("oc", 0, "OC") as oc:
                                    with Loop("n", 0, "N") as n:
                                        with Loop("ic", 0, "IC") as ic:
                                            with Loop("kh", 0, "KH") as kh:
                                                with Loop("kw", 0, "KW") as kw:
                                                    with Loop("y", 0, "OH") as y:
                                                        with Loop("x", 0, "OW") as x:

                                                            Comm("WBUF", "systolic_array", size, [oc, ic, kh, kw], [0, 0, 0, 0])

                                                            Comm("DRAM", "IBUF", size, [n, ], )
                                                            # IBUF = Mem()
                                                            # WBUF = Mem()
                                                            Compute("MVMUL", hag.get_node("IBUF"), hag.get_node("WBUF"))
