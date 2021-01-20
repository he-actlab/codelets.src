from codelets.adl.operation import OperandTemplate, Loop
from codelets.adl.codelet import Codelet

from codelets.adl.graph import ArchitectureNode
import numpy as np
from . import OP_DTYPES

def mmmul():
    pass
    # A = OperandTemplate("A", OP_DTYPES, ["M", "N"])
    # B = OperandTemplate("B", OP_DTYPES, ["N", "P"])
    # C = OperandTemplate("C", OP_DTYPES, ["M", "P"])
    # with Codelet("mmul", [A, B], [C]) as cdlt:
    #     # Add setup op
    #     with Loop(0, "M_T") as m_t:
    #         with Loop(0, "N_T") as n_t:
    #             with Loop(0, "P_T") as p_t:

def conv2d_nhwc_test(hag: ArchitectureNode):
    data = OperandTemplate("data", OP_DTYPES, ["N", "IC", "H", "W"])
    weight = OperandTemplate("weight", OP_DTYPES, ["OC", "IC", "KH", "KW"])
    out = OperandTemplate("out", OP_DTYPES, ["N", "OC", "OH", "OW"])
    required_params = {}
    required_params['OW_TILE_SIZE'] = 1
    required_params['OH_TILE_SIZE'] = 1
    required_params['IC_TILE_SIZE'] = 1
    required_params['OC_TILE_SIZE'] = 1
    required_params['KW_TILE_SIZE'] = 1
    required_params['KH_TILE_SIZE'] = 1
    required_params['N_TILE_SIZE'] = 1
    with Codelet("conv", [data, weight], [out], required_params=required_params) as cdlt:
        cdlt.configure("start", "systolic_array")
        cdlt.configure("start", "WBUF")
        cdlt.configure("start", "IBUF")
        cdlt.configure("start", "OBUF")
        with Loop(0, "OC", stride="OC_TILE_SIZE") as oc:
            with Loop(0, "IC", stride="IC_TILE_SIZE") as ic:
                with Loop(0, "KH", stride="KH_TILE_SIZE") as kh:
                    with Loop(0, "KW", stride="KW_TILE_SIZE") as kw:
                        cdlt.transfer(weight, ["DRAM", "WBUF"], [[oc, ic, kh, kw], 0], [1])
    return cdlt



def conv2d_nhwc(hag: ArchitectureNode):
    # TODO: Need to figure out how to change the memory layout
    data = OperandTemplate("data", OP_DTYPES, ["N", "H", "W", "IC"])
    weight = OperandTemplate("weight", OP_DTYPES, ["OC", "IC", "KH", "KW"])
    out = OperandTemplate("out", OP_DTYPES, ["N", "OH", "OW", "OC"])
    required_params = {}
    required_params['OW_TILE_SIZE'] = 1
    required_params['OH_TILE_SIZE'] = 1
    required_params['IC_TILE_SIZE'] = 1
    required_params['OC_TILE_SIZE'] = 1
    required_params['KW_TILE_SIZE'] = 1
    required_params['KH_TILE_SIZE'] = 1
    required_params['N_TILE_SIZE'] = 1

    with Codelet("conv", [data, weight], [out], required_params=required_params) as cdlt:
        cdlt.configure("start", "systolic_array")
        cdlt.configure("start", "WBUF")
        cdlt.configure("start", "IBUF")
        cdlt.configure("start", "OBUF")
        with Loop(0, "OC", stride="OC_TILE_SIZE") as oc:
            with Loop(0, "N", stride="N_TILE_SIZE") as n:
                with Loop(0, "IC", stride="IC_TILE_SIZE") as ic:
                    with Loop(0, "KH", stride="KH_TILE_SIZE") as kh:
                        with Loop(0, "KW", stride="KW_TILE_SIZE") as kw:
                            with Loop(0, "OH", stride="OH_TILE_SIZE") as y:
                                with Loop(0, "OW", stride="OW_TILE_SIZE") as x:
                                    cdlt.transfer(weight, ["DRAM", "WBUF"], [[oc, ic, kh, kw], 0], [1])
                                    cdlt.transfer(data, ["DRAM", "IBUF"],
                                                  [[n, y*"stride" + kh, x*"stride" + kw, ic], 0], [1])
                                    cdlt.transfer(out, ["DRAM", "OBUF"], [[n, y, x, oc], 0], [1])
                                    with Loop(0, "OC_TILE_SIZE") as oc_t:
                                        with Loop(0, "N_TILE_SIZE") as n_t:
                                            with Loop(0, "IC_TILE_SIZE") as ic_t:
                                                with Loop(0, "KH_TILE_SIZE") as kh_t:
                                                    with Loop(0, "KW_TILE_SIZE") as kw_t:
                                                        with Loop(0, "OH_TILE_SIZE") as y_t:
                                                            with Loop(0, "OW_TILE_SIZE") as x_t:
                                                                wbuf_data_size = np.prod(hag.get_subgraph_node("systolic_array").dimensions)
                                                                ibuf_data_size = hag.get_subgraph_node("systolic_array").dimensions[0]
                                                                cdlt.transfer(weight, ["WBUF", "systolic_array"], [[oc_t, ic_t, kh_t, kw_t], 0], [wbuf_data_size])
                                                                cdlt.transfer(data, ["IBUF", "systolic_array"], [[n_t, y_t*"stride" + kh_t, x_t*"stride" + kw_t, ic_t], 0], [ibuf_data_size])
                                                                cdlt.transfer(out, ["systolic_array", "OBUF"], [[n_t, y_t, x_t, oc_t], 0], [ibuf_data_size])
                                                                cdlt.compute("MVMUL", ["IBUF", "WBUF"], ["OBUF"], target="systolic_array")

        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "systolic_array")
    return cdlt

def conv2d_nchw(hag: ArchitectureNode):
    # TODO: Need to figure out how to change the memory layout
    data = OperandTemplate("data", OP_DTYPES, ["N", "IC", "IH", "IW"])
    weight = OperandTemplate("weight", OP_DTYPES, ["OC", "IC", "KH", "KW"])
    out = OperandTemplate("out", OP_DTYPES, ["N", "OC", "OH", "OW"])
    required_params = {}
    required_params['OW_TILE_SIZE'] = lambda OW: OW
    required_params['OH_TILE_SIZE'] = lambda OH: OH
    required_params['IC_TILE_SIZE'] = lambda IC: IC
    required_params['OC_TILE_SIZE'] = lambda OC: OC
    required_params['KW_TILE_SIZE'] = lambda KW: KW
    required_params['KH_TILE_SIZE'] = lambda KH: KH
    required_params['N_TILE_SIZE'] = lambda N: N

    with Codelet("conv", [data, weight], [out], required_params=required_params) as cdlt:
        cdlt.add_required_param("IH_TILE_SIZE", lambda stride, pad, OH_TILE_SIZE, KH_TILE_SIZE:  (OH_TILE_SIZE - 1)*stride - 2*pad + KH_TILE_SIZE)
        cdlt.add_required_param("IW_TILE_SIZE", lambda stride, pad, OW_TILE_SIZE, KW_TILE_SIZE:  (OW_TILE_SIZE - 1)*stride - 2*pad + KW_TILE_SIZE)
        cdlt.configure("start", "systolic_array")
        cdlt.configure("start", "WBUF")
        cdlt.configure("start", "IBUF")
        cdlt.configure("start", "OBUF")
        with Loop(0, "OC", stride="OC_TILE_SIZE") as oc:
            with Loop(0, "N", stride="N_TILE_SIZE") as n:
                with Loop(0, "IC", stride="IC_TILE_SIZE") as ic:
                    with Loop(0, "KH", stride="KH_TILE_SIZE") as kh:
                        with Loop(0, "KW", stride="KW_TILE_SIZE") as kw:
                            with Loop(0, "OH", stride="OH_TILE_SIZE") as y:
                                with Loop(0, "OW", stride="OW_TILE_SIZE") as x:
                                    cdlt.transfer(weight, ["DRAM", "WBUF"], [[oc, ic, kh, kw], 0], [[oc.stride, ic.stride, kh.stride, kw.stride]])
                                    cdlt.transfer(data, ["DRAM", "IBUF"],
                                                  [[n, ic, y*"stride" + kh, x*"stride" + kw], 0], [[n.stride, ic.stride, "IH_TILE_SIZE", "IW_TILE_SIZE"]])
                                    cdlt.transfer(out, ["DRAM", "OBUF"], [[n, oc, y, x], 0], [[n.stride, oc.stride, y.stride, x.stride]])
                                    with Loop(0, "OC_TILE_SIZE") as oc_t:
                                        with Loop(0, "N_TILE_SIZE") as n_t:
                                            with Loop(0, "IC_TILE_SIZE") as ic_t:
                                                with Loop(0, "KH_TILE_SIZE") as kh_t:
                                                    with Loop(0, "KW_TILE_SIZE") as kw_t:
                                                        with Loop(0, "OH_TILE_SIZE") as y_t:
                                                            with Loop(0, "OW_TILE_SIZE") as x_t:
                                                                wbuf_data_size = np.prod(hag.get_subgraph_node("systolic_array").dimensions)
                                                                ibuf_data_size = hag.get_subgraph_node("systolic_array").dimensions[0]
                                                                cdlt.transfer(weight, ["WBUF", "systolic_array"], [[oc_t, ic_t, kh_t, kw_t], 0], [[wbuf_data_size]])
                                                                cdlt.transfer(data, ["IBUF", "systolic_array"], [[n_t, ic_t, y_t*"stride" + kh_t, x_t*"stride" + kw_t], 0], [[ibuf_data_size]])
                                                                cdlt.transfer(out, ["OBUF", "systolic_array"], [[n_t, oc_t, y_t, x_t], 0], [[ibuf_data_size]])
                                                                cdlt.compute("MVMUL", ["IBUF", "WBUF"], ["OBUF"], target="systolic_array")
                                                                cdlt.transfer(out, ["systolic_array", "OBUF"], [0, [n_t, oc_t, y_t, x_t]], [[ibuf_data_size]])


        # TODO: Add store off chip
        cdlt.configure("end", "WBUF")
        cdlt.configure("end", "IBUF")
        cdlt.configure("end", "OBUF")
        cdlt.configure("end", "systolic_array")
    return cdlt

GENESYS_CODELETS = {
    "conv" : conv2d_nchw
}
