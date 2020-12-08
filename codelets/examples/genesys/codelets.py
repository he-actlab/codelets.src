from codelets.adl import Codelet, Capability, Operand, ArchitectureNode, ComputeNode
from codelets.adl.util import tile_perms, make_operand_from_node
import polymath as pm
import numpy as np
from collections import namedtuple
from typing import List
from . import SIMD_NS, SIMD_OPCODE_BITWIDTH, OP_DTYPES, \
    OP_LOCATIONS, NS_BITWIDTH, NS_IDX_BITWIDTH
SIMD_INPUT_COMPONENTS = ["OBUF", "VMEM", "IMM", "EXTMEM"]
SIMD_OUTPUT_COMPONENTS = ["IBUF", "VMEM", "EXTMEM", "IMM"]

SYSTOLIC_ARRAY_INPUT_COMPONENTS = ["IBUF", "WBUF", "BBUF"]
SYSTOLIC_ARRAY_OUTPUT_COMPONENTS = ["OBUF"]

def add_node_dtype(node, dtype):
    for i in node.inputs:
        if "hag_dtype" not in i.added_attrs:
            i.add_attribute("hag_dtype", dtype)

    for o in node.outputs:
        assert "hag_dtype" not in o.added_attrs
        o.add_attribute("hag_dtype", dtype)
    return node

def conv2d_systolic_array(hag):
    op_params = {"stride": lambda x: x.args[-2], "pad": lambda x: x.args[-1]}
    cdlt = Codelet("conv", OP_DTYPES, OP_DTYPES)
    sys_array = hag.get_subgraph_node("systolic_array")

    # Group instruction
    grp_instr = sys_array.get_capability("INSTR_ARRAY_GROUP").create_template()
    # Unset fields: group_num, loop_id, num_instr
    grp_instr.set_field_by_name("sa_simd", "SYSTOLIC_ARRAY")
    grp_instr.set_field_by_name("start_end", "START")
    cdlt.add_capability(grp_instr)

    # Set base address for inst fetch. Unset fields
    ifetch_low = sys_array.get_capability("BASEADDR").create_template()
    ifetch_low.set_field_by_name("low_high", "LOW")
    ifetch_low.set_field_by_name("namespace_or_imm", "INST_MEM")
    ifetch_low.set_field_value("namespace", 0)
    cdlt.add_capability(ifetch_low)


    ifetch_high = sys_array.get_capability("BASEADDR").create_template()
    ifetch_high.set_field_by_name("low_high", "HIGH")
    ifetch_high.set_field_by_name("namespace_or_imm", "INST_MEM")
    ifetch_high.set_field_value("namespace", 0)
    cdlt.add_capability(ifetch_high)


    # LD instr for instruction memory: Need instr size and loop id
    ld_instr = sys_array.get_capability("LD").create_template()
    ld_instr.set_field_by_name("namespace_or_imem", "INST_MEM")
    ld_instr.set_field_value("namespace", 0)
    cdlt.add_capability(ld_instr)


    # Base addr for low and high parts of buffers
    wbuf_low = sys_array.get_capability("BASEADDR").create_template()
    wbuf_low.set_field_by_name("low_high", "LOW")
    wbuf_low.set_field_by_name("namespace_or_imm", "NS")
    wbuf_low.set_field_by_name("namespace", "WBUF")
    cdlt.add_capability(wbuf_low)

    wbuf_high = sys_array.get_capability("BASEADDR").create_template()
    wbuf_high.set_field_by_name("low_high", "HIGH")
    wbuf_high.set_field_by_name("namespace_or_imm", "NS")
    wbuf_high.set_field_by_name("namespace", "WBUF")
    cdlt.add_capability(wbuf_high)


    ibuf_low = sys_array.get_capability("BASEADDR").create_template()
    ibuf_low.set_field_by_name("low_high", "LOW")
    ibuf_low.set_field_by_name("namespace_or_imm", "NS")
    ibuf_low.set_field_by_name("namespace", "IBUF")
    cdlt.add_capability(ibuf_low)


    ibuf_high = sys_array.get_capability("BASEADDR").create_template()
    ibuf_high.set_field_by_name("low_high", "HIGH")
    ibuf_high.set_field_by_name("namespace_or_imm", "NS")
    ibuf_high.set_field_by_name("namespace", "IBUF")
    cdlt.add_capability(ibuf_high)

    bbuf_low = sys_array.get_capability("BASEADDR").create_template()
    bbuf_low.set_field_by_name("low_high", "LOW")
    bbuf_low.set_field_by_name("namespace_or_imm", "NS")
    bbuf_low.set_field_by_name("namespace", "BBUF")
    cdlt.add_capability(bbuf_low)


    bbuf_high = sys_array.get_capability("BASEADDR").create_template()
    bbuf_high.set_field_by_name("low_high", "HIGH")
    bbuf_high.set_field_by_name("namespace_or_imm", "NS")
    bbuf_high.set_field_by_name("namespace", "BBUF")
    cdlt.add_capability(bbuf_high)

    obuf_low = sys_array.get_capability("BASEADDR").create_template()
    obuf_low.set_field_by_name("low_high", "LOW")
    obuf_low.set_field_by_name("namespace_or_imm", "NS")
    obuf_low.set_field_by_name("namespace", "OBUF")
    cdlt.add_capability(obuf_low)


    obuf_high = sys_array.get_capability("BASEADDR").create_template()
    obuf_high.set_field_by_name("low_high", "HIGH")
    obuf_high.set_field_by_name("namespace_or_imm", "NS")
    obuf_high.set_field_by_name("namespace", "OBUF")
    cdlt.add_capability(obuf_high)

    ## Now adding outermost loops. Need to add num iterations (num tiles)

    # WBUF LOAD
    wbuf_ld_loop = sys_array.get_capability("SA_LOOP").create_template()
    wbuf_ld_loop.set_field_value("loop_level", 1)
    wbuf_ld_loop.set_field_value("loop_id", 0)
    cdlt.add_capability(wbuf_ld_loop)

    wbuf_ld = sys_array.get_capability("GENADDR").create_template()
    wbuf_ld.set_field_by_name("low_high", "LOW")
    wbuf_ld.set_field_by_name("ld_st", "LD")
    wbuf_ld.set_field_by_name("namespace", "WBUF")
    wbuf_ld.set_field_value("loop_id", 0)

    cdlt.add_capability(wbuf_ld)

    # IBUF LOAD
    ibuf_ld_loop = sys_array.get_capability("SA_LOOP").create_template()
    ibuf_ld_loop.set_field_value("loop_level", 2)
    ibuf_ld_loop.set_field_value("loop_id", 1)
    cdlt.add_capability(ibuf_ld_loop)

    ibuf_ld = sys_array.get_capability("GENADDR").create_template()
    ibuf_ld.set_field_by_name("low_high", "LOW")
    ibuf_ld.set_field_by_name("ld_st", "LD")
    ibuf_ld.set_field_by_name("namespace", "IBUF")
    ibuf_ld.set_field_value("loop_id", 1)
    cdlt.add_capability(ibuf_ld)

    # OBUF ST
    obuf_ld_loop = sys_array.get_capability("SA_LOOP").create_template()
    obuf_ld_loop.set_field_value("loop_level", 3)
    obuf_ld_loop.set_field_value("loop_id", 2)
    cdlt.add_capability(obuf_ld_loop)

    obuf_ld = sys_array.get_capability("GENADDR").create_template()
    obuf_ld.set_field_by_name("low_high", "LOW")
    obuf_ld.set_field_by_name("ld_st", "ST")
    obuf_ld.set_field_by_name("namespace", "OBUF")
    obuf_ld.set_field_value("loop_id", 2)

    cdlt.add_capability(obuf_ld)

    # BBUF LOAD. TODO: Switch the loop level to be higher
    bbuf_ld_loop = sys_array.get_capability("SA_LOOP").create_template()
    bbuf_ld_loop.set_field_value("loop_level", 1)
    bbuf_ld_loop.set_field_value("loop_id", 3)
    cdlt.add_capability(bbuf_ld_loop)

    bbuf_ld = sys_array.get_capability("GENADDR").create_template()
    bbuf_ld.set_field_by_name("low_high", "LOW")
    bbuf_ld.set_field_by_name("ld_st", "LD")
    bbuf_ld.set_field_by_name("namespace", "BBUF")
    bbuf_ld.set_field_value("loop_id", 3)
    cdlt.add_capability(bbuf_ld)

    # Inner loops over tile offchip

    # WBUF READ
    wbuf_inner_ld_id = 4
    wbuf_inner_ld_loop = sys_array.get_capability("SA_LOOP").create_template()
    wbuf_inner_ld_loop.set_field_value("loop_level", 1)
    wbuf_inner_ld_loop.set_field_value("loop_id", wbuf_inner_ld_id)
    cdlt.add_capability(wbuf_inner_ld_loop)

    wbuf_inner_ld_gen = sys_array.get_capability("GENADDR").create_template()
    wbuf_inner_ld_gen.set_field_by_name("low_high", "LOW")
    wbuf_inner_ld_gen.set_field_by_name("ld_st", "LD")
    wbuf_inner_ld_gen.set_field_by_name("namespace", "WBUF")
    wbuf_inner_ld_gen.set_field_value("loop_id", wbuf_inner_ld_id)
    cdlt.add_capability(wbuf_inner_ld_gen)


    wbuf_inner_ld = sys_array.get_capability("LD").create_template()
    wbuf_inner_ld.set_field_by_name("namespace_or_imem", "NS")
    wbuf_inner_ld.set_field_value("loop_id", wbuf_inner_ld_id)
    cdlt.add_capability(wbuf_inner_ld)

    # IBUF READ
    ibuf_inner_ld_id = 5
    ibuf_inner_ld_loop = sys_array.get_capability("SA_LOOP").create_template()
    ibuf_inner_ld_loop.set_field_value("loop_level", 1)
    ibuf_inner_ld_loop.set_field_value("loop_id", ibuf_inner_ld_id)
    cdlt.add_capability(ibuf_inner_ld_loop)

    ibuf_inner_ld_gen = sys_array.get_capability("GENADDR").create_template()
    ibuf_inner_ld_gen.set_field_by_name("low_high", "LOW")
    ibuf_inner_ld_gen.set_field_by_name("ld_st", "LD")
    ibuf_inner_ld_gen.set_field_by_name("namespace", "IBUF")
    ibuf_inner_ld_gen.set_field_value("loop_id", ibuf_inner_ld_id)
    cdlt.add_capability(ibuf_inner_ld_gen)


    ibuf_inner_ld = sys_array.get_capability("LD").create_template()
    ibuf_inner_ld.set_field_by_name("namespace_or_imem", "NS")
    ibuf_inner_ld.set_field_value("loop_id", ibuf_inner_ld_id)
    cdlt.add_capability(ibuf_inner_ld)

    # OBUF READ
    obuf_inner_st_id = 6
    obuf_inner_st_loop = sys_array.get_capability("SA_LOOP").create_template()
    obuf_inner_st_loop.set_field_value("loop_level", 1)
    obuf_inner_st_loop.set_field_value("loop_id", obuf_inner_st_id)
    cdlt.add_capability(obuf_inner_st_loop)

    obuf_inner_st_gen = sys_array.get_capability("GENADDR").create_template()
    obuf_inner_st_gen.set_field_by_name("low_high", "LOW")
    obuf_inner_st_gen.set_field_by_name("ld_st", "ST")
    obuf_inner_st_gen.set_field_by_name("namespace", "OBUF")
    obuf_inner_st_gen.set_field_value("loop_id", obuf_inner_st_id)
    cdlt.add_capability(obuf_inner_st_gen)


    obuf_inner_st = sys_array.get_capability("ST").create_template()
    obuf_inner_st.set_field_by_name("namespace_or_imem", "NS")
    obuf_inner_st.set_field_value("loop_id", obuf_inner_st_id)
    cdlt.add_capability(obuf_inner_st)

    # BBUF READ
    bbuf_inner_ld_id = 7
    bbuf_inner_ld_loop = sys_array.get_capability("SA_LOOP").create_template()
    bbuf_inner_ld_loop.set_field_value("loop_level", 1)
    bbuf_inner_ld_loop.set_field_value("loop_id", bbuf_inner_ld_id)
    cdlt.add_capability(bbuf_inner_ld_loop)

    bbuf_inner_ld_gen = sys_array.get_capability("GENADDR").create_template()
    bbuf_inner_ld_gen.set_field_by_name("low_high", "LOW")
    bbuf_inner_ld_gen.set_field_by_name("ld_st", "LD")
    bbuf_inner_ld_gen.set_field_by_name("namespace", "BBUF")
    bbuf_inner_ld_gen.set_field_value("loop_id", bbuf_inner_ld_id)
    cdlt.add_capability(bbuf_inner_ld_gen)


    bbuf_inner_ld = sys_array.get_capability("LD").create_template()
    bbuf_inner_ld.set_field_by_name("namespace_or_imem", "NS")
    bbuf_inner_ld.set_field_value("loop_id", bbuf_inner_ld_id)
    cdlt.add_capability(bbuf_inner_ld)


    # Dimension loop tiling

    # # WBUF oc
    wbuf_oc = wbuf_inner_ld_id + 4
    wbuf_rd_loop_oc = sys_array.get_capability("SA_LOOP").create_template()
    wbuf_rd_loop_oc.set_field_value("loop_level", 1)
    wbuf_rd_loop_oc.set_field_value("loop_id", wbuf_oc)
    cdlt.add_capability(wbuf_rd_loop_oc)

    wbuf_rd_genaddr_oc = sys_array.get_capability("GENADDR").create_template()
    wbuf_rd_genaddr_oc.set_field_by_name("low_high", "LOW")
    wbuf_rd_genaddr_oc.set_field_by_name("ld_st", "RD")
    wbuf_rd_genaddr_oc.set_field_by_name("namespace", "WBUF")
    wbuf_rd_genaddr_oc.set_field_value("loop_id", wbuf_oc)
    cdlt.add_capability(wbuf_rd_genaddr_oc)
    #
    # # WBUF IC
    wbuf_ic = wbuf_inner_ld_id + 8
    wbuf_rd_loop_ic = sys_array.get_capability("SA_LOOP").create_template()
    wbuf_rd_loop_ic.set_field_value("loop_level", 2)
    wbuf_rd_loop_ic.set_field_value("loop_id", wbuf_ic)
    cdlt.add_capability(wbuf_rd_loop_ic)

    wbuf_rd_genaddr_ic = sys_array.get_capability("GENADDR").create_template()
    wbuf_rd_genaddr_ic.set_field_by_name("low_high", "LOW")
    wbuf_rd_genaddr_ic.set_field_by_name("ld_st", "RD")
    wbuf_rd_genaddr_ic.set_field_by_name("namespace", "WBUF")
    wbuf_rd_genaddr_ic.set_field_value("loop_id", wbuf_ic)
    cdlt.add_capability(wbuf_rd_genaddr_ic)
    #
    # # WBUF KW
    #
    wbuf_kw = wbuf_inner_ld_id + 12
    wbuf_rd_loop_kw = sys_array.get_capability("SA_LOOP").create_template()
    wbuf_rd_loop_kw.set_field_value("loop_level", 3)
    wbuf_rd_loop_kw.set_field_value("loop_id", wbuf_kw)
    cdlt.add_capability(wbuf_rd_loop_kw)

    wbuf_rd_genaddr_kw = sys_array.get_capability("GENADDR").create_template()
    wbuf_rd_genaddr_kw.set_field_by_name("low_high", "LOW")
    wbuf_rd_genaddr_kw.set_field_by_name("ld_st", "RD")
    wbuf_rd_genaddr_kw.set_field_by_name("namespace", "WBUF")
    wbuf_rd_genaddr_kw.set_field_value("loop_id", wbuf_kw)
    cdlt.add_capability(wbuf_rd_genaddr_kw)
    #
    # # WBUF KH
    #
    wbuf_kh = wbuf_inner_ld_id + 16
    wbuf_rd_loop_kh = sys_array.get_capability("SA_LOOP").create_template()
    wbuf_rd_loop_kh.set_field_value("loop_level", 4)
    wbuf_rd_loop_kh.set_field_value("loop_id", wbuf_kh)
    cdlt.add_capability(wbuf_rd_loop_kh)

    wbuf_rd_genaddr_kh = sys_array.get_capability("GENADDR").create_template()
    wbuf_rd_genaddr_kh.set_field_by_name("low_high", "LOW")
    wbuf_rd_genaddr_kh.set_field_by_name("ld_st", "RD")
    wbuf_rd_genaddr_kh.set_field_by_name("namespace", "WBUF")
    wbuf_rd_genaddr_kh.set_field_value("loop_id", wbuf_kh)
    cdlt.add_capability(wbuf_rd_genaddr_kh)

    # ############################
    #
    # # IBUF b
    ibuf_b = ibuf_inner_ld_id + 4
    ibuf_rd_loop_b = sys_array.get_capability("SA_LOOP").create_template()
    ibuf_rd_loop_b.set_field_value("loop_level", 1)
    ibuf_rd_loop_b.set_field_value("loop_id", ibuf_b)
    cdlt.add_capability(ibuf_rd_loop_b)

    ibuf_rd_genaddr_b = sys_array.get_capability("GENADDR").create_template()
    ibuf_rd_genaddr_b.set_field_by_name("low_high", "LOW")
    ibuf_rd_genaddr_b.set_field_by_name("ld_st", "RD")
    ibuf_rd_genaddr_b.set_field_by_name("namespace", "IBUF")
    ibuf_rd_genaddr_b.set_field_value("loop_id", ibuf_b)
    cdlt.add_capability(ibuf_rd_genaddr_b)
    #
    # # IBUF IC
    ibuf_ic = ibuf_inner_ld_id + 8
    ibuf_rd_loop_ic = sys_array.get_capability("SA_LOOP").create_template()
    ibuf_rd_loop_ic.set_field_value("loop_level", 2)
    ibuf_rd_loop_ic.set_field_value("loop_id", ibuf_ic)
    cdlt.add_capability(ibuf_rd_loop_ic)

    ibuf_rd_genaddr_ic = sys_array.get_capability("GENADDR").create_template()
    ibuf_rd_genaddr_ic.set_field_by_name("low_high", "LOW")
    ibuf_rd_genaddr_ic.set_field_by_name("ld_st", "RD")
    ibuf_rd_genaddr_ic.set_field_by_name("namespace", "IBUF")
    ibuf_rd_genaddr_ic.set_field_value("loop_id", ibuf_ic)
    cdlt.add_capability(ibuf_rd_genaddr_ic)
    #
    # # IBUF IW
    #
    ibuf_iw = ibuf_inner_ld_id + 12
    ibuf_rd_loop_iw = sys_array.get_capability("SA_LOOP").create_template()
    ibuf_rd_loop_iw.set_field_value("loop_level", 3)
    ibuf_rd_loop_iw.set_field_value("loop_id", ibuf_iw)
    cdlt.add_capability(ibuf_rd_loop_iw)

    ibuf_rd_genaddr_iw = sys_array.get_capability("GENADDR").create_template()
    ibuf_rd_genaddr_iw.set_field_by_name("low_high", "LOW")
    ibuf_rd_genaddr_iw.set_field_by_name("ld_st", "RD")
    ibuf_rd_genaddr_iw.set_field_by_name("namespace", "IBUF")
    ibuf_rd_genaddr_iw.set_field_value("loop_id", ibuf_iw)
    cdlt.add_capability(ibuf_rd_genaddr_iw)
    #
    # # IBUF IH
    #
    ibuf_ih = ibuf_inner_ld_id + 16
    ibuf_rd_loop_ih = sys_array.get_capability("SA_LOOP").create_template()
    ibuf_rd_loop_ih.set_field_value("loop_level", 4)
    ibuf_rd_loop_ih.set_field_value("loop_id", ibuf_ih)
    cdlt.add_capability(ibuf_rd_loop_ih)

    ibuf_rd_genaddr_ih = sys_array.get_capability("GENADDR").create_template()
    ibuf_rd_genaddr_ih.set_field_by_name("low_high", "LOW")
    ibuf_rd_genaddr_ih.set_field_by_name("ld_st", "RD")
    ibuf_rd_genaddr_ih.set_field_by_name("namespace", "IBUF")
    ibuf_rd_genaddr_ih.set_field_value("loop_id", ibuf_ih)
    cdlt.add_capability(ibuf_rd_genaddr_ih)

    ############################

    # OBUF b
    obuf_b = obuf_inner_st_id + 4
    obuf_wr_loop_b = sys_array.get_capability("SA_LOOP").create_template()
    obuf_wr_loop_b.set_field_value("loop_level", 1)
    obuf_wr_loop_b.set_field_value("loop_id", obuf_b)
    cdlt.add_capability(obuf_wr_loop_b)

    obuf_wr_genaddr_b = sys_array.get_capability("GENADDR").create_template()
    obuf_wr_genaddr_b.set_field_by_name("low_high", "LOW")
    obuf_wr_genaddr_b.set_field_by_name("ld_st", "WR")
    obuf_wr_genaddr_b.set_field_by_name("namespace", "OBUF")
    obuf_wr_genaddr_b.set_field_value("loop_id", obuf_b)
    cdlt.add_capability(obuf_wr_genaddr_b)

    # OBUF OC
    obuf_oc = obuf_inner_st_id + 8
    obuf_wr_loop_oc = sys_array.get_capability("SA_LOOP").create_template()
    obuf_wr_loop_oc.set_field_value("loop_level", 2)
    obuf_wr_loop_oc.set_field_value("loop_id", obuf_oc)
    cdlt.add_capability(obuf_wr_loop_oc)

    obuf_wr_genaddr_oc = sys_array.get_capability("GENADDR").create_template()
    obuf_wr_genaddr_oc.set_field_by_name("low_high", "LOW")
    obuf_wr_genaddr_oc.set_field_by_name("ld_st", "WR")
    obuf_wr_genaddr_oc.set_field_by_name("namespace", "OBUF")
    obuf_wr_genaddr_oc.set_field_value("loop_id", obuf_oc)
    cdlt.add_capability(obuf_wr_genaddr_oc)

    # OBUF OW

    obuf_ow = obuf_inner_st_id + 12
    obuf_wr_loop_ow = sys_array.get_capability("SA_LOOP").create_template()
    obuf_wr_loop_ow.set_field_value("loop_level", 3)
    obuf_wr_loop_ow.set_field_value("loop_id", obuf_ow)
    cdlt.add_capability(obuf_wr_loop_ow)

    obuf_wr_genaddr_ow = sys_array.get_capability("GENADDR").create_template()
    obuf_wr_genaddr_ow.set_field_by_name("low_high", "LOW")
    obuf_wr_genaddr_ow.set_field_by_name("ld_st", "WR")
    obuf_wr_genaddr_ow.set_field_by_name("namespace", "OBUF")
    obuf_wr_genaddr_ow.set_field_value("loop_id", obuf_ow)
    cdlt.add_capability(obuf_wr_genaddr_ow)

    # OBUF OH

    obuf_oh = obuf_inner_st_id + 16
    obuf_wr_loop_oh = sys_array.get_capability("SA_LOOP").create_template()
    obuf_wr_loop_oh.set_field_value("loop_level", 4)
    obuf_wr_loop_oh.set_field_value("loop_id", obuf_oh)
    cdlt.add_capability(obuf_wr_loop_oh)

    obuf_wr_genaddr_oh = sys_array.get_capability("GENADDR").create_template()
    obuf_wr_genaddr_oh.set_field_by_name("low_high", "LOW")
    obuf_wr_genaddr_oh.set_field_by_name("ld_st", "WR")
    obuf_wr_genaddr_oh.set_field_by_name("namespace", "OBUF")
    obuf_wr_genaddr_oh.set_field_value("loop_id", obuf_oh)
    cdlt.add_capability(obuf_wr_genaddr_oh)

    # BBUF OH

    bbuf_oc = bbuf_inner_ld_id + 4
    bbuf_rd_loop_oc = sys_array.get_capability("SA_LOOP").create_template()
    bbuf_rd_loop_oc.set_field_value("loop_level", 1)
    bbuf_rd_loop_oc.set_field_value("loop_id", bbuf_oc)
    cdlt.add_capability(bbuf_rd_loop_oc)

    bbuf_rd_genaddr_oc = sys_array.get_capability("GENADDR").create_template()
    bbuf_rd_genaddr_oc.set_field_by_name("low_high", "LOW")
    bbuf_rd_genaddr_oc.set_field_by_name("ld_st", "RD")
    bbuf_rd_genaddr_oc.set_field_by_name("namespace", "BBUF")
    bbuf_rd_genaddr_oc.set_field_value("loop_id", bbuf_oc)
    cdlt.add_capability(bbuf_rd_genaddr_oc)


    # END
    end_sys_array = sys_array.get_capability("INSTR_ARRAY_GROUP").create_template()
    end_sys_array.set_field_by_name("sa_simd", "SYSTOLIC_ARRAY")
    end_sys_array.set_field_by_name("start_end", "END")
    cdlt.add_capability(end_sys_array)

    # END block
    end_block = sys_array.get_capability("BLOCK_END").create_template()
    end_block.set_field_value("last", 0)
    cdlt.add_capability(end_block)

    return cdlt





def conv2d_bias_systolic_array(node, hag):
    cap = Codelet("conv2d_bias")
    cap.input_dimension_names = [["n", "ic", "h", "w"], ["oc", "ic", "kh", "kw"], ["oc"]]
    cap.output_dimension_names = ["n", "oc", "oh", "ow"]
    cap.input_dtypes = ["fxp32", "fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SYSTOLIC_ARRAY_INPUT_COMPONENTS
    cap.output_components = SYSTOLIC_ARRAY_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    op_params = {"stride": node.args[-2], "pad": node.args[-1]}
    loop_order = ["oc", "kh", "kw", "ic", "n", "h", "w"]
    op = Capability(cap, "systolic_array", inputs, outputs, loop_order, 1, **op_params)

    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0

    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    op.tiling = tiling_dict
    return op


def gemm_systolic_array(node, hag):
    # TODO: Need to handle transpose
    cap = Codelet("gemm")
    if "transB" in node.kwargs and node.kwargs["transB"]:
        cap.input_dimension_names = [["M", "N"], ["P", "N"], ["P"]]
        cap.output_dimension_names = ["M", "P"]
    else:
        cap.input_dimension_names = [["M", "N"], ["N", "P"], ["N"]]
        cap.output_dimension_names = ["M", "P"]
    cap.input_dtypes = ["fxp32", "fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SYSTOLIC_ARRAY_INPUT_COMPONENTS
    cap.output_components = SYSTOLIC_ARRAY_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = ["M", "N", "P"]
    op_params = {}
    op = Capability(cap, "systolic_array", inputs, outputs, loop_order, 1, **op_params)
    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    op.tiling = tiling_dict
    return op


def tanh_simd(node, hag: ComputeNode):
    cap = Codelet("tanh")

    cap.input_dtypes = ["fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    # TODO: Fix exec num
    executions = 1
    loop_order = [f"d{i}" for i in range(len(outputs[0].dimensions))]
    cap.input_dimension_names = [loop_order]
    cap.output_dimension_names = loop_order
    op_params = {}
    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)
    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]
    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    op.tiling = tiling_dict
    target_component = hag.get_subgraph_node("SIMD")
    return op

def average_pool_simd(node, hag):
    cap = Codelet("avg_pool")
    cap.input_dimension_names = [["n", "c", "ih", "iw"]]
    cap.output_dimension_names = ["n", "c", "oh", "ow"]
    cap.input_dtypes = ["fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = ["n", "c", "ih", "iw"]
    op_params = {"stride": node.args[4], "kh": node.args[2], "kw": node.args[3],
                 "pad": node.args[5]}
    executions = 1
    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)
    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    op.tiling = tiling_dict

    return op

def softmax_simd(node, hag):
    cap = Codelet("softmax")
    cap.input_dimension_names = [["n", "c"]]
    cap.output_dimension_names = ["n", "c"]
    cap.input_dtypes = ["fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = ["n", "c"]
    op_params = {}
    executions = 1
    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)

    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    op.tiling = tiling_dict
    return op

def add_simd(node, hag):
    cap = Codelet("add")

    cap.input_dtypes = ["fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = [f"d{i}" for i in range(len(outputs[0].dimensions))]
    cap.input_dimension_names = [loop_order, loop_order]
    cap.output_dimension_names = loop_order

    # TODO: Fix executions
    executions = 1
    op_params = {}
    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)

    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1

    return op



def batch_normalization_simd(node, hag):
    cap = Codelet("batchnorm")
    cap.input_dimension_names = [["n", "ic", "h", "w"], ["ic"], ["ic"], ["ic"], ["ic"]]
    cap.output_dimension_names = ["n", "ic", "h", "w"]
    cap.input_dtypes = ["fxp32", "fxp32", "fxp32", "fxp32", "fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = cap.output_dimension_names
    # TODO: Fix executions
    executions = 1
    op_params = {}

    if "eps" in node.kwargs:
        op_params["eps"] = node.kwargs["eps"]
    else:
        op_params["eps"] = 1e-05

    if "momentum" in node.kwargs:
        op_params["momentum"] = node.kwargs["momentum"]
    else:
        op_params["momentum"] = 0.9

    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)

    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1

    return op

def relu_simd(node, hag: ArchitectureNode):
    cdlt = Codelet("relu")
    target = "SIMD"
    capability_sequence = []
    simd_component = hag.get_subgraph_node("SIMD")
    ld = simd_component.get_capability('LD')
    return cdlt

def fully_connected_systolic_array(node, hag):
    cap = Codelet("fc")
    cap.input_dimension_names = [["M", "N"], ["N", "P"]]
    cap.output_dimension_names = ["M", "P"]
    cap.input_dtypes = ["fxp32", "fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SYSTOLIC_ARRAY_INPUT_COMPONENTS
    cap.output_components = SYSTOLIC_ARRAY_OUTPUT_COMPONENTS
    return cap

def get_binary_simd(node, hag):
    cap = Codelet(f"{node.op_name}")
    cap.input_dtypes = ["fxp32", "fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS
    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = [f"d{i}" for i in range(len(outputs[0].dimensions))]
    cap.input_dimension_names = [loop_order, loop_order]
    cap.output_dimension_names = loop_order
    # TODO: Fix executions
    executions = 1
    op_params = {}
    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)

    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1

    return op

def maxpool_simd(node, hag):
    cap = Codelet("maxpool")
    cap.input_dimension_names = [["n", "ic", "h", "w"]]
    cap.output_dimension_names = ["n", "ic", "oh", "ow"]
    cap.input_dtypes = ["fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS

    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = cap.input_dimension_names[0]
    # TODO: Fix executions
    executions = 1
    op_params = {"stride": node.args[4], "kernel_size": (node.args[2], node.args[3]), "pad": node.args[4]}

    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)

    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    return op

def flatten(node, hag):
    cap = Codelet("flatten")
    cap.input_dtypes = ["fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS
    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = [f"d{i}" for i in range(len(inputs[0].dimensions))]

    cap.input_dimension_names = [loop_order]
    cap.output_dimension_names = ["M", "N"]
    op_params = {"axis": 1}

    if "axis" in node.kwargs:
        op_params["axis"] = node.kwargs["axis"]

    executions = 1
    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)

    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    return op


def global_avg_pool_simd(node, hag):
    cap = Codelet("global_avg_pool")
    cap.input_dimension_names = [["n", "ic", "h", "w"]]
    cap.output_dimension_names = ["n", "ic", "oh", "ow"]
    cap.input_dtypes = ["fxp32"]
    cap.output_dtypes = ["fxp32"]
    cap.input_components = SIMD_INPUT_COMPONENTS
    cap.output_components = SIMD_OUTPUT_COMPONENTS
    inputs = [make_operand_from_node(i, hag) for i in node.inputs]
    outputs = [make_operand_from_node(o, hag, location="OBUF") for o in node.outputs]
    loop_order = cap.input_dimension_names[0]
    # TODO: Fix executions
    executions = 1
    op_params = {}
    op = Capability(cap, "SIMD", inputs, outputs, loop_order, executions, **op_params)

    tilings = tile_perms(list(op.input_dims), 2)
    selected_tiling = tilings[len(tilings) // 2]

    tiling_dict = {}
    i = 0
    for k in op.dimension_values.keys():
        if k in loop_order:
            tiling_dict[k] = selected_tiling[i]
            i += 1
    return op

def macc_simd(node, hag):
    pass

def max_simd(node, hag):
    pass

def min_simd(node, hag):
    pass

def rshift_simd(node, hag):
    pass

def move_simd(node, hag):
    pass

def cond_mv_true_simd(node, hag):
    pass

def cond_mv_false_simd(node, hag):
    pass

def not_simd(node, hag):
    pass

def and_simd(node, hag):
    pass

def or_simd(node, hag):
    pass

def leaky_relu_simd(node, hag):
    pass

def sigmoid_simd(node, hag):
    pass

def exp_simd(node, hag):
    pass

def ln_simd(node, hag):
    pass

def sqrt_simd(node, hag):
    pass

def inv_sqrt_simd(node, hag):
    pass

def log2_simd(node, hag):
    pass

def eq_simd(node, hag):
    pass

def neq_simd(node, hag):
    pass

def gt_simd(node, hag):
    pass

def gte_simd(node, hag):
    pass

def lt_simd(node, hag):
    pass

def lte_simd(node, hag):
    pass


GENESYS_SA_CAPS = {
    "conv": conv2d_systolic_array,
    # "conv_bias": conv2d_bias_systolic_array,
    # "tanh": tanh_simd,
    # "elem_tanh": tanh_simd,
    # "gemm": gemm_systolic_array,
    # "avg_pool": average_pool_simd,
    # "softmax": softmax_simd,
    # "elem_add": get_binary_simd,
    # "elem_mul": get_binary_simd,
    # "elem_sub": get_binary_simd,
    # "elem_div": get_binary_simd,
    # "batch_norm": batch_normalization_simd,
    # "max_pool": maxpool_simd,
    # "global_avg_pool": global_avg_pool_simd,
    # "relu": relu_simd,
    # "coarse_flatten": flatten
}