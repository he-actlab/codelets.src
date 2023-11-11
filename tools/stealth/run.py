import os
import sys
import subprocess
import csv


_CURRENT_FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def run_layer(layer_directory: str, thread_id: int = 0):
    config_path = _CURRENT_FILE_DIRECTORY + "/simulator/configs"
    test_path = layer_directory
    simulator_dir = _CURRENT_FILE_DIRECTORY + "/simulator"
    output_file = _CURRENT_FILE_DIRECTORY + f"/genesys_output_{thread_id}.csv"
    subprocess.run([sys.executable, _CURRENT_FILE_DIRECTORY + "/simulator/genesys_sim/genesys.py", config_path, test_path, "--log_path", output_file, "--mode", "energy"], env={"PYTHONPATH": simulator_dir})
    with open(output_file, "r") as f:
        reader = csv.reader(f)
        simulator_output = list(reader)
    os.remove(output_file)
    return simulator_output


def get_simulator_output_field(simulator_output, field):
    return simulator_output[2][simulator_output[1].index(field)]


def convert_output_field(output_field_to_float):
    try:
        return int(output_field_to_float)
    except ValueError:
        pass
    try:
        return float(output_field_to_float)
    except ValueError:
        return ""


def get_relevant_simulator_outputs(simulator_output):
    return {
        "total_num_tiles": convert_output_field(get_simulator_output_field(simulator_output, "NumTiles")),
        "total_cycles": convert_output_field(get_simulator_output_field(simulator_output, "totCycles")),
        "mem_wait_cycles": convert_output_field(get_simulator_output_field(simulator_output, "memWaitCycles")),
        "sa_cycles": convert_output_field(get_simulator_output_field(simulator_output, "systotalCycles")),
        "sa_compute_cycles_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "sysComputeCyclesPerTile")),
        "sa_load_cycles_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "sysLoadCyclesPerTile")),
        "sa_store_cycles_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "sysStoreCyclesPerTile")),
        "sa_ibuf_util_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "perTileIbufUtil")),
        "sa_obuf_util_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "perTileObufUtil")),
        "sa_wbuf_util_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "perTileWbufUtil")),
        "sa_bbuf_util_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "perTileBbufUtil")),
        "sa_compute_util_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "perTileComputeUtils")),
        "simd_cycles": convert_output_field(get_simulator_output_field(simulator_output, "simdtotalCycles")),
        "simd_compute_cycles_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "simdComputeCyclesPerTile")),
        "simd_load_vmem1_cycles": convert_output_field(get_simulator_output_field(simulator_output, "simdLoadVmem1Cycles")),
        "simd_load_vmem2_cycles": convert_output_field(get_simulator_output_field(simulator_output, "simdLoadVmem2Cycles")),
        "simd_store_cycles": convert_output_field(get_simulator_output_field(simulator_output, "simdStoreCycles")),
        "simd_num_compute_tiles": convert_output_field(get_simulator_output_field(simulator_output, "NumComputeTiles")),
        "simd_vmem1_load_tiles": convert_output_field(get_simulator_output_field(simulator_output, "VMEM1LoadTiles")),
        "simd_vmem2_load_tiles": convert_output_field(get_simulator_output_field(simulator_output, "VMEM2LoadTiles")),
        "simd_store_tiles": convert_output_field(get_simulator_output_field(simulator_output, "StoreTiles")),
        "simd_util_per_vmem1_tile": convert_output_field(get_simulator_output_field(simulator_output, "perTileVMEM1Util")),
        "simd_util_per_vmem2_tile": convert_output_field(get_simulator_output_field(simulator_output, "perTileVMEM2Util")),
        "sa_num_ibuf_tiles": convert_output_field(get_simulator_output_field(simulator_output, "ibuf_numTile")),
        "sa_ibuf_read_energy": convert_output_field(get_simulator_output_field(simulator_output, "ibuf_readEnergy")),
        "sa_ibuf_leak_power": convert_output_field(get_simulator_output_field(simulator_output, "ibuf_leakPower")),
        "sa_ibuf_area": convert_output_field(get_simulator_output_field(simulator_output, "ibuf_area")),
        "sa_ibuf_num_reads_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "ibuf_perTileNumReads")),
        "sa_ibuf_read_energy_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "ibuf_perTileReadEnergy")),
        "sa_ibuf_total_read_energy": convert_output_field(get_simulator_output_field(simulator_output, "ibuf_totalReadEnergy")),
        "sa_ibuf_total_DDR_read_energy": convert_output_field(get_simulator_output_field(simulator_output, "ibuf_totalDDRReadEnergy")),
        "sa_wbuf_num_tiles": convert_output_field(get_simulator_output_field(simulator_output, "wbuf_numTile")),
        "sa_wbuf_read_energy": convert_output_field(get_simulator_output_field(simulator_output, "wbuf_readEnergy")),
        "sa_wbuf_leak_power": convert_output_field(get_simulator_output_field(simulator_output, "wbuf_leakPower")),
        "sa_wbuf_area": convert_output_field(get_simulator_output_field(simulator_output, "wbuf_area")),
        "sa_wbuf_num_reads_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "wbuf_perTileNumReads")),
        "sa_wbuf_read_energy_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "wbuf_perTileReadEnergy")),
        "sa_wbuf_total_read_energy": convert_output_field(get_simulator_output_field(simulator_output, "wbuf_totalReadEnergy")),
        "sa_wbuf_total_DDR_read_energy": convert_output_field(get_simulator_output_field(simulator_output, "wbuf_totalDDRReadEnergy")),
        "sa_bbuf_num_tiles": convert_output_field(get_simulator_output_field(simulator_output, "bbuf_numTile")),
        "sa_bbuf_read_energy": convert_output_field(get_simulator_output_field(simulator_output, "bbuf_readEnergy")),
        "sa_bbuf_leak_power": convert_output_field(get_simulator_output_field(simulator_output, "bbuf_leakPower")),
        "sa_bbuf_area": convert_output_field(get_simulator_output_field(simulator_output, "bbuf_area")),
        "sa_bbuf_num_reads_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "bbuf_perTileNumReads")),
        "sa_bbuf_read_energy_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "bbuf_perTileReadEnergy")),
        "sa_bbuf_total_read_energy": convert_output_field(get_simulator_output_field(simulator_output, "bbuf_totalReadEnergy")),
        "sa_obuf_num_tiles": convert_output_field(get_simulator_output_field(simulator_output, "obuf_numTile")),
        "sa_obuf_read_energy": convert_output_field(get_simulator_output_field(simulator_output, "obuf_readEnergy")),
        "sa_obuf_write_energy": convert_output_field(get_simulator_output_field(simulator_output, "obuf_writeEnergy")),
        "sa_obuf_leak_power": convert_output_field(get_simulator_output_field(simulator_output, "obuf_leakPower")),
        "sa_obuf_area": convert_output_field(get_simulator_output_field(simulator_output, "obuf_area")),
        "sa_obuf_num_reads_by_sa_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "obuf_perTileSysNumReads")),
        "sa_obuf_num_writes_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "obuf_perTileNumWrite")),
        "sa_obuf_write_energy_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "obuf_perTileWriteEnergy")),
        "sa_obuf_num_reads_by_simd_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "obuf_perTileSIMDNumReads")),
        "sa_obuf_simd_read_energy_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "obuf_perTileSIMDReadEnergy")),
        "sa_obuf_total_energy_per_tile": convert_output_field(get_simulator_output_field(simulator_output, "obuf_perTileTotalEnergy")),
        "sa_obuf_total_read_write_energy": convert_output_field(get_simulator_output_field(simulator_output, "obuf_totalReadWriteEnergy")),
        "sa_obuf_total_DDR_write_energy": convert_output_field(get_simulator_output_field(simulator_output, "obuf_totalDDRWriteEnergy")),
        "simd_vmem1_num_tiles": convert_output_field(get_simulator_output_field(simulator_output, "vmem1_numTile")),
        "simd_vmem1_read_energy": convert_output_field(get_simulator_output_field(simulator_output, "vmem1_readEnergy")),
        "simd_vmem1_write_energy": convert_output_field(get_simulator_output_field(simulator_output, "vmem1_writeEnergy")),
        "simd_vmem1_leak_power": convert_output_field(get_simulator_output_field(simulator_output, "vmem1_leakPower")),
        "simd_vmem1_area": convert_output_field(get_simulator_output_field(simulator_output, "vmem1_area")),
        "simd_vmem1_total_num_reads": convert_output_field(get_simulator_output_field(simulator_output, "vmem1_totalNumReads")),
        "simd_vmem_total_read_energy": convert_output_field(get_simulator_output_field(simulator_output, "vmem1_totalReadEnergy")),
        "simd_vmem1_total_num_writes": convert_output_field(get_simulator_output_field(simulator_output, "vmem1_totalNumWrites")),
        "simd_vmem1_total_write_energy": convert_output_field(get_simulator_output_field(simulator_output, "vmem1_totalWriteEnergy")),
        "simd_vmem1_total_read_data_size_in_bits": convert_output_field(get_simulator_output_field(simulator_output, "vmem1_totalReadDataSizeBits")),
        "simd_vmem1_total_DDR_read_energy": convert_output_field(get_simulator_output_field(simulator_output, "vmem1_totalDDRReadEnergy")),
        "simd_vmem1_total_write_data_size_in_bits": convert_output_field(get_simulator_output_field(simulator_output, "vmem1_totalWriteDataSizeBits")),
        "simd_vmem1_total_DDR_write_energy": convert_output_field(get_simulator_output_field(simulator_output, "vmem1_totalDDRWriteEnergy")),
        "simd_vmem2_num_tiles": convert_output_field(get_simulator_output_field(simulator_output, "vmem2_numTile")),
        "simd_vmem2_read_energy": convert_output_field(get_simulator_output_field(simulator_output, "vmem2_readEnergy")),
        "simd_vmem2_write_energy": convert_output_field(get_simulator_output_field(simulator_output, "vmem2_writeEnergy")),
        "simd_vmem2_leak_power": convert_output_field(get_simulator_output_field(simulator_output, "vmem2_leakPower")),
        "simd_vmem2_area": convert_output_field(get_simulator_output_field(simulator_output, "vmem2_area")),
        "simd_vmem2_total_num_reads": convert_output_field(get_simulator_output_field(simulator_output, "vmem2_totalNumReads")),
        "simd_vmem2_total_read_energy": convert_output_field(get_simulator_output_field(simulator_output, "vmem2_totalReadEnergy")),
        "simd_vmem2_total_num_writes": convert_output_field(get_simulator_output_field(simulator_output, "vmem2_totalNumWrites")),
        "simd_vmem2_total_write_energy": convert_output_field(get_simulator_output_field(simulator_output, "vmem2_totalWriteEnergy")),
        "simd_vmem2_total_read_data_size_in_bits": convert_output_field(get_simulator_output_field(simulator_output, "vmem2_totalReadDataSizeBits")),
        "simd_vmem2_total_DDR_read_energy": convert_output_field(get_simulator_output_field(simulator_output, "vmem2_totalDDRReadEnergy")),
        "simd_vmem2_total_write_data_size_in_bits": convert_output_field(get_simulator_output_field(simulator_output, "vmem2_totalWriteDataSizeBits")),
        "simd_vmem2_total_DDR_write_energy": convert_output_field(get_simulator_output_field(simulator_output, "vmem2_totalDDRWriteEnergy")),
    }
