import os
from stealth.simulator.genesys_sim.genesys import run_single_test


_CURRENT_FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def run_layer(layer_directory: str):
    config_path = _CURRENT_FILE_DIRECTORY + f"/simulator/configs/"
    test_path_components = layer_directory.split("/")
    test_path = layer_directory
    test_name = test_path_components[-1]
    mode = "perf"
    test_output = run_single_test(config_path, mode, {"name": test_name, "path": test_path})
    return test_output


def get_simulator_output_field(simulator_output, field):
    return simulator_output[2][simulator_output[1].index(field)]


def get_relevant_simulator_outputs(simulator_output):
    return {
        "total_cycles": int(get_simulator_output_field(simulator_output, "totCycles")),
        "sa_cycles": int(get_simulator_output_field(simulator_output, "systotalCycles")),
        "sa_compute_cycles_per_tile": float(get_simulator_output_field(simulator_output, "sysComputeCyclesPerTile")),
        "sa_load_cycles_per_tile": float(get_simulator_output_field(simulator_output, "sysLoadCyclesPerTile")),
        "sa_store_cycles_per_tile": float(get_simulator_output_field(simulator_output, "sysStoreCyclesPerTile")),
        "sa_ibuf_util_per_tile": float(get_simulator_output_field(simulator_output, "perTileIbufUtil")),
        "sa_obuf_util_per_tile": float(get_simulator_output_field(simulator_output, "perTileObufUtil")),
        "sa_wbuf_util_per_tile": float(get_simulator_output_field(simulator_output, "perTileWbufUtil")),
        "sa_bbuf_util_per_tile": float(get_simulator_output_field(simulator_output, "perTileBbufUtil")),
        "sa_compute_util_per_tile": float(get_simulator_output_field(simulator_output, "perTileComputeUtils")),
        "simd_cycles": int(get_simulator_output_field(simulator_output, "simdtotalCycles")),
        "simd_compute_cycles_per_tile": int(get_simulator_output_field(simulator_output, "simdComputeCyclesPerTile")),
        "simd_load_cycles": float(get_simulator_output_field(simulator_output, "simdLoadCycles")),
        "simd_store_cycles": float(get_simulator_output_field(simulator_output, "simdStoreCycles")),
    }
