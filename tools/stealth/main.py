import argparse
from codelet_string import *
from compile import compile
from stealth_codelet import *
from parse import parse_stealth_codelet


def main() -> None:
    codelet_string = generate_simd_element_wise_relu_codelet(
        ("input_activation", ("N", "H", "W", "C")),
        ("output_activation", ("N", "H", "W", "C")),
        16,
        dimension_tiling=(1, 1, 1, 2),
    )
    parsed_codelet = parse_stealth_codelet(codelet_string)
    stealth_codelet = build_codelet_from_parse_tree(parsed_codelet, codelet_string)
    compile("../../codelets/examples/genesys/configs/benchmark_16x16.json", stealth_codelet, {"N": 1, "H": 20, "W": 20, "C": 32})

if __name__ == "__main__":
    main()
