from typing import Dict
import numpy as np
import inspect

## These assume FPGA16x16 config
TILE_FUNCS = {
    "conv_bias_relu": [
        lambda sizes, splits: sizes['KH']*sizes['KW']*sizes['IC']*sizes['OC']*splits['N']*splits['OH']*splits['OW'] > 1,
        lambda sizes, splits: splits['IC'] == 1,
        lambda sizes, splits: sizes['IC']*8 % 512 == 0,
        lambda sizes, splits: sizes['IC']*32 % 512 == 0,
        lambda sizes, splits: sizes['OC']*32 % 512 == 0,
        lambda sizes, splits: sizes['OC']*32 % 8 == 0,
        lambda sizes, splits: sizes['KH']*sizes['KW']*sizes['IC']*sizes['OC'] <= 524288,
        lambda sizes, splits: sizes['N']*sizes['OH']*sizes['OW']*sizes['OC'] <= 32768,
        lambda sizes, splits: splits['KH'] == 1,
        lambda sizes, splits: splits['KW'] == 1,
        lambda sizes, splits: sizes['OC'] % 16 == 0,
        lambda sizes, splits: sizes['IH']*sizes['IW']*sizes['N']*sizes['IC']*8 <= 131072.0,
        lambda sizes, splits: sizes['KW']*sizes['KH']*sizes['IC']*sizes['OC']*8 <= 2097152.0,
        lambda sizes, splits: sizes['N']*sizes['OW']*sizes['OH']*sizes['OC']*32 <= 524288.0,
        lambda sizes, splits: sizes['N']*sizes['OW']*sizes['OH']*sizes['OC']*32 <= 523776.0,
        lambda sizes, splits: sizes['OC']*32 <= 262144.0,
                        ],
    "gemm_bias": [
        lambda sizes, splits: sizes['N']*sizes['P'] <= 134217728,
        lambda sizes, splits: np.prod(list(splits.values())) > 1,
        lambda sizes, splits: sizes['M']*sizes['P'] <= 524288,
        lambda sizes, splits: sizes['N']*8 % 512 == 0,
        lambda sizes, splits: sizes['P'] % 128 == 0,
        lambda sizes, splits: sizes['N'] % 128 == 0,
    ]
}

def check_tile(layer_name: str, sizes: Dict, tile_sizes: Dict):
    assert layer_name in TILE_FUNCS
    assert set(sizes.keys()) == set(tile_sizes.keys())
    splits = {}
    for k in tile_sizes.keys():
        if sizes[k] % tile_sizes[k] != 0 and k not in ["IH", "IW"]:
            raise RuntimeError(f"Invalid tiling for {k}:\n"
                               f"Size: {sizes[k]}\n"
                               f"Tile size: {tile_sizes[k]}")
        splits[k] = sizes[k] // tile_sizes[k]
    for i, constr in enumerate(TILE_FUNCS[layer_name]):
        if not constr(tile_sizes, splits):
            string_fn = str(inspect.getsourcelines(TILE_FUNCS[layer_name][i])[0])
            print(f"Unsatisfied constraint: {string_fn}")


if __name__ == "__main__":
    # tsize = { "OC": 64, "N": 1, "IC": 512, "KH": 3, "KW": 3, "OH": 7, "OW": 7, "IH": 16, "IW": 16}
    # size = {
    #     "OC": 512,
    #     "N": 1,
    #     "IC": 512,
    #     "KH": 3,
    #     "KW": 3,
    #     "OH": 7,
    #     "OW": 7,
    #     "IH": 3,
    #     "IW": 3,
    #   }
    size = {'M': 1, 'N': 832, 'P': 128}
    tsize = {'M': 1, 'N': 128, 'P': 128}
    check_tile("gemm_bias", size, tsize)