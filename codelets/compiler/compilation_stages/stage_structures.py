from pytools import memoize_method
from typing import List, Dict, Tuple
from codelets.adl.flex_param import FlexParam
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TilingInfo_:
    name: str
    constraint_str: str
    src_node: str
    dst_node: str
    loop_dim_map: Dict[str, str]
    constraint_fp: FlexParam = field(default=None)
    tile_hint: FlexParam = field(default=None)


    def __post_init__(self):
        if self.constraint_fp is None:
            self.constraint_fp = FlexParam(self.name, ["size"], self.constraint_str)

    def evaluate_constraint(self, sizes, dtype_bits):
        total_size = np.prod(list(sizes.values())) * dtype_bits
        constraint_sat = self.constraint_fp.evaluate_fn(total_size)
        return constraint_sat


@dataclass
class TilingInfo:
    name: str
    loop_dim_map: Dict[str, str]
    levels: int
    level_map: Dict[Tuple[str, str], int] = field(default_factory=dict)
    constraint_fps: Dict[str, FlexParam] = field(default_factory=dict)
    tile_hints: Dict[int, Dict[str, FlexParam]] = field(default_factory=dict)

    def __post_init__(self):
        for i in range(self.levels):
            self.tile_hints[i] = {}

    def add_constraint(self, src: str, dst: str, level: int, constraint_str: str):
        self.constraint_fps[src,dst] = FlexParam(f"{self.name}_{src}_{dst}", ["size"], constraint_str)
        self.level_map[(src, dst)] = level

    def evaluate_constraint(self, key: Tuple[str, str], sizes: Dict[str, int], dtype_bits: int):
        total_size = np.prod(list(sizes.values())) * dtype_bits
        constraint_sat = self.constraint_fps[key].evaluate_fn(total_size)
        return constraint_sat

    def add_tile_hint(self, level: int, loop_name: str, hint_str):
        hint = FlexParam(f"{loop_name}_lvl{level}_hint", ["size", "split"], hint_str)
        self.tile_hints[level][loop_name] = hint

    def add_level_hint(self, level: int, hint_str):
        name = f"LEVEL{level}_hint"
        hint = FlexParam(name, ["sizes", "splits"], hint_str)
        assert name not in self.tile_hints
        self.tile_hints[name] = hint

    def check_tile_hints(self, level, loop_deps, sizes, splits):

        for l, th in self.tile_hints[level].items():
            idx = loop_deps.index(l)
            size = sizes[idx]
            split = splits[idx]
            valid = th.evaluate_fn(size, split)
            if not valid:
                return False
        level_name = f"LEVEL{level}_hint"
        if level_name in self.tile_hints:
            sizes = {self.loop_dim_map[l]: sizes[i] for i, l in enumerate(loop_deps)}
            splits = {self.loop_dim_map[l]: splits[i] for i, l in enumerate(loop_deps)}
            valid = self.tile_hints[level_name].evaluate_fn(sizes, splits)
            if not valid:
                return False

        return True


