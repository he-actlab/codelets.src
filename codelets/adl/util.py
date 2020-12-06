from types import FunctionType
import inspect
import copy
from codelets.adl import Operand
from operator import mul
import itertools
import polymath as pm
from functools import reduce


def lambda_to_str(fn):
    fn_str = str(inspect.getsourcelines(fn)[0])
    fn_str = fn_str.strip("['\\n']").split(" = ")[1]
    return fn_str


def is_lambda(param) -> bool:
    if isinstance(param, FunctionType) and param.__code__.co_name == "<lambda>":
        return True
    return False

def loop_tile_with_hint(tile_permutations, loop_extent, num_level, loop_hint):
    # TODO support more than 1 level of para hint
    for level in range(num_level):
        if loop_hint[level] != None:
            loop_hint_level = level
            break

    blocking_hint = 1 if loop_hint[loop_hint_level][1] == None else loop_hint[loop_hint_level][1]
    assert loop_hint[loop_hint_level][2]
    para_hint = loop_hint[loop_hint_level][2]
    # para_hint = 1 if loop_hint[loop_hint_level][2] == None else loop_hint[loop_hint_level][2]
    blocking_factor = blocking_hint * para_hint

    pre_tile_permutations = []
    if loop_hint_level == 0:
        pre_tile_permutations.append([])
    else:
        for sub_extent in factors((loop_extent + blocking_factor - 1) // blocking_factor):
            recursive_tile(pre_tile_permutations, [], sub_extent, 0, loop_hint_level)

    for pre_tile in pre_tile_permutations:
        # TODO support not fixed blocking hint
        if loop_hint[loop_hint_level][1]:
            pre_tile.append(blocking_factor)
            blocking_accum = reduce(mul, pre_tile, 1)
            recursive_tile(tile_permutations, pre_tile, (loop_extent + blocking_accum - 1) // blocking_accum,
                           loop_hint_level + 1, num_level)
        else:
            blocking_accum = reduce(mul, pre_tile, 1)
            for i in factors((loop_extent + blocking_accum - 1) // blocking_accum):
                if i >= para_hint:
                    new_pre_tile = copy.copy(pre_tile)
                    new_pre_tile.append(i)
                    new_blocking_accum = blocking_accum * i
                    recursive_tile(tile_permutations, new_pre_tile,
                                   (loop_extent + new_blocking_accum - 1) // new_blocking_accum, loop_hint_level + 1,
                                   num_level)

def factors(n):
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

def recursive_tile(tile_permutations, curr_loop_tile, n, curr_level, num_level):
    if curr_level == num_level - 1:
        curr_loop_tile.append(n)
        tile_permutations.append(curr_loop_tile)
        return

    for i in factors(n):
        new_loop_tile = copy.copy(curr_loop_tile)
        new_loop_tile.append(i)
        recursive_tile(tile_permutations, new_loop_tile, n // i, curr_level + 1, num_level)

def tile_perms(dimensions, num_levels, schedule=None):

    hint = schedule.schedule_hint if schedule else None

    all_tile_permutations = []
    for i, d in enumerate(dimensions):
        loop_hint = hint[i] if hint and i in hint else None
        all_tile_permutations.append(loop_tile(d, num_levels, loop_hint))
    tiles = list(itertools.product(*all_tile_permutations))
    # for tile in itertools.product(*all_tile_permutations):
        # TODO here the generated is a list of lists, not a list of tuples
        # if cost_model.valid_blocking_size(resource, dummy_mapping_point, layer):
        # yield list(tile)
    return tiles


def loop_tile(loop_extent, num_level, loop_hint=None):
    tile_permutations = []
    if not loop_hint:
        recursive_tile(tile_permutations, [], loop_extent, 0, num_level)
    else:
        loop_tile_with_hint(tile_permutations, loop_extent, num_level, loop_hint)

    return tile_permutations

def get_node_location(node: pm.Node) -> str:
    if "component_location" not in node.added_attrs:
        node.add_attribute("component_location", "DRAM")
        location = "DRAM"
    else:
        location = node.kwargs["component_location"]
    return location

# TODO: Fix this
def get_node_dtype(node, hag):
    return "fxp32"

def make_operand_from_node(node, hag, location=None):
    if not location:
        location = get_node_location(node)
    dtype = get_node_dtype(node, hag)
    dimensions = node.shape
    op = Operand(component_name=location, datatype=dtype, dimensions=dimensions)
    return op

