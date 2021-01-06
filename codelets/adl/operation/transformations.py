from .loop_op import Loop

def unroll(loop):
    pass

def fuse(loops):
    pass

def reorder(loops, loop_permutation):
    pass

def split(loop: Loop, split_size: int):
    assert loop.iter_count % split_size == 0

    outer_loop = Loop(f"{loop.name}", 0, split_size)
    inner_loop = Loop(f"{loop.name}{loop.name}", 0, split_size)




