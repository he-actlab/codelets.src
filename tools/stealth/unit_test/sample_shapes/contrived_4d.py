def simd_1_128_128_2048() -> tuple[tuple[int, ...], str]:
    return ((1, 128, 128, 2048), "i32")


def simd_1_256_256_1024() -> tuple[tuple[int, ...], str]:
    return ((1, 256, 256, 1024), "i32")


def simd_1_512_512_512() -> tuple[tuple[int, ...], str]:
    return ((1, 512, 512, 512), "i32")


def simd_1_1024_1024_256() -> tuple[tuple[int, ...], str]:
    return ((1, 1024, 1024, 256), "i32")


def simd_1_2048_2048_64() -> tuple[tuple[int, ...], str]:
    return ((1, 2048, 2048, 64), "i32")
