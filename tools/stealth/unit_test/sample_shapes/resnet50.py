def resnet50_relu4d_1_7_7_512() -> tuple[tuple[int, ...], str]:
    # ReLU 4d from ResNet50
    return ((1, 7, 7, 512), "i32")


def resnet50_relu4d_1_14_14_256() -> tuple[tuple[int, ...], str]:
    # ReLU 4d from ResNet50
    return ((1, 14, 14, 256), "i32")


def resnet50_relu4d_1_28_28_128() -> tuple[tuple[int, ...], str]:
    # ReLU 4d from ResNet50
    return ((1, 28, 28, 128), "i32")


def resnet50_relu4d_1_56_56_64() -> tuple[tuple[int, ...], str]:
    # ReLU 4d from ResNet50
    return ((1, 56, 56, 64), "i32")


def resnet50_relu4d_1_112_112_64() -> tuple[tuple[int, ...], str]:
    # ReLU 4d from ResNet50
    return ((1, 112, 112, 64), "i32")


def resnet50_add4d4d_1_56_56_256() -> tuple[tuple[int, ...], str]:
    return ((1, 56, 56, 256), "i32")


def resnet50_add4d4d_1_28_28_512() -> tuple[tuple[int, ...], str]:
    return ((1, 28, 28, 512), "i32")
