from codelets.examples import QLayer, shuffle_weights, dram_layout, gen_conv_testcase, get_model_values, gen_fc_layer_testcase
import torch
import numpy as np

def test_maxpool2D():
    input_var = torch.randint(10, (1, 1, 8, 8)) * 30
    model = QLayer('MaxPool2d', kernel_size=2, stride=2)
    output = model(input_var)
    model.eval()

def test_maxpool2D_8bit_truncate_output():
    input_var = torch.randint(10, (1, 1, 8, 8)) * 30
    model = QLayer('MaxPool2d', output_width=8, method='truncate', kernel_size=2, stride=2)
    output = model(input_var)
    model.eval()

def test_maxpool2D_8bit_scale_output():
    input_var = torch.randint(10, (1, 1, 8, 8)) * 30
    # The quantized output is scaled by 1 and shifted by 0. qoutput = int8(output / 1 + 0)
    model = QLayer('MaxPool2d', output_width=[1, 0, torch.qint8], method='scale', kernel_size=2, stride=2)
    output = model(input_var)
    model.eval()

def test_maxpool2D_8bit_truncate_input():
    input_var = torch.randint(10, (1, 1, 8, 8)) * 30
    model = QLayer('MaxPool2d', input_width=8, method='truncate', kernel_size=2, stride=2)
    output = model(input_var)
    model.eval()

def test_maxpool2D_8bit_scale_input():
    input_var = torch.randint(10, (1, 1, 8, 8)) * 30
    model = QLayer('MaxPool2d', input_width=[1, 0, torch.qint8], method='scale', kernel_size=2, stride=2)
    output = model(input_var)
    model.eval()

def test_conv2D():
    input_var = torch.randint(10, (1, 1, 8, 8)) * 30
    model = QLayer('Conv2d', in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
    model.weight.data.fill_(1)
    output = model(input_var)
    model.eval()
    # since conv2d is done with kernel of size 1 with all weights 1, output is equal to input
    assert (input_var != output).sum() == 0

def test_conv2D_8bit_turncate_output():
    input_var = torch.randint(10, (1, 1, 8, 8)) * 30
    model = QLayer('Conv2d', output_width=8, method='truncate', in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
    model.weight.data.fill_(1)
    output = model(input_var)
    model.eval()

def test_shuffle_weights():
    weights = np.random.randint(low=0, high=64, size=(1,1,128,128), dtype=np.int8)
    print(weights)
    print(shuffle_weights(weights))

def test_dram_layout():
    weights = np.random.randint(low=-3, high=3, size=(1,1,8,8), dtype=np.int8)
    weights[0][0][0][0] = 2
    weights[0][0][0][1] = -3
    weights[0][0][0][2] = 2
    weights[0][0][0][3] = 0
    print(weights)
    concat_dram_weights = dram_layout(weights)
    print(concat_dram_weights)
    # (2 << 24) + (253 << 16) + (2 << 8) + 0
    # -3 is represented as 253 in 2s complement
    assert concat_dram_weights[0] == 50135552

def test_gen_conv_testcase():
    gen_conv_testcase((1, 128, 128, 64), (1, 1, 64, 64))

def test_gen_fc_layer_testcase():
    np.random.seed(10)
    gen_fc_layer_testcase((1,512), (1, 1024), big_tile_size={'P':64, 'N':512}, bias = False)

def test_resnet_layer_extraction():
    # get_model_values("resnet18", "Conv2D", 0)
    get_model_values("resnet18", "Linear", 0)






