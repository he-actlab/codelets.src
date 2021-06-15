from examples.genesys.genesys_qmodels import QLayer
import torch

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


