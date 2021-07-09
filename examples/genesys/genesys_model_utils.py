import argparse
import torch
from torchvision import models
from PIL import Image
from torchvision import transforms
import urllib

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children

def extract_layer(model, layer_type, layer_number):
    layers = get_children(model)
    layer_num = 0

    for l in layers:
        if layer_type in l._get_name().lower():
            if layer_num == layer_number:
                return (l.weight(), l.bias())
            else:
                layer_num += 1
    raise RuntimeError(f"Unable to find layer {layer_type} in model")

def get_resnet18(quantized, layer_type, layer_number):
    model = models.quantization.resnet18(pretrained=True, quantize=quantized)
    return extract_layer(model, layer_type, layer_number), model


def get_resnet50(quantized, layer_type, layer_number):
    model = models.quantization.resnet50(pretrained=True, quantized=quantized)
    return extract_layer(model, layer_type, layer_number), model

def get_sample_input(quantize):
    import urllib
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    ## Preprocess image
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    if quantize:
        pass
    return input_batch

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='PyTorch Model Weight Extractor')
    argparser.add_argument('-m', '--model_name', required=True,
                           help='Name of the benchmark to create. One of "resnet18", "lenet')
    argparser.add_argument('-q', '--quantized', type=str2bool, nargs='?', default=True,
                           const=True, help='Whether or not to retreive a quantized model')
    argparser.add_argument('-l', '--layer_type', required=True, help='The layer type to extract from the model')
    argparser.add_argument('-ln', '--layer_number', type=int, default=0, help='The nth layer of type "layer-type" '
                                                                              'to extract. Default is 0.')
    args = argparser.parse_args()

    if args.model_name == "resnet18":
        _ = get_resnet18(args.quantized, args.layer_type, args.layer_number)
    elif args.model_name == "resnet50":
        _ = get_resnet50(args.quantized, args.layer_type, args.layer_number)
    raise RuntimeError(f"Model {args.model_name} is not currently supported")