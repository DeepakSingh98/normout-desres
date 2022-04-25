from lzma import MODE_NORMAL
import torch
import torch.nn as nn
import pytorch_lightning as pl

from pl_bolts.models.self_supervised import resnets

def resnet_layers(model_name: str, pretrained: bool, **kwargs):

    if model_name == "resnet18":
        model = resnets.resnet18(pretrained)
    elif model_name == "resnet34":
        model = resnets.resnet34(pretrained)
    elif model_name == "resnet50":
        model = resnets.resnet50(pretrained)
    elif model_name == "resnet101":
        model = resnets.resnet101(pretrained)
    elif model_name == "resnet152":
        model = resnets.resnet152(pretrained)
    elif model_name == "resnext50_32x4d":
        model = resnets.resnet50_32x4d(pretrained)
    elif model_name == "resnext101_32x8d":
        model = resnets.resnet101_32x8(pretrained)
    elif model_name == "wide_resnet50_2":
        model = resnets.wide_resnet50_2(pretrained)
    elif model_name == "wide_resnet101_2":
        model = resnets.wide_resnet101_2(pretrained)
    else:
        raise NotImplementedError("model type not implemented")
    
    layers = []
    
    for module in model.children():
        layers.append(module)

    return layers