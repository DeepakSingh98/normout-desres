import torch
import torch.nn as nn

from custom_layers.custom_layer import Custom_Layer

class AlwaysDropout(nn.Module, Custom_Layer):
    """
    Sets neurons to zero with probability p, during both training and testing.
    """
    def __init__(self, p=0.5, **kwargs):
        nn.Module.__init__(self)
        Custom_Layer.__init__(self, **kwargs)
        self.p = p

    def forward(self, x):
        x_mask = torch.rand_like(x) < self.p
        return  x * x_mask
