import torch
import torch.nn as nn

from custom_layers.custom_layer import CustomLayer

class AlwaysDropout(nn.Module, CustomLayer):
    """
    Sets neurons to zero with probability p, during both training and testing.
    """
    def __init__(self, p=0.5, **kwargs):
        nn.Module.__init__(self)
        CustomLayer.__init__(self, custom_layer_name="AlwaysDropout")
        self.p = p

    def forward(self, x):
        x_mask = torch.rand_like(x) < self.p
        out = x * x_mask
        self.log_sparsity(out)
        return out
