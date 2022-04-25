import torch
import torch.nn as nn

from custom_layers.custom_layer import CustomLayer

class CustomDropout(nn.Module, CustomLayer):
    """
    Sets neurons to zero with probability p, during both training and testing.
    """
    def __init__(self, p: int, on_at_inference: bool, **kwargs):
        nn.Module.__init__(self)
        CustomLayer.__init__(self, custom_layer_name="AlwaysDropout")
        self.p = p
        self.on_at_inference = on_at_inference
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        if self.training or self.on_at_inference:
            x_mask = torch.rand_like(x) < self.p
            x = x * x_mask
            x = x / self.p
            self.log_sparsity(x)
        else:
            return x
