import torch
import torch.nn as nn

from custom_layers.custom_layer import CustomLayer

class CustomDropout(nn.Module, CustomLayer):
    """
    Sets neurons to zero with probability p, during both training and testing.
    """
    def __init__(self, p: int, on_at_inference: bool, log_input_stats_bool: bool, log_sparsity_bool: bool, **kwargs):
        nn.Module.__init__(self)
        CustomLayer.__init__(self, custom_layer_name="Dropout")
        self.p = p
        self.on_at_inference = on_at_inference
        self.dropout = nn.Dropout(p)
        self.log_input_stats_bool = log_input_stats_bool
        self.log_sparsity_bool = log_sparsity_bool

    def forward(self, x):
        if self.log_input_stats_bool:
            self.log_input_stats(x)
        if self.training or self.on_at_inference:
            x_mask = torch.rand_like(x) < self.p
            x = x * x_mask
            x = x / self.p
        if self.log_sparsity_bool:
            self.log_sparsity(x)
        return x
