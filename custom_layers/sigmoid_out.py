import torch
import torch.nn as nn

from custom_layers.custom_layer import CustomLayer

class SigmoidOut(nn.Module, CustomLayer):
    """
    Sets ith neurons to zero with probability p_i, where p_i is the sigmoid of the ith neuron.
    """
    def __init__(
        self, 
        log_sparsity_bool: bool,
        log_input_stats_bool: bool,
        **kwargs
    ):

        nn.Module.__init__(self)
        CustomLayer.__init__(self, custom_layer_name="SigmoidOut")

        self.log_sparsity_bool = log_sparsity_bool
        self.log_input_stats_bool = log_input_stats_bool

    def forward(self, x):

        if self.log_input_stats_bool:
            self.log_input_stats(x)
        if self.training:
            # dropout x with probability sigmoid(x)
            norm_x = torch.abs(torch.sigmoid(x))
            x_mask = torch.rand_like(x) < norm_x
            x = x * x_mask
        if self.log_sparsity_bool:
            self.log_sparsity(x)
        return x