from tkinter import X
import torch
import torch.nn as nn

from custom_layers.custom_layer import CustomLayer


class SigmoidOut(nn.Module, CustomLayer):
    """
    Sets ith neurons to zero with probability p_i, where p_i is the sigmoid of the standardized ith neuron.
    """

    def __init__(self, use_abs: bool, log_sparsity_bool: bool, log_input_stats_bool: bool, normalization_type: str, **kwargs):

        nn.Module.__init__(self)
        CustomLayer.__init__(self, custom_layer_name="SigmoidOut")

        self.use_abs = use_abs
        self.normalization_type = normalization_type
        self.log_sparsity_bool = log_sparsity_bool
        self.log_input_stats_bool = log_input_stats_bool

    def forward(self, x):

        if self.log_input_stats_bool:
            self.log_input_stats(x)
        if self.training:

            if self.use_abs:
                x_prime = abs(x)

            if self.normalization_type == "temporal_sigmoid":
                # Standardize across batch
                x_prime = (x - torch.mean(x, dim=0)) / torch.std(x, dim=0)
                norm_x = torch.sigmoid(x_prime)
            
            elif self.normalization_type == "temporal_max":
                # Take max across batch
                x_prime_max = torch.max(x_prime, dim=0, keepdim=True)[0]
                norm_x = x_prime / x_prime_max

            elif self.normalization_type == "spatial_sigmoid":
                # Standardize across layer
                x_prime = (x - torch.mean(x, dim=1)) / torch.std(x, dim=1)
                norm_x = torch.sigmoid(x_prime)

            elif self.normalization_type == "spatial_max":
                # Take max across layer
                x_prime_max =  torch.max(x_prime, dim=1, keepdim=True)[0]
                norm_x = x_prime / x_prime_max
            
            elif self.normalization_type == "spatiotemporal_sigmoid":
                # Standardize across batch
                x_prime = (x - torch.mean(x, dim=0)) / torch.std(x, dim=0)
                # Standardize across layer
                x_prime = (x_prime - torch.mean(x_prime, dim=1)) / torch.std(x_prime, dim=1)
                norm_x = torch.sigmoid(x_prime)

            elif self.normalization_type == "spatiotemporal_max":
                # Take max across batch
                x_prime_max = torch.max(x_prime, dim=0, keepdim=True)[0]
                # Take max across layer
                x_prime_max = torch.max(x_prime_max, dim=1, keepdim=True)[0]
                norm_x = x_prime / x_prime_max

            else:
                raise NotImplementedError("normalization type not implemented")
 
            x_mask = torch.rand_like(x) < norm_x
            x = x * x_mask

        if self.log_sparsity_bool:
            self.log_sparsity(x)
            
        return x
