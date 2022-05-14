from os import truncate
import torch
import torch.nn as nn

from custom_layers.custom_layer import CustomLayer


class NormOut(nn.Module, CustomLayer):
    """
    Sets ith neurons to zero with probability p_i, where p_i is the activation of the ith neuron divided 
    by the max activation of the layer. When `use_abs` is True, we use the absolute value of the activations 
    instead of the activations themselves.
    """
    def __init__(self, normalization_type: str, log_sparsity_bool: bool, log_input_stats_bool: bool, use_abs: bool, **kwargs):

        nn.Module.__init__(self)
        CustomLayer.__init__(self, custom_layer_name="NormOut")

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
            else:
                x_prime = x
            
            if self.normalization_type == "TemporalMax":
                # Take max across batch
                x_prime_max = torch.max(x_prime, dim=0, keepdim=True)[0]
                norm_x = x_prime / x_prime_max

            elif self.normalization_type == "SpatialMax":
                # Take max across layer
                x_prime_max =  torch.max(x_prime, dim=1, keepdim=True)[0]
                norm_x = x_prime / x_prime_max

            elif self.normalization_type == "SpatiotemporalMax":
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
        
        self.log_input_stats(x)

        return x
