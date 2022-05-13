from os import truncate
import torch
import torch.nn as nn

from custom_layers.custom_layer import CustomLayer


class SigmoidOut(nn.Module, CustomLayer):
    """
    Sets ith neurons to zero with probability p_i, where p_i is the sigmoid of the standardized ith neuron.
    """

    def __init__(self, use_abs: bool, normalization_type: str, log_sparsity_bool: bool, log_input_stats_bool: bool, **kwargs):

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
            else:
                x_prime = x

            if self.normalization_type == "TemporalSigmoid":
                # Standardize across batch
                x_prime = (x_prime - torch.mean(x_prime, dim=0, keepdim=True)) / torch.std(x_prime, dim=0, keepdim=True)
                norm_x = torch.sigmoid(x_prime)
            
            elif self.normalization_type == "TemporalMax":
                # Take max across batch
                x_prime_max = torch.max(x_prime, dim=0, keepdim=True)[0]
                norm_x = x_prime / x_prime_max

            elif self.normalization_type == "SpatialSigmoid":
                # Standardize across layer
                x_prime = (x_prime - torch.mean(x_prime, dim=1, keepdim=True)) / torch.std(x_prime, dim=1, keepdim=True)
                norm_x = torch.sigmoid(x_prime)

            elif self.normalization_type == "SpatialMax":
                # Take max across layer
                x_prime_max =  torch.max(x_prime, dim=1, keepdim=True)[0]
                norm_x = x_prime / x_prime_max
            
            elif self.normalization_type == "SpatiotemporalSigmoid":
                # Standardize across batch
                x_prime = (x_prime - torch.mean(x_prime, dim=0, keepdim=True)) / torch.std(x_prime, dim=0, keepdim=True)
                # Standardize across layer
                x_prime = (x_prime - torch.mean(x_prime, dim=1, keepdim=True)) / torch.std(x_prime, dim=1, keepdim=True)
                norm_x = torch.sigmoid(x_prime)

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
