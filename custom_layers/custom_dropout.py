import torch
import torch.nn as nn
import wandb

from custom_layers.custom_layer import CustomLayer

class CustomDropout(nn.Module, CustomLayer):
    """
    Sets neurons to zero with probability p, during both training and testing.
    """
    def __init__(self, dropout_p: int, log_stats_bool: bool, log_sparsity_bool: bool, **kwargs):
        nn.Module.__init__(self)
        CustomLayer.__init__(self, custom_layer_name="Dropout")
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p)
        self.log_stats_bool = log_stats_bool
        self.log_sparsity_bool = log_sparsity_bool

    def forward(self, x):

        if self.log_stats_bool:
            self.log_stats(x)
        
        x = self.dropout(x)

        if self.log_sparsity_bool:
            self.log_sparsity(x)
            
        return x
