import torch
import torch.nn as nn

from custom_layers.custom_layer import CustomLayer

class TopK(nn.Module, CustomLayer):
    """
    Sets all but the K highest activation values to zero.
    """
    def __init__(self, k, on_at_inference, log_input_stats_bool, log_sparsity_bool, **kwargs):
        nn.Module.__init__(self)
        CustomLayer.__init__(self, custom_layer_name="TopK")
        self.k = k
        self.on_at_inference = on_at_inference
        self.log_input_stats_bool = log_input_stats_bool
        self.log_sparsity_bool = log_sparsity_bool
    
    def forward(self, x):
        if self.log_input_stats_bool:
            self.log_input_stats(x)
        if self.training or self.on_at_inference:
            _, indices = torch.topk(x, self.k, dim=1)
            top_k_mask = torch.zeros_like(x)
            top_k_mask = top_k_mask.scatter(1, indices, 1)
            x = x * top_k_mask
        if self.log_sparsity_bool:
            self.log_sparsity(x)
        return x
