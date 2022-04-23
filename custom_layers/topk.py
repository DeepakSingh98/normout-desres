import torch
import torch.nn as nn

from custom_layers.custom_layer import Custom_Layer

class TopK(nn.Module, Custom_Layer):
    """
    Sets all but the K highest activation values to zero.
    """
    def __init__(self, k, **kwargs):
        nn.Module.__init__(self)
        Custom_Layer.__init__(self, **kwargs)
        self.k = k
    
    def forward(self, x):
        _, indices = torch.topk(x, self.k, dim=1)
        top_k_mask = torch.zeros_like(x)
        top_k_mask = top_k_mask.scatter(1, indices, 1)
        x = x * top_k_mask
        return x
