import torch
import torch.nn as nn

class TopK(nn.Module):
    """
    Sets all but the K highest activation values to zero.
    """
    def __init__(self, k):
        super().__init__()
        self.k = k
    
    def forward(self, x):
        _, indices = torch.topk(x, self.k, dim=1)
        top_k_mask = torch.zeros_like(x)
        top_k_mask = top_k_mask.scatter(1, indices, 1)
        x = x * top_k_mask
        return x
