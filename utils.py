import torch
import torch.nn as nn

class NormOut(nn.Module):
    """
    The normout layer takes the activations of the previous layer and sets neurons to zero
    with probability of their activation divided by the largest activation.
    """
    def forward(self, x):
        """
        Args:
            input: The activations of the previous layer.
        """
        # divide by biggest value in the activation per input
        norm_x = x / torch.max(x, dim=1, keepdim=True)[0]
        x_mask = torch.rand_like(x) < norm_x
        x = x * x_mask
        return x

class TopK(nn.Module):
    """
    The TopK layer sets all but the K highest activation values to zero.
    """
    def __init__(self, k: int):
        self.k = k
    
    def forward(self, x):
        _, indices = torch.topk(x, self.topk_k, dim=1)
        top_k_mask = torch.zeros_like(x)
        top_k_mask = top_k_mask.scatter(1, indices, 1)
        x = x * top_k_mask
        return x
