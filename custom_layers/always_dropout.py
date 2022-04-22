import torch
import torch.nn as nn

class AlwaysDropout(nn.Module):
    """
    Sets neurons to zero with probability p, during both training and testing.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        x_mask = torch.rand_like(x) < self.p
        return  x * x_mask
