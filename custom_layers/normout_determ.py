import torch
import torch.nn as nn

class DeterministicNormOut(nn.Module):
    """
    During training, sets ith neurons to zero with probability p_i, where p_i is the activation of the ith neuron divided 
    by the max activation of the layer, and then scales by 1/(1-p_i). When `use_abs` is True, we use the absolute value of 
    the activations instead of the activations themselves.
    """
    def __init__(self, use_abs=False):
        super().__init__()
        self.use_abs = use_abs
        if not self.use_abs:
            print("Not using absolute value of activations in NormOut!")
        
    def forward(self, x):
        if self.training:
            if self.use_abs:
                x_prime = abs(x)
            else:
                x_prime = x

            norm_x = x_prime / torch.max(x_prime, dim=1, keepdim=True)[0]
            x_mask = torch.rand_like(x) < norm_x
            x = x * x_mask
            # upscale by 1/(1-p_i)
            x = x / (1 - norm_x)
        return x