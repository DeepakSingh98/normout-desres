import torch
import torch.nn as nn

class NormOut(nn.Module):
    """
    Sets ith neurons to zero with probability p_i, where p_i is the activation of the ith neuron divided 
    by the max activation of the layer. When `use_abs` is True, we use the absolute value of the activations 
    instead of the activations themselves.
    """
    def __init__(self, use_abs: bool, channel_max: bool, off_at_inference: bool):
        super().__init__()
        self.use_abs = use_abs
        self.channel_max = channel_max
        self.off_at_inference = off_at_inference

        if self.use_abs:
            print("Using absolute value of activations in NormOut!")
        else:
            print("Not using absolute value of activations in NormOut!")

        if self.channel_max:
            print("Using channel max in NormOut!")
        else:
            print("Not using channel max in NormOut!")

        
    def forward(self, x):
        if self.training or not self.off_at_inference:
            if self.use_abs: 
                x_prime = abs(x)
            else:
                x_prime = x

            if self.channel_max:
                norm_x = x_prime / torch.max(x_prime, dim=1, keepdim=True)
            else:
                norm_x = x_prime / torch.max(x_prime, dim=1, keepdim=True)[0]

            x_mask = torch.rand_like(x) < norm_x
            x = x * x_mask
        
        return x