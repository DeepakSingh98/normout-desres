import torch
import torch.nn as nn

from custom_layers.custom_layer import Custom_Layer

class NormOut(nn.Module, Custom_Layer):
    """
    Sets ith neurons to zero with probability p_i, where p_i is the activation of the ith neuron divided 
    by the max activation of the layer. When `use_abs` is True, we use the absolute value of the activations 
    instead of the activations themselves.
    """
    def __init__(self, use_abs: bool,
                channel_max: bool, 
                on_at_inference: bool, 
                **kwargs):

        nn.Module.__init__(self)
        Custom_Layer.__init__(self, **kwargs)

        self.use_abs = use_abs
        self.channel_max = channel_max
        self.on_at_inference = on_at_inference

        if self.use_abs:
            print("Using absolute value of activations in NormOut!")
        else:
            print("Not using absolute value of activations in NormOut!")

        if self.channel_max:
            print("Using channel max in NormOut!")
        else:
            print("Not using channel max in NormOut!")

        
    def forward(self, x):
        if self.training or self.on_at_inference:
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