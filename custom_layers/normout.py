import torch
import torch.nn as nn

from custom_layers.custom_layer import CustomLayer

class NormOut(nn.Module, CustomLayer):
    """
    Sets ith neurons to zero with probability p_i, where p_i is the activation of the ith neuron divided 
    by the max activation of the layer. When `use_abs` is True, we use the absolute value of the activations 
    instead of the activations themselves.
    """
    def __init__(self, use_abs: bool,
                max_type: str, 
                on_at_inference: bool, 
                log_sparsity_bool: bool,
                log_input_stats_bool: bool,
                **kwargs):

        nn.Module.__init__(self)
        CustomLayer.__init__(self, custom_layer_name="NormOut", use_abs=use_abs, max_type=max_type)

        self.use_abs = use_abs
        self.max_type = max_type
        self.on_at_inference = on_at_inference
        self.log_sparsity_bool = log_sparsity_bool
        self.log_input_stats_bool = log_input_stats_bool

        if self.use_abs:
            print("Using absolute value of activations in NormOut!")
        else:
            print("Not using absolute value of activations in NormOut!")

        if self.max_type == "spatial":
            print("Taking max across pixels per channel!")
        elif self.max_type == "channel":
            print("Taking max across channels per pixel!")
        elif self.max_type == "global":
            print("Taking max across both channels and pixels!")
        else:
            raise NotImplementedError("NormOut max type not implemented")


    def forward(self, x):

        if self.log_input_stats_bool:
            self.log_input_stats(x)
        if self.training or self.on_at_inference:
            if self.use_abs: 
                x_prime = abs(x)
            else:
                x_prime = x

            if self.max_type == "spatial":
                import ipdb; ipdb.set_trace()
                x_prime_max = torch.max(x_prime, dim=-2, keepdim=True)[0] #TODO: check if this is SpatioTemporal
                x_prime_max = torch.max(x_prime_max, dim=-1, keepdim=True)[0]
                norm_x = x_prime / x_prime_max
            elif self.max_type == "channel":
                norm_x = x_prime / torch.max(x_prime, dim=1, keepdim=True)[0]
            elif self.max_type == 'global':
                assert len(x.shape) == 4, "NormOut max type 'global' only implemented for 4D tensors. Use 'channel' or 'spatial' instead."
                x_prime_max = torch.max(x_prime, dim=1, keepdim=True)[0]
                x_prime_max = torch.max(x_prime, dim=2, keepdim=True)[0]
                x_prime_max = torch.max(x_prime_max, dim=3, keepdim=True)[0]
                norm_x = x_prime / x_prime_max
            else:
                raise NotImplementedError("NormOut max type not implemented")

            x_mask = torch.rand_like(x) < norm_x
            x = x * x_mask
            #x = x / (norm_x + 10e-8) # Inverted NormOut scaling
        if self.log_sparsity_bool:
            self.log_sparsity(x)
        return x