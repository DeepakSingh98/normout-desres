import torch
import torch.nn as nn

from custom_layers.custom_layer import CustomLayer

class ExpOut(nn.Module, CustomLayer):

    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        CustomLayer.__init__(self, custom_layer_name="ExpOut")

    def forward(self, x):
        # for each input, we want the max activation of the whole activation tensor
        max_val = torch.zeros(x.shape[0], requires_grad=True).to(x.device)
        # for every input image, select the max of x_prime[channel]
        for i in range(x.shape[0]):
            max_val[i] = torch.max(x[i])
            
        norm_x = x / max_val.reshape(max_val.shape[0], *[1]*(len(x.shape)-1))        
        x = x * norm_x
        self.log_sparsity(x)
        return x