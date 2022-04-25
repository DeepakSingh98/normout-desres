import torch
import torch.nn as nn

from custom_layer import CustomLayer

class Sigmoid(nn.Module, CustomLayer):

    def __init__(self,
                log_sparsity_bool: bool,
                log_input_stats_bool: bool,
                **kwargs
                ):

        nn.Module.__init__(self)
        CustomLayer.__init__(self, custom_layer_name="Sigmoid")

        self.log_sparsity_bool = log_sparsity_bool
        self.log_input_stats_bool = log_input_stats_bool

    def forward(self, x):
        if self.log_input_stats_bool:
            self.log_input_stats(x)
        x = 1 / (1 + np.exp(-x))
        if self.log_sparsity_bool:
            self.log_sparsity(x)
        return x