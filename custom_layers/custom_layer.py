import torch
import wandb

from abc import ABC, abstractmethod

class CustomLayer(ABC):

    def __init__(self, custom_layer_name="NormOut", use_abs=True, max_type="spatial", on_at_inference=True):
        self.custom_layer_name = custom_layer_name
        self.use_abs = use_abs
        self.max_type = max_type
        self.on_at_inference = on_at_inference

    def set_index(self, num_id):
        self.num_id = num_id

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Forward pass not implemented")

    def log_sparsity(self, x: torch.Tensor):
        if len(x.shape) == 2:
            sparsity_vector = (x == 0).sum(dim=1)/x.shape[1]
        elif len(x.shape) == 4:
            sparsity_vector = (x == 0).sum(dim=(1,2,3))/(x.shape[1]*x.shape[2]*x.shape[3])
        else:
            raise ValueError("Sparsity can only be computed for 2D or 4D tensors")
        wandb.log({f"{self.custom_layer_name} {self.num_id} Sparsity": sparsity_vector}, commit=False)
        wandb.log({f"{self.custom_layer_name} {self.num_id} Sparsity Mean": sparsity_vector.mean()}, commit=False)
    
    '''
    
    def log_input_stats(self, x: torch.Tensor):
        if self.training or self.on_at_inference:
            if self.use_abs: 
                x_prime = abs(x)
            else:
                x_prime = x

            if self.max_type == "spatial":
                x_prime_max = torch.max(x_prime, dim=2, keepdim=True)[0]
                x_prime_mean = torch.max(x_prime_max, dim=3, keepdim=True)[0]
                x_prime_mean = torch.max(x_prime, dim=2, keepdim=True)[0]
                x_prime_min = torch.max(x_prime_max, dim=3, keepdim=True)[0]
                x_prime_min = torch.max(x_prime, dim=2, keepdim=True)[0]
                x_prime_max = torch.max(x_prime_max, dim=3, keepdim=True)[0]

                x_prime_mean = torch.mean(x_prime, dim=2)
                x_prime_min
                norm_x = x_prime / x_prime_max
            elif self.max_type == "channel":
                norm_x = x_prime / torch.max(x_prime, dim=1, keepdim=True)[0]
            elif self.max_type == 'global':
                x_prime_max = torch.max(x_prime, dim=1, keepdim=True)[0]
                x_prime_max = torch.max(x_prime, dim=2, keepdim=True)[0]
                x_prime_max = torch.max(x_prime_max, dim=3, keepdim=True)[0]
                norm_x = x_prime / x_prime_max
            else:
                raise NotImplementedError("NormOut max type not implemented")
    
    '''
