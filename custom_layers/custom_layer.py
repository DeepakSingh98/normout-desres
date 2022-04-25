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
    
    def log_input_stats(self, x: torch.Tensor):

        if self.max_type == "spatial":
            x_max = torch.max(x, dim=-2, keepdim=True)[0]
            x_max = torch.max(x_max, dim=-1, keepdim=True)[0]
            x_mean = torch.mean(x, dim=-2, keepdim=True)
            x_mean = torch.mean(x_mean, dim=-1, keepdim=True)
            x_min = torch.min(x, dim=-2, keepdim=True)[0]
            x_min = torch.min(x_min, dim=-1, keepdim=True)[0]
            x_std = torch.std(x, dim=-2, keepdim=True)
            x_std = torch.std(x_std, dim=-1, keepdim=True)
        elif self.max_type == "channel":
            x_max = torch.max(x, dim=1, keepdim=True)[0]
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x_min = torch.min(x, dim=1, keepdim=True)[0]
            x_std = torch.std(x, dim=1, keepdim=True)
        elif self.max_type == 'global':
            x_max = torch.max(x, dim=1, keepdim=True)[0]
            x_max = torch.max(x_max, dim=2, keepdim=True)[0]
            x_max = torch.max(x_max, dim=3, keepdim=True)[0]
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x_mean = torch.mean(x_mean, dim=2, keepdim=True)
            x_mean = torch.mean(x_mean, dim=3, keepdim=True)
            x_min = torch.min(x, dim=1, keepdim=True)[0]
            x_min = torch.min(x_min, dim=2, keepdim=True)[0]
            x_min = torch.min(x_min, dim=3, keepdim=True)[0]
            x_std = torch.std(x, dim=1, keepdim=True)
            x_std = torch.std(x_std, dim=2, keepdim=True)
            x_std = torch.std(x_std, dim=3, keepdim=True)
        else:
            raise NotImplementedError("NormOut max type not implemented")
            
        wandb.log({f"{self.custom_layer_name} {self.num_id} Input mean": x_mean}, commit=False)
        wandb.log({f"{self.custom_layer_name} {self.num_id} Input min": x_min}, commit=False)
        wandb.log({f"{self.custom_layer_name} {self.num_id} Input max": x_max}, commit=False)
        wandb.log({f"{self.custom_layer_name} {self.num_id} Input std dev": x_std}, commit=False)