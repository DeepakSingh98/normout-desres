import torch
import wandb

from abc import ABC, abstractmethod

class CustomLayer(ABC):

    def __init__(self, custom_layer_name):
        self.custom_layer_name = custom_layer_name

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