import torch
import torch.nn as nn

class NormOut(nn.Module):
    """
    The normout layer takes the activations of the previous layer and sets neurons to zero
    with probability of their activation divided by the largest activation.
    """
    def __init__(self, method="default", exponent=2, delay_epochs=0):
        super().__init__()
        self.delay_epochs = delay_epochs
        self.method = method
        
    def forward(self, x):
        """
        Args:
            input: The dot product of the weights of the previous layer and that layer's 
            input.
            Moved ReLU into this function for implementation of abs_normout, which does not
            use ReLU.
        """
        if self.method == "abs":
            x = abs(x)

        elif self.method == "exp":
            x = x ** exponent
        
        else:
            x = nn.ReLU(x)

        # divide by biggest value in the activation per input
        norm_x = x / torch.max(x, dim=1, keepdim=True)[0]
        x_mask = torch.rand_like(x) < norm_x
        x = x * x_mask
        return x

class TopK(nn.Module):
    """
    The TopK layer sets all but the K highest activation values to zero.
    """
    def __init__(self, , activation="ReLU", k: int):
        super().__init__()
        self.k = k
    
    def forward(self, x):
        x = nn.ReLU(x)
        _, indices = torch.topk(x, self.topk_k, dim=1)
        top_k_mask = torch.zeros_like(x)
        top_k_mask = top_k_mask.scatter(1, indices, 1)
        x = x * top_k_mask
        return x
