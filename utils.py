import torch
import torch.nn as nn

# CUSTOM LAYERS: take in z values and apply activations

class NormOut(nn.Module):
    """
    The normout layer takes the activations of the previous layer and sets neurons to zero
    with probability of their activation divided by the largest activation.
    """
    def __init__(self, method="abs", delay_epochs=0, exponent=2):
        super().__init__()
        self.delay_epochs = delay_epochs
        self.method = method
        self.exponent = exponent
        print(f"Normout method is {method}!")

        if self.method == "Abs" or self.method == "Overwrite":
            self.preprocess = abs

        elif self.method == "Exp":
            self.preprocess = lambda x: x ** self.exponent
        
        elif self.method == "ReLU":
            self.preprocess = nn.ReLU(True)

        elif self.method == "Softmax":
            self.preprocess = nn.Softmax()
    
        else:
            raise NotImplementedError("Normout method not implemented.")
        
    def forward(self, x):

        if self.method == "Softmax":
            norm_x = self.preprocess(x)
        
        elif self.method == "Overwrite":
            x = self.preprocess(x)
            norm_x = x / torch.max(x, dim=1, keepdim=True)[0]

        else:
            x_prime = self.preprocess(x)
            norm_x = x_prime / torch.max(x_prime, dim=1, keepdim=True)[0]

        x_mask = torch.rand_like(x) < norm_x
        x = x * x_mask
        return x


class TopK(nn.Module):
    """
    The TopK layer sets all but the K highest activation values to zero.
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k
    
    def forward(self, x):
        x = nn.ReLU(True)(x)
        _, indices = torch.topk(x, self.k, dim=1)
        top_k_mask = torch.zeros_like(x)
        top_k_mask = top_k_mask.scatter(1, indices, 1)
        x = x * top_k_mask
        return x
