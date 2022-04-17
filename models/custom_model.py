from base_model import BasicLightningModel
from layers.normout import NormOut
from layers.topk import TopK
from models.vgg16 import vgg16
from attacks import Attacks
import torch.nn as nn
import torch

import gc

# new models must define a forward, training_step, and validation_step method, and may define a on_train_epoch_end method.
class Custom_Model(Attacks, BasicLightningModel):
    def __init__(
        self, 
        model_name,
        vgg_no_batch_norm, 
        custom_layer_name, 
        normout_delay_epochs,
        normout_method,
        p,
        k,
        remove_layers,
        insert_layers,
        replace_layers,
        **kwargs
    ):
        BasicLightningModel.__init__(self, **kwargs)
        Attacks.__init__(self, **kwargs)

        # custom_layer
        if custom_layer_name == "None":
            custom_layer = None
        elif custom_layer_name == "ReLU":
            custom_layer = nn.ReLU(True)
        elif custom_layer_name == "NormOut":
            custom_layer = NormOut(delay_epochs=normout_delay_epochs, method=normout_method)
        elif custom_layer_name == "TopK":
            custom_layer = TopK(k=k)
        else:
            raise ValueError("custom_layer_name must be 'ReLU', 'BaselineDropout', 'NormOut', or 'TopK'")

        # Get model layers
        if model_name == "VGG16":
            layers = vgg16(self.num_channels, self.num_classes, vgg_no_batch_norm)
        else:
            raise NotImplementedError("model type not implemented")

        # Model surgery
        if custom_layer is not None and replace_layers is not None:
            print("Layer replacements:")
            for i in replace_layers:
                print(f"{layers[i]} at index {i} replaced with {custom_layer_name}")
                layers[i] = custom_layer
                
        if remove_layers is not None:
            print("Layer removals:")
            for i in reversed(remove_layers): # Reversed to stop indices getting messed up
                print(f"{layers[i]} removed from index {i}")
                layers.pop(i)

        if custom_layer is not None and insert_layers is not None:
            print("Layer insertions:")
            for i in insert_layers:
                print(f"{custom_layer_name} inserted at index {i}")
                layers.insert(i, custom_layer)
                
        self.model = nn.Sequential(*layers)

        del layers

        print(f"Model is {model_name}!")

        if custom_layer is not None:
            print(f"{custom_layer_name} layers in use at indices {insert_layers}")
            
        print(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

    def on_validation_epoch_end(self):
        Attacks.on_validation_epoch_end(self)
