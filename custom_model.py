from typing import List
from base_model import BasicLightningModel
from custom_layers.always_dropout import AlwaysDropout
from custom_layers.custom_layer import CustomLayer
from custom_layers.expout import ExpOut
from custom_layers.normout import NormOut
from custom_layers.topk import TopK
from custom_layers.expout import ExpOut
from models.vgg16_layers import vgg16_layers
import torch.nn as nn
import copy

class CustomModel(BasicLightningModel):
    """
    Inherits `BasicLightningModel` and defines a base model architecture specified by `model-name`. Models can be customized as follows:
    1. Replacing existing layers with custom layers (e.g., `--custom-layer-name NormOut --replace-layers 47 50` replaces the layers at indices 47 and 50 with `NormOut` layers).
    2. Removing layers (e.g., `--remove-layers 20` removes the layer at index 20)
    3. Inserting custom layers (e.g., `--custom-layer-name NormOut --insert-layer 53` inserts a `NormOut` layer at index 53). 
    *Note: The editing is performed in the following order: replace, remove, insert. All indices should be listed in increasing order.*
    """

    def __init__(
        self, 
        model_name,
        no_batch_norm, 
        custom_layer_name, 
        no_abs,
        no_standard_max,
        on_at_inference, 
        dropout_p,
        topk_k,
        remove_layers,
        insert_layers,
        replace_layers,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.custom_layer_name = custom_layer_name
        use_batch_norm = not no_batch_norm
        use_abs = not no_abs
        standard_max = not no_standard_max
        
        # configure custom layer
        if custom_layer_name is None:
            self.custom_layer = None
        elif custom_layer_name == "ReLU":
            self.custom_layer = nn.ReLU(True)
        elif custom_layer_name == "NormOut":
            self.custom_layer = NormOut(use_abs, standard_max, on_at_inference)
        elif custom_layer_name == "NormOutBlock":
            self.custom_layer = [nn.Conv2d(self.num_channels, 64, 3, 1), NormOut(use_abs, standard_max, on_at_inference), nn.ReLU(True)]
        elif custom_layer_name == "TopK":
            self.custom_layer = TopK(k=topk_k)
        elif custom_layer_name == "AlwaysDropout":
            self.custom_layer = AlwaysDropout(p=dropout_p)
        elif custom_layer_name == "ExpOut":
            self.custom_layer = ExpOut()
        else:
            raise ValueError("custom_layer_name must be 'ReLU', 'NormOut', or 'TopK'")

        # get model
        if model_name == "VGG16":
            layers: List[nn.Module] = vgg16_layers(self.num_channels, self.num_classes, use_batch_norm, dropout_p=dropout_p)
        else:
            raise NotImplementedError("model type not implemented")

        # perform surgery
        if remove_layers is not None:
            print("Layer removals:")
            for i in reversed(remove_layers): # Reversed to stop indices getting messed up
                print(f"{layers[i]} removed from index {i}")
                layers.pop(i)

        if self.custom_layer is not None and insert_layers is not None:
            print("Layer insertions:")
            for i in insert_layers:
                self.insert_custom_layer(layers, i)
        
        if self.custom_layer is not None and replace_layers is not None:
            print("Layer replacements:")
            for i in replace_layers:
                self.replace_custom_layer(layers, i)
                
        self.model = nn.Sequential(*layers)
        self.report_state(model_name, custom_layer_name, insert_layers, self.custom_layer)

    def replace_custom_layer(self, layers, i):
        print(f"{layers[i]} at index {i} replaced with {self.custom_layer_name}")
        layer = copy.copy(self.custom_layer)
        layer.set_index(i)
        layers[i] = layer

    def insert_custom_layer(self, layers, i):
        print(f"{self.custom_layer_name} inserted at index {i}")
        layer = copy.copy(self.custom_layer)
        layer.set_index(i)
        layers.insert(i, layer)

    def forward(self, x):
        x = self.model(x)
        return x

    def report_state(self, model_name, custom_layer_name, insert_layers, custom_layer):
        """
        Useful logging.
        """
        print(f"Model is {model_name}!")
        if custom_layer is not None:
            print(f"{custom_layer_name} layers in use at indices {insert_layers}")
        print(self.model)
