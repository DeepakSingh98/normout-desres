from typing import List
from base_model import BasicLightningModel
from layers.normout import NormOut
from layers.topk import TopK
from models.vgg16_layers import vgg16_layers
import torch.nn as nn

class EditableModel(BasicLightningModel):
    """
    Creates an editable version of `model_name`, where specified layers can be removed or replaced by 
    `custom_layer_name` layers, and `custom_layer_name` layers can be inserted at specified indices.
    """

    def __init__(
        self, 
        model_name,
        vgg_no_batch_norm, 
        custom_layer_name, 
        normout_delay_epochs,
        normout_method,
        dropout_p,
        topk_k,
        remove_layers,
        insert_layers,
        replace_layers,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # configure custom layer
        custom_layer = None
        if custom_layer_name == "ReLU":
            custom_layer = nn.ReLU(True)
        elif custom_layer_name == "NormOut":
            custom_layer = NormOut(delay_epochs=normout_delay_epochs, method=normout_method)
        elif custom_layer_name == "TopK":
            custom_layer = TopK(k=topk_k)
        else:
            raise ValueError("custom_layer_name must be 'ReLU', 'NormOut', or 'TopK'")

        # get model
        if model_name == "VGG16":
            layers: List[nn.Module] = vgg16_layers(self.num_channels, self.num_classes, vgg_no_batch_norm, dropout_p=dropout_p)
        else:
            raise NotImplementedError("model type not implemented")

        # perform surgery
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
        self.report_state(model_name, custom_layer_name, insert_layers, custom_layer)

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
