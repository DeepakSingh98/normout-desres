from typing import List
from base_model import BasicLightningModel
from custom_layers.custom_dropout import CustomDropout
from custom_layers.normout import NormOut
from custom_layers.normout import NormOut
from custom_layers.topk import TopK
from models.resnet_layers import resnet_layers
from models.robustbench_model import robustbench_model
from models.vgg16_layers import vgg16_layers
import torchvision.transforms as transforms
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
        get_robustbench_layers,
        pretrained,
        no_batch_norm, 
        custom_layer_name, 
        dropout_p,
        topk_k,
        remove_layers,
        insert_layers,
        replace_layers,
        no_abs,
        normout_delay_epochs,
        no_log_sparsity,
        no_log_stats,
        normalization_type,
        temperature,
        softmax,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.custom_layer_name = custom_layer_name
        self.pretrained = pretrained
        self.model_name = model_name
        use_batch_norm = not no_batch_norm
        use_abs = not no_abs
        log_sparsity = not no_log_sparsity
        log_stats = not no_log_stats
        
        # configure custom layer
        if custom_layer_name is None:
            self.custom_layer = None
        elif custom_layer_name == "TopK":
            self.custom_layer = TopK(topk_k, log_stats, log_sparsity)
        elif custom_layer_name == "Dropout":
            self.custom_layer = CustomDropout(dropout_p, log_sparsity_bool=log_sparsity, log_stats_bool=log_stats)
        elif custom_layer_name == "NormOut":
            self.custom_layer = NormOut(normalization_type=normalization_type, 
                                        log_sparsity_bool=log_sparsity, 
                                        log_stats_bool=log_stats, 
                                        use_abs=use_abs,
                                        softmax=softmax,
                                        temperature=temperature)
        else:
            raise ValueError("custom_layer_name must be 'Dropout', 'NormOut', or 'TopK'")
        
        already_sequential = False
        self.using_robustbench = False

        # get model
        if model_name == "VGG16":
            layers: List[nn.Module] = vgg16_layers(self.num_channels, 
                                                    self.num_classes,
                                                    use_batch_norm,
                                                    dropout_p=dropout_p)
        elif model_name in [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "resnext50_32x4d",
            "resnext101_32x8d",
            "wide_resnet50_2",
            "wide_resnet101_2"
            ]:
            layers = resnet_layers(model_name, pretrained, self.num_classes)
        elif model_name in [
            "Carmon2019Unlabeled",
            "Standard",
            "Rebuffi2021Fixing_70_16_cutmix_extra"
            ]:
            self.model = robustbench_model(model_name, get_robustbench_layers)
            if self.custom_layer is not None:
                self.model.bn1 = self.custom_layer
                self.custom_layer.set_index(0)
            already_sequential = True
            self.using_robustbench = True
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
            for i in replace_layers:
                self.replace_custom_layer(layers, i)

        if not already_sequential:        
            self.model = nn.Sequential(*layers)

        self.report_state(model_name, custom_layer_name, insert_layers, self.custom_layer)

    def replace_custom_layer(self, layers, i, replace_with=None):
        if replace_with is not None:
            layer = copy.copy(replace_with)
        else:
            layer = copy.copy(self.custom_layer)
        print(f"{layers[i]} at index {i} replaced with {self.custom_layer_name}")
        layer.set_index(i)
        layers[i] = layer

    def insert_custom_layer(self, layers, i):
        print(f"{self.custom_layer_name} inserted at index {i}")
        layer = copy.copy(self.custom_layer)
        layer.set_index(i)
        layers.insert(i, layer)

    def forward(self, x, **kwargs):
        x = self.model(x)
        return x

    def report_state(self, model_name, custom_layer_name, insert_layers, custom_layer):
        """
        Useful logging.
        """
        print(f"Model is {model_name}!")
        if self.pretrained:
            print("Using pretrained model!")
        if custom_layer is not None:
            print(f"{custom_layer_name} layers in use at indices {insert_layers}")
        print(self.model)
