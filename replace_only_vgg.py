from base_model import BasicLightningModel
from custom_layers.custom_dropout import CustomDropout
from custom_layers.normout import NormOut
from custom_layers.normout import NormOut
from custom_layers.topk import TopK
from models.vgg16_layers import vgg16_features_avgpool_classifier
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import copy

import argparse
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from custom_model import CustomModel
from utils import set_tags
import wandb

class ReplacableVGG16BN(BasicLightningModel):
    """
    Inherits `BasicLightningModel` and defines a base model architecture specified by `model-name`. Models can be customized as follows:
    1. Replacing existing layers with custom layers (e.g., `--custom-layer-name NormOut --replace-layers 47 50` replaces the layers at indices 47 and 50 with `NormOut` layers).
    2. Removing layers (e.g., `--remove-layers 20` removes the layer at index 20)
    3. Inserting custom layers (e.g., `--custom-layer-name NormOut --insert-layer 53` inserts a `NormOut` layer at index 53). 
    *Note: The editing is performed in the following order: replace, remove, insert. All indices should be listed in increasing order.*
    """

    def __init__(
        self, 
        custom_layer_name, 
        neg_replace_layers,
        normalization_type,
        dropout_p=0.5,
        topk_k=40,
        on_at_inference=False, 
        no_abs=False,
        no_log_sparsity=False,
        log_input_stats=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.custom_layer_name = custom_layer_name
        self.preprocess_during_forward = False
        use_abs = not no_abs
        log_sparsity = not no_log_sparsity
        
        # configure custom layer
        if custom_layer_name is None:
            self.custom_layer = None
        elif custom_layer_name == "TopK":
            self.custom_layer = TopK(topk_k, on_at_inference, log_input_stats, log_sparsity)
        elif custom_layer_name == "Dropout":
            self.custom_layer = CustomDropout(dropout_p, on_at_inference, log_sparsity_bool=log_sparsity, log_input_stats_bool=log_input_stats)
        elif custom_layer_name == "NormOut":
            self.custom_layer = NormOut(normalization_type, log_sparsity_bool=log_sparsity, log_input_stats_bool=log_input_stats, use_abs=use_abs)
        else:
            raise ValueError("custom_layer_name must be 'Dropout', 'NormOut', or 'TopK'")
        
        # get model
        features, self.avgpool, classifier = vgg16_features_avgpool_classifier(self.num_channels, self.num_classes, True, dropout_p=dropout_p)
    
        if self.custom_layer is not None and neg_replace_layers is not None:
            for i in neg_replace_layers:
                classifier = self.replace_custom_layer(classifier, -i)
        
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(*classifier)

        print(self.classifier)

    def replace_custom_layer(self, layers, i, replace_with=None):
        if replace_with is not None:
            layer = copy.copy(replace_with)
        else:
            layer = copy.copy(self.custom_layer)
        print(f"{layers[i]} at index {i} replaced with {self.custom_layer_name}")
        layer.set_index(i)
        layers[i] = layer
        return layers

    def set_preprocess_during_forward(self, state: bool):
        self.preprocess_during_forward = state

    def forward(self, x):
        if self.preprocess_during_forward:
            x = transforms.Normalize(self.pretrained_means, self.pretrained_stds)(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # parse arguments
    # general settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs (default 100)")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size (default 256)")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers used for data loading (default 4)")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus to use (default 1)")
    parser.add_argument("--dset-name", type=str, default="CIFAR10", help="dataset name (default CIFAR10, also supports MNIST-Fashion)")
    parser.add_argument("--no-data-augmentation", default=False, action="store_true", help="Don't use data augmentation (default False)")
    parser.add_argument("--optimizer", type=str, default="SGDM", help="optimizer (default SGDM, also supports Adam)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default 0.01)")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum value (default 0.9)")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="weight decay value (default 0.0001)")
    parser.add_argument("--custom-tag", type=str, default=None, help="custom tag to be added to wandb log")
    # model settings
    parser.add_argument("--model-name", type=str, default="VGG16", help="model name (default VGG16, supports resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2)")
    parser.add_argument("--custom-layer-name", type=str, default=None, help="custom layer (default None, supports 'ReLU', 'NormOut', 'Dropout', 'SigmoidOut', and 'TopK')")
    parser.add_argument("--use-robustbench-data", default=False, action="store_true", help="Use robustbench data")
    parser.add_argument("--get-robustbench-layers", default=False, action="store_true", help="Get robustbench model layers (default False)")
    parser.add_argument("--pretrained", default=False, action="store_true", help="Use pretrained model")
    parser.add_argument("--topk-k", type=int, default=10, help="k value for TopK")
    parser.add_argument("--dropout-p", type=float, default=0.5, help="p value for Dropout (probability of neuron being dropped)")
    parser.add_argument("--no-batch-norm", default=False, action="store_true", help="Don't use batch norm (default False)")
    parser.add_argument("--replace-layers", type=int, nargs="+", default=None, help="layer indices at which the layer is placed with the custom layer (NOTE: happens after removal and insertion)")
    parser.add_argument("--remove-layers", type=int, nargs="+", default=None, help="layer indices at which the layer is removed from the model; give vals in ascending order")
    parser.add_argument("--insert-layers", type=int, nargs="+", default=None, help="layer indices at which a custom layer is inserted (NOTE: happens after removal)")
    # NormOut settings
    parser.add_argument("--no-abs", default=False, action="store_true", help="Don't use absolute value of input during NormOut (default False)")
    parser.add_argument("--max-type", type=str, default="spatial", help="Type of max to use in NormOut (default spatial, supports channel, global)")
    parser.add_argument("--on-at-inference", default=False, action="store_true", help="Turn layer on at inference time (default False)")
    parser.add_argument("--normout-delay-epochs", type=int, default=0, help="number of epochs to delay using normout")
    # SigmoidOut settings
    parser.add_argument("--normalization-type", type=str, default="SpatialMax", help="type of normalization to use (defautl SpatialMax), supports SpatialMax, SpatialSigmoid, TemporalMax, TemporalSigmoid, SpatiotemporalSigmoid, SpatiotemporalMax")
    # attacks
    parser.add_argument("--all-attacks-off", default=True, action="store_true", help="Turn all attacks off (default False)")
    parser.add_argument("--no-fgm", default=True, action="store_true", help="Don't use adversarial fgm (default False)")
    parser.add_argument("--no-pgd-ce", default=True, action="store_true", help="Don't use adversarial pgd-ce (default False)")
    parser.add_argument("--no-pgd-t", default=True, action="store_true", help="Don't use adversarial pgd-t (default False)")
    parser.add_argument("--no-cw-l2-ce", default=True, action="store_true", help="Don't use untargeted Carlini wagner L2 attacks (default False)")
    parser.add_argument("--no-cw-l2-t", default=True, action="store_true", help="Don't use targeted Carlini wagner L2 attacks (default False)")
    parser.add_argument("--no-fab",  default=True, action="store_true", help="Don't use Untargeted FAB attack (default False)")
    parser.add_argument("--no-fab-t",  default=True, action="store_true", help="Don't use FAB-T attack (default False)")
    parser.add_argument("--no-square-attack", default=True, action="store_true", help="Don't use square attack (default False)")
    parser.add_argument("--no-randomized-attack", default=True, action="store_true", help="Don't use randomized attacks (default False)") #TODO make False default
    parser.add_argument("--no-robustbench-linf", default=False, action="store_true", help="Don't use robustbench Linf autoattack (default False)")
    parser.add_argument("--no-robustbench-l2", default=True, action="store_true", help="Don't use robustbench L2 autoattack (default False)")
    parser.add_argument("--no-salt-and-pepper-attack", default=True, action="store_true", help="Don't use salt and pepper attack (default False)")
    parser.add_argument("--adv-eps", type=float, default=0.03, help="adversarial epsilon (default 0.03)")
    parser.add_argument("--pgd-steps", type=int, default=10, help="number of steps for PGD (default 10)")
    parser.add_argument("--corruption-types", type=str, nargs="+", default=None, help="type of corruption to add (supports shot_noise, motion_blur, snow, pixelate, gaussian_noise, defocus_blur, brightness, fog, zoom_blur, frost, glass_blur, impulse_noise, contrast, jpeg_compression, elastic_transform")
    parser.add_argument("--corruption-severity", type=int, default=1, help="Severity of corruption, supports ints 1 through 5 (default 1)")
    parser.add_argument("--log-adversarial-examples", default=False, action="store_true", help="Log adversarial examples (default False)")
    
    parser.add_argument("--neg-replace-layers", type=int, nargs="+", default=None, help="layer indices at which the layer is placed with the custom layer (NOTE: happens after removal and insertion)")
    args = parser.parse_args()

    # wandb
    tags = set_tags(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_logger = WandbLogger(
        project="normout",
        name=(("-").join(tags) + "-" + timestamp),
        tags=tags,
        entity="normout",
        config=args,
        settings=wandb.Settings(start_method="thread")
    )
    config = wandb.config

    # update args with wandb config
    for k, v in config.items():
        if k in args:
            args.__setattr__(k, v)

    # get model
    model = ReplacableVGG16BN(**vars(args))

    wandb_logger.watch(model)

    # train
    trainer = Trainer(gpus=args.num_gpus, logger=wandb_logger, max_epochs=args.epochs)
    trainer.fit(model)
