from basic_lightning_model import BasicLightningModel
from utils import NormOut, TopK, Dropout
from torchvision.models import vgg16, vgg16_bn, VGG
from attacks import Attacks
import torch.nn as nn
import torch

# new models must define a forward, training_step, and validation_step method, and may define a on_train_epoch_end method.
class VGG16NormOut(Attacks, BasicLightningModel):
    def __init__(
        self, 
        vgg_no_batch_norm=False, 
        custom_layer_name="None", 
        normout_delay_epochs=0,
        normout_method="default",
        **kwargs
    ):
        BasicLightningModel.__init__(self, **kwargs)
        Attacks.__init__(self, **kwargs)

        # custom_layer
        if custom_layer_name == "ReLU":
            custom_layer = nn.ReLU(True)
        elif custom_layer_name == "Dropout":
            custom_layer = Dropout(p=p)
        elif custom_layer_name == "NormOut":
            custom_layer = NormOut(delay_epochs=normout_delay_epochs, method=normout_method)
        elif custom_layer_name == "TopK":
            custom_layer = TopK(k=k)
        else:
            raise ValueError("custom_layer_name must be 'None', 'Dropout', 'NormOut', or 'TopK'")
        if vgg_no_batch_norm:
            model: VGG = vgg16(pretrained=False, num_classes=self.num_classes)
        else: 
            model: VGG = vgg16_bn(pretrained=False, num_classes=self.num_classes)
        
        # logging
        # self.save_hyperparameters() # TODO Not working.

        # important bit - overriding the classifier
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            custom_layer,
            nn.Linear(4096, 4096),
            custom_layer,
            nn.Linear(4096, self.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def on_validation_epoch_end(self):
        Attacks.on_validation_epoch_end(self)
