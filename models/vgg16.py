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
        dropout_style="None", 
        normout_delay_epochs=0,
        normout_method="default",
        **kwargs
    ):
        BasicLightningModel.__init__(self, **kwargs)
        Attacks.__init__(self, **kwargs)

        # dropout
        if dropout_style == "None":
            dropout = NormOut(method="None")
        elif dropout_style == "Dropout":
            dropout = Dropout(p=p)
        elif dropout_style == "NormOut":
            dropout = NormOut(delay_epochs=normout_delay_epochs, method=normout_method)
        elif dropout_style == "TopK":
            dropout = TopK(k=k)
        else:
            raise ValueError("dropout_style must be 'None', 'Dropout', 'NormOut', or 'TopK'")
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
            dropout,
            nn.Linear(4096, 4096),
            dropout,
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
