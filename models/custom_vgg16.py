import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
from utils import NormOut

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time

class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()

        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d(7)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h
    
def get_vgg_layers(config, custom_layer_indices, batch_norm):

    layers = []
    in_channels = 3

    # Add in NormOut layer indices
    for i in custom_layer_indices:
        config.insert(i, "N")

    for c in config:
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size=2)]
        
        elif c == 'N':
            layers += [NormOut(method="Abs", delay_epochs=0)]

        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c

    return nn.Sequential(*layers)

def get_vgg(custom_layer_indices, vgg_no_batch_norm=False, num_classes=10):

    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
                    'M', 512, 512, 512, 'M']

    vgg16_layers = get_vgg_layers(vgg16_config, custom_layer_indices, batch_norm=not vgg_no_batch_norm)

    OUTPUT_DIM = num_classes

    model = VGG(vgg16_layers, OUTPUT_DIM)

    print(model)

    return model