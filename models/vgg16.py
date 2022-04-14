import torch
import torch.nn as nn

# Function which returns a list of the layers in VGG16
def vgg16(in_channels, num_classes, vgg_no_batch_norm):

    layers = []

    features_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
                    'M', 512, 512, 512, 'M']

    for c in features_config:
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size=2)]

        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if vgg_no_batch_norm:
                layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            in_channels = c

    layers += [nn.AdaptiveAvgPool2d(7)]
    
    layers += [nn.Flatten()]

    layers += [
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes)
    ]

    return layers