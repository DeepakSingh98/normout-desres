from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision
import torchvision.transforms as transforms

from attacks import Attacks

class BasicLightningModel(Attacks, pl.LightningModule, ABC):
    """
    Defines a base `pl.LightningModule` inherited by all models, responsible for configuring 
    optimizers and dataloaders, and running adversarial attacks (by inheriting `Attacks`).
    """

    def __init__(
        self,
        # dataloader
        batch_size,
        num_workers,
        dset_name,
        use_cifar_data_augmentation,
        # optimizer
        optimizer,
        lr,
        weight_decay,
        momentum,
        # catch other kwargs
        **kwargs
        ):

        pl.LightningModule.__init__(self)
        Attacks.__init__(self, **kwargs)
        
        # set attributes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dset_name = dset_name
        self.use_data_augmentation = use_cifar_data_augmentation
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

        # dataset
        self.define_dataset(dset_name)

        # accuracy metrics
        self.train_acc: torchmetrics.Accuracy = torchmetrics.Accuracy()
        self.valid_acc: torchmetrics.Accuracy = torchmetrics.Accuracy()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Forward pass not implemented")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat, y)
        self.log("Train Accuracy", self.train_acc, on_step=False, on_epoch=True)
        self.log("Train Loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.valid_acc(y_hat, y)
        self.log("Validation Accuracy", self.valid_acc, on_step=False, on_epoch=True)
        self.log("Validation Loss", loss, on_step=False, on_epoch=True)
        return loss

    def define_dataset(self, dset_name):
        if dset_name == "MNIST-Fashion":
            
            assert not self.use_data_augmentation # no data aug supported for MNIST-Fashion

            transform = transforms.Compose([
                            transforms.Resize(32), # TODO: why doesn't this work for 28x28?
                            transforms.ToTensor(), 
                            transforms.Normalize((0.5,), (0.5,))
                        ])
            self.training_set = torchvision.datasets.FashionMNIST(
                "./data", train=True, transform=transform, download=True
            )
            self.validation_set = torchvision.datasets.FashionMNIST(
                "./data", train=False, transform=transform, download=True
            )
            
            self.num_channels = 1
            self.num_classes = 10

        elif dset_name == "CIFAR10":

            pretrained_means = [0.4914, 0.4822, 0.4465]
            pretrained_stds = [0.2023, 0.1994, 0.2010]

            # per https://github.com/moritzhambach/Image-Augmentation-in-Keras-CIFAR-10-
            augment_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                # width shift
                transforms.RandomApply([
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1),
                        shear=0
                    )
                ], p=0.5),
                # height shift
                transforms.RandomApply([
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1),
                        shear=0
                    )
                ], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(pretrained_means, pretrained_stds)
            ])

            plain_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=pretrained_means,
                                                         std=pretrained_stds)
                                ])

            if self.use_data_augmentation:
                print("Using data augmentation")
                self.training_set = torchvision.datasets.CIFAR10(
                    "./data", train=True, transform=augment_transforms, download=True
                )
            else:
                self.training_set = torchvision.datasets.CIFAR10(
                    "./data", train=True, transform=plain_transforms, download=True
                )
                
            self.validation_set = torchvision.datasets.CIFAR10(
                "./data", train=False, transform=plain_transforms, download=True
            )
                
            self.num_channels = 3
            self.num_classes = 10

        else:
            raise NotImplementedError("Dataset not implemented (must be MNIST-Fashion or CIFAR10)")
    
    def configure_optimizers(self):
        if self.optimizer == "SGDM":
            optimizer =  torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer == "Adam":
            optimizer =  torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("Optimizer not implemented")
        return optimizer

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    # run attacks 
    def on_validation_epoch_end(self):
        Attacks.on_validation_epoch_end(self)