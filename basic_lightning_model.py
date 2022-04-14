import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import torchvision
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

class BasicLightningModel(pl.LightningModule, ABC):
    """
    Handles setting up the optimizer and data loader.
    """

    def __init__(
        self,
        # dataloader
        batch_size=256,
        num_workers=4,
        dset_name="CIFAR10",  # 'MNIST-Fashion' or 'CIFAR10'
        use_cifar_data_augmentation=False,
        # optimizer
        optimizer="SGDM",
        lr=0.01,
        weight_decay = 0.0001,
        momentum = 0.9,
        # catch other kwargs
        **kwargs
        ):

        pl.LightningModule.__init__(self)
        
        self.dset_name = dset_name
        self.use_data_augmentation = use_cifar_data_augmentation
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.num_workers = num_workers

        # dataset
        self.define_dataset(dset_name)

        # metrics
        self.train_acc: torchmetrics.Accuracy = torchmetrics.Accuracy()
        self.valid_acc: torchmetrics.Accuracy = torchmetrics.Accuracy()

    def define_dataset(self, dset_name):
        self.num_channels = 1
        if dset_name == "MNIST-Fashion":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
            self.training_set = torchvision.datasets.FashionMNIST(
                "./data", train=True, transform=transform, download=True
            )
            self.validation_set = torchvision.datasets.FashionMNIST(
                "./data", train=False, transform=transform, download=True
            )
            
            self.num_channels = 1
            self.num_classes = 10

        elif dset_name == "CIFAR10":
            # TODO USING DATA AUGMENTATION TO GET BASELINE

            # My version of CIFAR10 data augmentation
            pretrained_size = 224
            pretrained_means = [0.485, 0.456, 0.406]
            pretrained_stds = [0.229, 0.224, 0.225]

            train_transforms = transforms.Compose([
                                    transforms.Resize(pretrained_size),
                                    transforms.RandomRotation(5),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.RandomCrop(pretrained_size, padding=10),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=pretrained_means,
                                                            std=pretrained_stds)
                                ])

            test_transforms = transforms.Compose([
                                    transforms.Resize(pretrained_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=pretrained_means,
                                                            std=pretrained_stds)
                                ])

            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),                
                ]
                )

            if self.use_data_augmentation:
                self.training_set = torchvision.datasets.CIFAR10(
                    "./data", train=True, transform=train_transforms, download=True
                )

                self.validation_set = torchvision.datasets.CIFAR10(
                "./data", train=False, transform=test_transforms, download=True)

            else:
                self.training_set = torchvision.datasets.CIFAR10(
                    "./data", train=True, transform=transform, download=True
                )
                
                self.validation_set = torchvision.datasets.CIFAR10(
                    "./data", train=False, transform=transform, download=True
                )
                
            self.num_channels = 3
            self.num_classes = 10
        else:
            raise NotImplementedError("Dataset not implemented")
    
    def configure_optimizers(self):
        if self.optimizer == "SGDM":
            optimizer =  torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer == "Adam":
            optimizer =  torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("Optimizer not implemented")
        
        return {"optimizer": optimizer,
                "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1), # TODO note doing this
                "interval": "step",
                }
        }

    # dataloaders
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