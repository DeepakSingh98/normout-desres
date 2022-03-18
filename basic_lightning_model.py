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
        batch_size=64,
        num_workers=4,
        dset_name="MNIST-Fashion",  # 'MNIST-Fashion' or 'CIFAR10'
        # optimizer
        optimizer="SGDM",
        lr=0.01,
        # catch other kwargs
        **kwargs
        ):

        super().__init__()
        
        self.dset_name = dset_name
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
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
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
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
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == "Adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("Optimizer not implemented")

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
        pass

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