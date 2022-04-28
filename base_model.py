from abc import ABC, abstractmethod
import numpy as np
import scipy

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
        no_data_augmentation,
        use_robustbench_data,
        # optimizer
        optimizer,
        lr,
        weight_decay,
        momentum,
        # catch other kwargs
        use_ecoc,
        **kwargs
        ):

        pl.LightningModule.__init__(self)
        Attacks.__init__(self, **kwargs)
        
        # set attributes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dset_name = dset_name
        self.use_data_augmentation = not no_data_augmentation
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.use_robustbench_data = use_robustbench_data
        self.use_ecoc = use_ecoc

        if use_robustbench_data:
            self.use_data_augmentation = False

        # dataset
        self.define_dataset(dset_name)

        # accuracy metrics
        self.train_acc: torchmetrics.Accuracy = torchmetrics.Accuracy()
        self.valid_acc: torchmetrics.Accuracy = torchmetrics.Accuracy()

        if self.use_ecoc:
            # create encoding matrix M from a 16x16 Hadamard matrix
            M = scipy.linalg.hadamard(16).astype(np.float32)
            M[np.arange(0, self.num_classes, 2), 0]= -1
            np.random.seed(59); np.random.shuffle(M)
            idx = np.random.permutation(16)
            M = M[0:self.num_classes, idx[0:16]]
            self.M = torch.Tensor(M, device=self.device)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Forward pass not implemented")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y, y_hat)
        self.train_acc(y_hat, y)
        self.log("Train Accuracy", self.train_acc, on_step=False, on_epoch=True)
        self.log("Train Loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y, y_hat)
        self.valid_acc(y_hat, y)
        self.log("Validation Accuracy", self.valid_acc, on_step=False, on_epoch=True)
        self.log("Validation Loss", loss, on_step=False, on_epoch=True)
        return loss

    def calculate_loss(self, y, y_hat):
        if self.use_ecoc:
            # y is batch x 1, y_hat batch x 16
            # first, do tanh on y
            self.M = self.M.to(self.device)
            y_hat = torch.tanh(y_hat)
            # y is max of 0 and matrix mult of y and self.M
            y_hat = torch.max(torch.zeros((y_hat.shape[0], self.M.T.shape[1])).to(self.device), torch.matmul(y_hat, self.M.T))
            # y is now batch x 10
            # normalize by dividing by max of each row
            y_hat = y_hat / torch.max(y_hat, dim=1, keepdim=True)[0] # batch x 10 (class probs)
            loss = F.cross_entropy(y_hat, y)

        else:
            loss = F.cross_entropy(y_hat, y)
        return loss

    def define_dataset(self, dset_name):
        if dset_name == "MNIST-Fashion":
            
            assert not self.use_data_augmentation # no data aug supported for MNIST-Fashion

            self.pretrained_means = (0.5,)
            self.pretrained_stds = (0.5,)

            self.plain_transforms = transforms.Compose([
                            transforms.Resize(32), # TODO: why doesn't this work for 28x28?
                            transforms.ToTensor(), 
                            transforms.Normalize(self.pretrained_means, self.pretrained_stds)
                        ])
            
            self.training_set = torchvision.datasets.FashionMNIST(
                "./data", train=True, transform=self.plain_transforms, download=True
            )
            self.validation_set = torchvision.datasets.FashionMNIST(
                "./data", train=False, transform=self.plain_transforms, download=True
            )
            
            self.num_channels = 1
            self.num_classes = 10

        elif dset_name == "CIFAR10":

            self.pretrained_means = [0.4914, 0.4822, 0.4465]
            self.pretrained_stds = [0.2023, 0.1994, 0.2010]

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
                transforms.Normalize(self.pretrained_means, self.pretrained_stds)
            ])

            self.plain_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=self.pretrained_means,
                                                         std=self.pretrained_stds)
                                ])
            
            self.robustbench_transforms = transforms.Compose([transforms.ToTensor()])

            if self.use_data_augmentation:
                print("Using data augmentation")
                self.training_set = torchvision.datasets.CIFAR10(
                    "./data", train=True, transform=augment_transforms, download=True
                )
            elif self.use_robustbench_data:
                self.training_set = torchvision.datasets.CIFAR10(
                    "./data", train=True, transform=self.robustbench_transforms, download=True
                )

            else:
                self.training_set = torchvision.datasets.CIFAR10(
                    "./data", train=True, transform=self.plain_transforms, download=True
                )
            
            if self.use_robustbench_data:
                    self.validation_set = torchvision.datasets.CIFAR10(
                    "./data", train=False, transform=self.robustbench_transforms, download=True
                )

            else:    
                self.validation_set = torchvision.datasets.CIFAR10(
                    "./data", train=False, transform=self.plain_transforms, download=True
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