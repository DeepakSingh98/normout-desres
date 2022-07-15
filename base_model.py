from abc import ABC, abstractmethod
from logging import logProcesses
import numpy as np
import scipy

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision
import torchvision.transforms as transforms

from adversarial_attacks.attacks import Attacks

from data_utils import SubDataset

class BasicLightningModel(Attacks, pl.LightningModule, ABC):
    """
    Defines a base `pl.LightningModule` inherited by all models, responsible for configuring 
    optimizers and dataloaders, and running adversarial attacks (by inheriting `Attacks`).
    """

    def __init__(
        self,
        # dataloader
        batch_size,
        epochs,
        num_workers,
        dataset_name,
        num_tasks,
        no_data_augmentation,
        use_robustbench_data,
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
        self.dataset_name = dataset_name
        self.num_tasks = num_tasks
        self.current_task = 0
        self.use_data_augmentation = not no_data_augmentation
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.use_robustbench_data = use_robustbench_data

        if use_robustbench_data:
            self.use_data_augmentation = False
        
        self.dataset(dataset_name, num_tasks)

        # accuracy metrics
        self.train_acc: torchmetrics.Accuracy = torchmetrics.Accuracy()
        self.test_acc: torchmetrics.Accuracy = torchmetrics.Accuracy()

        self.epochs_per_task = int(epochs // num_tasks)
        if epochs % num_tasks != 0:
            raise ValueError("Total number of epochs must be exactly divisible by the number of tasks.")

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

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.calculate_loss(y, y_hat)
        self.test_acc(y_hat, y)
        self.log("Test Accuracy", self.test_acc, on_step=False, on_epoch=True)
        self.log("Test Loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def calculate_loss(self, y, y_hat):
        return F.cross_entropy(y_hat, y)

    def dataset(self, dataset_name, num_tasks):

        print(f"Dataset: {dataset_name}")

        if dataset_name == "CIFAR10" or dataset_name == "SplitCIFAR10":

            self.preprocess_means = [0.4914, 0.4822, 0.4465]
            self.preprocess_stds = [0.2023, 0.1994, 0.2010]

            # per https://github.com/moritzhambach/Image-Augmentation-in-Keras-CIFAR-10-
            self.augment_transforms = [
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
                transforms.Normalize(self.preprocess_means, self.preprocess_stds)
            ]

            self.plain_transforms = [
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=self.preprocess_means,
                                                         std=self.preprocess_stds)
                                ]
            
            self.robustbench_transforms = [transforms.ToTensor()]

            self.augment_transforms = transforms.Compose(self.augment_transforms)
            self.plain_transforms = transforms.Compose(self.plain_transforms)
            self.robustbench_transforms = transforms.Compose(self.robustbench_transforms)

            if dataset_name == "SplitCIFAR10":

                # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
                permutation = np.random.permutation(list(range(10)))
                target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            
            else: 
                target_transform = None
            
            if self.use_data_augmentation:
                print("Using data augmentation")
                self.training_set = torchvision.datasets.CIFAR10(
                    "./data", train=True, transform=self.augment_transforms, download=True, target_transform=target_transform
                )
            elif self.use_robustbench_data:
                self.training_set = torchvision.datasets.CIFAR10(
                    "./data", train=True, transform=self.robustbench_transforms, download=True
                )

            else:
                self.training_set = torchvision.datasets.CIFAR10(
                    "./data", train=True, transform=self.plain_transforms, download=True, target_transform=target_transform
                )
            
            if self.use_robustbench_data:
                    self.test_set = torchvision.datasets.CIFAR10(
                    "./data", train=False, transform=self.robustbench_transforms, download=True
                )

            else:    
                self.test_set = torchvision.datasets.CIFAR10(
                    "./data", train=False, transform=self.plain_transforms, download=True, target_transform=target_transform
                )
                
            self.num_channels = 3
            self.num_classes = 10
        
        if dataset_name == 'SplitCIFAR10':

            # check for number of tasks
            if num_tasks > 10:
                raise ValueError("Experiment 'SplitCIFAR10' cannot have more than 10 tasks!")

            self.classes_per_task = int(np.floor(10 / num_tasks))

            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(self.classes_per_task)) + self.classes_per_task * task_id) for task_id in range(num_tasks)
            ]
            # split them up into sub-tasks
            self.train_datasets = []
            self.test_datasets = []
            for labels in labels_per_task:
                self.train_datasets.append(SubDataset(self.training_set, labels))
                self.test_datasets.append(SubDataset(self.test_set, labels))

        #else:
         #   raise NotImplementedError("dataset_name not implemented (must be SplitCIFAR10)")
    
    def configure_optimizers(self):
        if self.optimizer == "SGDM":
            optimizer =  torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer == "Adam":
            optimizer =  torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("Optimizer not implemented")
        return optimizer

    def train_dataloader(self):

        # Increment the task each time the dataloader is reloaded
        if self.current_epoch % self.epochs_per_task == 0 and self.current_epoch != 0:
            self.current_task += 1
        
        if self.dataset_name == "SplitCIFAR10":

            # Load train sets per task
            current_training_set = self.training_set[self.current_task]

            return torch.utils.data.DataLoader(
                current_training_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

        else:
            return torch.utils.data.DataLoader(
                self.training_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

    def test_dataloader(self):

        # Increment the task each time the dataloader is reloaded
        if self.current_epoch % self.epochs_per_task == 0 and self.current_epoch != 0:
            self.current_task += 1

        if self.dataset_name == "SplitCIFAR10":

            # Load val dataset concat all seen tasks
            current_test_set = self.test_set[:self.current_task]

            return torch.utils.data.DataLoader(
                current_test_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        
        else:
            return torch.utils.data.DataLoader(
                self.validation_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

    # run attacks 
    def on_test_epoch_end(self):
        Attacks.on_test_epoch_end(self)