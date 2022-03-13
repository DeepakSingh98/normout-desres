from importlib_metadata import requires
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import torchvision
import torchvision.transforms as transforms
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
import numpy as np
import wandb


class NormOutModel(pl.LightningModule):
    def __init__(
        self,
        normout_fc1=False,
        normout_fc2=False,
        optimizer="SGDM",
        lr=0.01,
        batch_size=64,
        num_workers=4,
        adversarial_fgm=True,
        adversarial_pgd=True,
        adv_eps=0.03,
        pgd_steps=40,
        normout_delay_epochs=0,
        dset_name="MNIST-Fashion",  # 'MNIST-Fashion' or 'CIFAR10'
        **kwargs,
    ):
        super(NormOutModel, self).__init__()

        # model
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # settings
        self.normout_fc1 = normout_fc1
        self.normout_fc2 = normout_fc2
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normout_delay_epochs = normout_delay_epochs
        self.dset_name = dset_name

        # trackers
        self.fc1_neuron_tracker = torch.zeros(
            self.fc1.out_features, requires_grad=False
        ).type_as(self.fc1.weight)
        self.fc2_neuron_tracker = torch.zeros(
            self.fc2.out_features, requires_grad=False
        ).type_as(self.fc1.weight)

        # dataset
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
        else:
            raise NotImplementedError("Dataset not implemented")

        # logging
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.save_hyperparameters()
        self.run_info = dict()

        # adversarial
        self.adversarial_fgm = adversarial_fgm
        self.adversarial_pgd = adversarial_pgd
        self.fgm_acc = torchmetrics.Accuracy()
        self.pgd_acc = torchmetrics.Accuracy()
        self.adv_eps = adv_eps
        self.pgd_steps = pgd_steps

    def forward(self, x):
        self.run_info = dict()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        if self.normout_fc1 and self.current_epoch >= self.normout_delay_epochs:
            # divide by biggest value in the activation per input
            norm_x = torch.tensor(x / torch.max(x, dim=1, keepdim=True)[0])
            x_mask = torch.tensor(torch.rand_like(x) < norm_x)
            self.run_info["x_mask"] = x_mask
            x = x * x_mask
        self.run_info["fc1_mask"] = x > 0
        x = F.relu(self.fc2(x))
        if self.normout_fc2 and self.current_epoch >= self.normout_delay_epochs:
            # divide by biggest value in the activation per input
            norm_x = x / torch.max(x, dim=1, keepdim=True)[0]
            x_mask = torch.rand_like(x) < norm_x
            x = x * x_mask
        self.run_info["fc2_mask"] = x > 0
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        if self.optimizer == "SGDM":
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == "Adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("Optimizer not implemented")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat, y)
        self.log("Train Accuracy", self.train_acc, on_step=False, on_epoch=True)
        self.log("Train Loss", loss, on_step=True, on_epoch=True)
        self.logger.log_metrics(
            {
                "FC1 Average Percent Activated": self.run_info["fc1_mask"]
                .sum(dim=0)
                .double()
                .mean(),
                "FC2 Average Percent Activated": self.run_info["fc2_mask"]
                .sum(dim=0)
                .double()
                .mean(),
            },
        )

        self.fc1_neuron_tracker += (
            self.run_info["fc1_mask"].sum(dim=0).type_as(self.fc1_neuron_tracker)
        )
        self.fc2_neuron_tracker += (
            self.run_info["fc2_mask"].sum(dim=0).type_as(self.fc2_neuron_tracker)
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        torch.set_grad_enabled(True)
        x.requires_grad = True
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.valid_acc(y_hat, y)
        self.log("Validation Accuracy", self.valid_acc, on_step=False, on_epoch=True)
        self.log("Validation Loss", loss, on_step=False, on_epoch=True)

        # adversarial attack, only do after the first epoch
        if self.current_epoch > 0:
            if self.adversarial_fgm:
                loss_adv_fgm, y_hat_adv_fgm, x_adv = self.fgm_attack(x, y)
                self.log(
                    f"Adversarial FGM Loss \n(eps={self.adv_eps}, norm=inf)",
                    loss_adv_fgm,
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    f"Adversarial FGM Accuracy \n(eps={self.adv_eps}, norm=inf)",
                    self.valid_acc(y_hat_adv_fgm, y),
                    on_step=False,
                    on_epoch=True,
                )
                if self.current_epoch % 50 == 1:
                    # show two examples
                    self.logger.log_metrics(
                        {
                            "FGM Adversarial Examples": wandb.Image(
                                np.concatenate(
                                    [
                                        x_adv[0].cpu().detach().numpy(),
                                        x_adv[1].cpu().detach().numpy(),
                                    ],
                                    axis=1,
                                )
                            )
                        }
                    )
            if self.adversarial_pgd:
                loss_adv_pgd, y_hat_adv_pgd, x_adv = self.pgd_attack(x, y)
                self.log(
                    f"Adversarial PGD Loss \n(eps={self.adv_eps}, norm=inf, eps_iter={self.pgd_steps}, step_size=0.01)",
                    loss_adv_pgd,
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    f"Adversarial PGD Accuracy \n(eps={self.adv_eps}, norm=inf, eps_iter={self.pgd_steps}, step_size=0.01)",
                    self.valid_acc(y_hat_adv_pgd, y),
                    on_step=False,
                    on_epoch=True,
                )
                if self.current_epoch % 50 == 1:
                    self.logger.log_metrics(
                        {
                            "PGD Adversarial Example": wandb.Image(
                                np.concatenate(
                                    [
                                        x_adv[0].cpu().detach().numpy(),
                                        x_adv[1].cpu().detach().numpy(),
                                    ],
                                    axis=1,
                                )
                            )
                        }
                    )
        return loss

    def on_train_epoch_start(self) -> None:
        self.fc1_neuron_tracker.zero_()
        self.fc2_neuron_tracker.zero_()

    def on_train_epoch_end(self) -> None:
        self.logger.log_metrics(
            {
                "FC1 Dead Neuron Prevalence": (self.fc1_neuron_tracker == 0)
                .sum()
                .double()
                .item()
                / self.fc1_neuron_tracker.numel(),
                "FC2 Dead Neuron Prevalence": (self.fc2_neuron_tracker == 0)
                .sum()
                .double()
                .item()
                / self.fc2_neuron_tracker.numel(),
            }
        )

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

    # attacks
    def pgd_attack(self, x, y):
        """ 
        Performs a projected gradient descent attack on the model as described in
        https://arxiv.org/abs/1706.06083
        """
        x_adv = projected_gradient_descent(
            self, x, self.adv_eps, 0.01, self.pgd_steps, np.inf
        )
        y_hat_adv = self(x_adv)
        loss_adv = F.cross_entropy(y_hat_adv, y)
        return loss_adv, y_hat_adv, x_adv

    def fgm_attack(self, x, y):
        """
        Performs a fast gradient method attack on the model as described in
        https://arxiv.org/abs/1412.6572
        """
        x_adv = fast_gradient_method(self, x, self.adv_eps, norm=np.inf)
        y_hat_adv = self(x_adv)
        loss_adv = F.cross_entropy(y_hat_adv, y)
        return loss_adv, y_hat_adv, x_adv


class NormOutTopK(NormOutModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.run_info = dict()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        # top 10 mask
        x = x * torch.topk(x, 10, dim=1)[0][:, :, None]
        self.run_info["fc1_mask"] = x > 0
        x = F.relu(self.fc2(x))
        x = x * torch.topk(x, 10, dim=1)[0][:, :, None]
        self.run_info["fc2_mask"] = x > 0
        x = self.fc3(x)
        return x
