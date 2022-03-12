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

class NormOutModel(pl.LightningModule):
    def __init__(
        self, 
        normout_fc1=False, 
        normout_fc2=False, 
        optimizer="SGDM", 
        lr=0.001, 
        batch_size=64, 
        num_workers=4, 
        adversarial_fgm=True, 
        adversarial_pgd=True,
        adv_eps=0.03,
        pgd_steps=40,
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

        # trackers
        self.fc1_neuron_tracker = torch.zeros(self.fc1.out_features).type_as(self.fc1.weight)
        self.fc2_neuron_tracker = torch.zeros(self.fc2.out_features).type_as(self.fc1.weight)

        # dataset
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.training_set = torchvision.datasets.FashionMNIST(
            "./data", train=True, transform=transform, download=True
        )
        self.validation_set = torchvision.datasets.FashionMNIST(
            "./data", train=False, transform=transform, download=True
        )

        # logging
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.save_hyperparameters()

        # adversarial
        self.adversarial_fgm = adversarial_fgm
        self.adversarial_pgd = adversarial_pgd
        self.fgm_acc = torchmetrics.Accuracy()
        self.pgd_acc = torchmetrics.Accuracy()
        self.adv_eps = adv_eps
        self.pgd_steps = pgd_steps

        

    def forward(self, x):
        run_info = dict()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        if self.normout_fc1:
            # divide by biggest value in the activation per input
            norm_x = x / torch.max(x, dim=1, keepdim=True)[0]
            x_mask = torch.rand_like(x) < norm_x
            x = x * x_mask
        run_info["fc1_mask"] = x > 0
        x = F.relu(self.fc2(x))
        if self.normout_fc2:
            # divide by biggest value in the activation per input
            norm_x = x / torch.max(x, dim=1, keepdim=True)[0]
            x_mask = torch.rand_like(x) < norm_x
            x = x * x_mask
        run_info["fc2_mask"] = x > 0
        x = self.fc3(x)
        return x, run_info

    def configure_optimizers(self):
        if self.optimizer == "SGDM":
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == "Adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("Optimizer not implemented")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, run_info = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat, y)
        self.log("Train Accuracy", self.train_acc, on_step=False, on_epoch=True)
        self.log("Train Loss", loss, on_step=True, on_epoch=True)
        self.logger.log_metrics({
            "FC1 Average Percent Activated": run_info["fc1_mask"].sum(dim=0).double().mean(),
            "FC2 Average Percent Activated": run_info["fc2_mask"].sum(dim=0).double().mean(),
            },
        )
        self.fc1_neuron_tracker += run_info["fc1_mask"].sum(dim=0).type_as(self.fc1_neuron_tracker)
        self.fc2_neuron_tracker += run_info["fc2_mask"].sum(dim=0).type_as(self.fc2_neuron_tracker)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.valid_acc(y_hat, y)
        self.log("Validation Accuracy", self.valid_acc, on_step=False, on_epoch=True)
        self.log("Validation Loss", loss, on_step=False, on_epoch=True)

        # adversarial attacks
        if self.adversarial_fgm:
            self.fgm_attack(x, y)
        if self.adversarial_pgd:
            self.pgd_attack(x, y)
        return loss

    def on_train_epoch_start(self) -> None:
        self.fc1_neuron_tracker.zero_()
        self.fc2_neuron_tracker.zero_()
    
    def on_train_epoch_end(self) -> None:
        self.logger.log_metrics({
            "FC1 Dead Neuron Prevalence": (self.fc1_neuron_tracker == 0).sum().double().item() / self.fc1_neuron_tracker.numel(),
            "FC2 Dead Neuron Prevalence": (self.fc2_neuron_tracker == 0).sum().double().item() / self.fc2_neuron_tracker.numel(),
        })

    # dataloaders
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.training_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    # attacks
    def pgd_attack(self, x, y):
        """ 
        Performs a projected gradient descent attack on the model as described in
        https://arxiv.org/abs/1706.06083
        """
        x_adv = projected_gradient_descent(
                self, x, self.adv_eps, norm=np.inf, eps_iter=self.pgd_steps, step_size=0.01, 
            ) 
        y_hat_adv, _ = self(x_adv)
        loss_adv = F.cross_entropy(y_hat_adv, y)
        self.log(f"Adversarial PGD Loss \n(eps={self.adv_eps}, norm=inf, eps_iter={self.pgd_steps}, step_size=0.01)", loss_adv, on_step=False, on_epoch=True)
        self.log("Adversarial PGD Accuracy \n(eps={self.adv_eps}, norm=inf, eps_iter={self.pgd_steps}, step_size=0.01)", self.valid_acc(y_hat_adv, y), on_step=False, on_epoch=True)

    def fgm_attack(self, x, y):
        """
        Performs a fast gradient method attack on the model as described in
        https://arxiv.org/abs/1412.6572
        """
        x_adv = fast_gradient_method(self, x, self.adv_eps, norm=np.inf)
        y_hat_adv, _ = self(x_adv)
        loss_adv = F.cross_entropy(y_hat_adv, y)
        self.log(f"Adversarial FGSM Loss \n(eps={self.adv_eps}, norm=inf)", loss_adv, on_step=False, on_epoch=True)
        self.log(f"Adversarial FGSM Accuracy \n(eps={self.adv_eps}, norm=inf)", self.valid_acc(y_hat_adv, y), on_step=False, on_epoch=True)