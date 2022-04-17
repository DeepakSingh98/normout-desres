from abc import ABC
import datetime
import numpy as np
import os
import wandb

from autoattack import AutoAttack
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
import torchmetrics
import torch
import torch.nn.functional as F

class Attacks(ABC):
    """
    Defines adversarial attacks via PyTorch Lightning's `on_val_epoch_end` hook. Inherited by models.
    """

    def __init__(
        self,
        use_adversarial_fgm,
        use_adversarial_pgd,
        use_autoattack,
        adv_eps,
        pgd_steps,
        # catch other kwargs
        **kwargs
    ):
        # set attributes
        self.use_adversarial_fgm = use_adversarial_fgm
        self.use_adversarial_pgd = use_adversarial_pgd
        self.use_autoattack = use_autoattack
        self.adv_eps = adv_eps
        self.pgd_steps = pgd_steps

        # mkdir logging directory for AutoAttack if needed
        if self.use_autoattack:
            self.log_path = f'./autoattack_logs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
            if not os.path.isdir(self.log_path):
                print(f"creating {self.log_path} directory...")
                os.makedirs(self.log_path)

    def on_validation_epoch_end(self):
        """
        `pytorch_lightning` hook override to insert attacks at end of val epoch.
        """
        if self.current_epoch % 10 == 0:
           
            # get validation data (TODO: currently uses just one batch.)
            batch = next(iter(self.val_dataloader())) 
            x, y = batch
            x = x.to(self.device); y = y.to(self.device)

            # for pgd attack, need to backpropogate to input.
            torch.set_grad_enabled(True)
            x.requires_grad = True

            if self.use_autoattack:
                adversary = AutoAttack(self, norm='Linf',eps=8/255, version='rand', log_path=f'{self.log_path}/{self.logger._name}.txt')
                adversary.run_standard_evaluation(x, y)  
                wandb.save(f'{self.log_path}/{self.logger._name}.txt')

            if self.use_adversarial_fgm:
                loss_adv_fgm, y_hat_adv_fgm, _ = self.fgm_attack(x, y)
                self.log(
                    f"Adversarial FGM Loss \n(eps={self.adv_eps}, norm=inf)",
                    loss_adv_fgm,
                )
                self.log(
                    f"Adversarial FGM Accuracy \n(eps={self.adv_eps}, norm=inf)",
                    (y_hat_adv_fgm.argmax(dim=1) == y).float().mean(),
                )

            if self.use_adversarial_pgd:
                loss_adv_pgd, y_hat_adv_pgd, _ = self.pgd_attack(x, y)
                self.log(
                    f"Adversarial PGD Loss \n(eps={self.adv_eps}, norm=inf, eps_iter={self.pgd_steps}, step_size=0.007)",
                    loss_adv_pgd,
                )
                self.log(
                    f"Adversarial PGD Accuracy \n(eps={self.adv_eps}, norm=inf, eps_iter={self.pgd_steps}, step_size=0.007)",
                    (y_hat_adv_pgd.argmax(dim=1) == y).float().mean(),
                )

    def pgd_attack(self, x, y):
        """ 
        Performs a projected gradient descent attack on the model as described in
        https://arxiv.org/abs/1706.06083
        """
        x_adv = projected_gradient_descent(
            self, x, self.adv_eps, 0.007, self.pgd_steps, np.inf
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