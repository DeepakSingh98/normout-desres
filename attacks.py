from abc import ABC
import torchmetrics
import torch
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
import wandb
import numpy as np
import torch.nn.functional as F

class Attacks(ABC):
    """
    Handles adversarial attacks via on_val_epoch_end.
    """

    def __init__(
        self,
        # attacks
        adversarial_fgm=True,
        adversarial_pgd=True,
        adv_eps=0.03,
        pgd_steps=40,
        **kwargs
    ):
        # adversarial
        self.adversarial_fgm = adversarial_fgm
        self.adversarial_pgd = adversarial_pgd
        self.fgm_acc = torchmetrics.Accuracy()
        self.pgd_acc = torchmetrics.Accuracy()
        self.adv_eps = adv_eps
        self.pgd_steps = pgd_steps

    # attacks at end of val epoch
    def on_val_epoch_end(self):
        # TODO this is bad, should use full dset.
        x, y = next(iter(self.val_dataloader()))
        torch.set_grad_enabled(True)
        x.requires_grad = True
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
                self.fgm_acc(y_hat_adv_fgm, y),
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
                self.pgd_acc(y_hat_adv_pgd, y),
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


# scrap code TODO where do I put this?
# self.save_hyperparameters()
# self.run_info = dict()