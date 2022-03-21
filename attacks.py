from abc import ABC
import datetime
import torchmetrics
import torch
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
import wandb
import numpy as np
import torch.nn.functional as F
from autoattack import AutoAttack

class Attacks(ABC):
    """
    Handles adversarial attacks via on_val_epoch_end.
    """

    def __init__(
        self,
        # attacks
        no_adversarial_fgm=True,
        no_adversarial_pgd=True,
        no_autoattack=True,
        adv_eps=0.03,
        pgd_steps=40,
        **kwargs
    ):
        # adversarial
        self.adversarial_fgm = not no_adversarial_fgm
        self.adversarial_pgd = not no_adversarial_pgd
        self.autoattack = not no_autoattack
        self.fgm_acc = torchmetrics.Accuracy()
        self.pgd_acc = torchmetrics.Accuracy()
        self.adv_eps = adv_eps
        self.pgd_steps = pgd_steps

    # attacks at end of val epoch
    def on_validation_epoch_end(self):
        if self.current_epoch % 10 == 0:
           # import ipdb; ipdb.set_trace()
           # print("\nAttacking!\n\n")

            # get entire validation set
            l = [x for (x, y) in self.val_dataloader()]
            x = torch.cat(l, 0)
            l = [y for (x, y) in self.val_dataloader()]
            y = torch.cat(l, 0)
            x = x.to(self.device); y = y.to(self.device)

            # for gradient tracking of image to backprop through it
            # torch.set_grad_enabled(True) # TODO need this?
            x.requires_grad = True

            if self.autoattack:
                adversary = AutoAttack(self, norm='Linf',eps=8/255, version='rand', log_path=f'./autoattack_logs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/{self.logger._name}.txt')
                x_adv = adversary.run_standard_evaluation(x, y)  

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
                    f"Adversarial PGD Loss \n(eps={self.adv_eps}, norm=inf, eps_iter={self.pgd_steps}, step_size=0.007)",
                    loss_adv_pgd,
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    f"Adversarial PGD Accuracy \n(eps={self.adv_eps}, norm=inf, eps_iter={self.pgd_steps}, step_size=0.007)",
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