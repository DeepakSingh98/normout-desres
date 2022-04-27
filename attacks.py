from abc import ABC
from pickle import FALSE
import numpy as np
from PIL import Image

from robustbench.eval import benchmark
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from robustbench.data import load_cifar10c
from robustbench.utils import clean_accuracy

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from autoattack import AutoAttack
import foolbox as fb

class Attacks(ABC):
    """
    Defines adversarial attacks via PyTorch Lightning's `on_val_epoch_end` hook. Inherited by models.
    """

    def __init__(
        self,
        no_fgm,
        no_pgd_ce,
        no_pgd_t,
        no_square_attack,
        no_randomized_attack,
        no_robustbench,
        no_salt_and_pepper_attack,
        corruption_types,
        corruption_severity,
        adv_eps,
        pgd_steps,
        all_attacks_off,
        # catch other kwargs
        **kwargs
    ):
        # set attributes
        self.use_adversarial_fgm = not no_fgm
        self.use_adversarial_pgd_ce = not no_pgd_ce
        self.use_adversarial_pgd_t = not no_pgd_t
        self.use_square_attack = not no_square_attack
        self.use_randomized_attack = not no_randomized_attack
        self.use_robustbench = not no_robustbench
        self.use_salt_and_pepper_attack = not no_salt_and_pepper_attack
        self.adv_eps = adv_eps
        self.pgd_steps = pgd_steps
        self.corruption_types = corruption_types
        self.corruption_severity = corruption_severity

        if all_attacks_off:
            self.use_adversarial_fgm = False
            self.use_adversarial_pgd_ce = False
            self.use_adversarial_pgd_t = False
            self.use_square_attack = False
            self.use_randomized_attack = False
            self.use_robustbench = False
            self.use_salt_and_pepper_attack = False


    def on_validation_epoch_end(self):
        """
        `pytorch_lightning` hook override to insert attacks at end of val epoch.
        """
        if self.current_epoch % 10 == 0:

            self.set_preprocess_during_forward(True)

            if self.corruption_types is not None:
                x, y = load_cifar10c(n_examples=256, corruptions=self.corruption_types, severity=self.corruption_severity)
            
            else:
                transform_list = [transforms.ToTensor()]
                transform_chain = transforms.Compose(transform_list)
                item = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform_chain, download=True)
                test_loader = torch.utils.data.DataLoader(item, batch_size=256, shuffle=False, num_workers=self.num_workers)
                
                # TODO: currently uses just one batch.
                x, y = next(iter(test_loader))

            x = x.to(self.device); y = y.to(self.device)

            # for pgd attack, need to backpropogate to input.
            torch.set_grad_enabled(True)
            x.requires_grad = True

            if self.corruption_types is not None:
                acc = clean_accuracy(self, x, y)
                print(f'Model: {self.model_name}, CIFAR-10-C accuracy: {acc:.1%}')
                self.log(
                    f"Corruption Accuracy with corruptions={self.corruption_types}, severity={self.corruption_severity}: ", acc
                )

            if self.use_robustbench:
                print("Evaluating RobustBenchmark...")
                clean_acc, robust_acc = benchmark(self,
                                                  dataset='cifar10',
                                                  threat_model='Linf', 
                                                  eps=8/255,
                                                  n_examples=40,
                                                  device=self.device,
                                                  ) # TODO: do more than just 40 examples during final evaluation. Is .copy() sketchy?
                print("RobustBench Clean Accuracy: ", clean_acc)
                print("RobustBench Robust Accuracy: ", robust_acc)
                self.log(
                    "RobustBench Clean Accuracy", clean_acc,
                )
                self.log(
                    "RobustBench Robust Accuracy (Linf 8/255)", robust_acc,
                )

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
                print("Adversarial FGM Accuracy: ", (y_hat_adv_fgm.argmax(dim=1) == y).float().mean())

            if self.use_adversarial_pgd_ce:
                loss_adv_pgd, y_hat_adv_pgd, _ = self.untargeted_pgd_attack(x, y)
                self.log(
                    f"Adversarial PGD-CE Loss \n(eps={self.adv_eps}, norm=inf, eps_iter={self.pgd_steps}, step_size=0.007)",
                    loss_adv_pgd,
                )
                self.log(
                    f"Adversarial PGD-CE Accuracy \n(eps={self.adv_eps}, norm=inf, eps_iter={self.pgd_steps}, step_size=0.007)",
                    (y_hat_adv_pgd.argmax(dim=1) == y).float().mean(),
                )
                print("Adversarial PGD-CE Accuracy: ", (y_hat_adv_pgd.argmax(dim=1) == y).float().mean())
            
            if self.use_adversarial_pgd_t:
                loss_adv_pgd, y_hat_adv_pgd, _ = self.targeted_pgd_attack(x, y)
                self.log(
                    f"Adversarial PGD-T Loss \n(eps={self.adv_eps}, norm=inf, eps_iter={self.pgd_steps}, step_size=0.007)",
                    loss_adv_pgd,
                )
                self.log(
                    f"Adversarial PGD-T Accuracy \n(eps={self.adv_eps}, norm=inf, eps_iter={self.pgd_steps}, step_size=0.007)",
                    (y_hat_adv_pgd.argmax(dim=1) == y).float().mean(),
                )
                print("Adversarial PGD-T Accuracy: ", (y_hat_adv_pgd.argmax(dim=1) == y).float().mean())

            if self.use_square_attack:
                y_hat_adv_square = self.square_attack(x, y)
                self.log(
                    f"Adversarial Square Accuracy \n(eps=.3, norm=inf)",
                    (y_hat_adv_square == y).float().mean(),
                )
                print("Adversarial Square Accuracy: ", (y_hat_adv_square == y).float().mean())  
            
            if self.use_randomized_attack:
                y_hat_ce, y_hat_dlr = self.randomized_attacks(x, y)
                self.log(
                    f"Randomized CE Accuracy \n(eps=.3, norm=inf)",
                    (y_hat_ce == y).float().mean(),
                )
                self.log(
                    f"Randomized DLR Accuracy \n(eps=.3, norm=inf)",
                    (y_hat_dlr == y).float().mean(),
                )
                print("Randomized CE Accuracy: ", (y_hat_ce == y).float().mean())
                print("Randomized DLR Accuracy: ", (y_hat_dlr == y).float().mean())

            if self.use_salt_and_pepper_attack:
                robust_accuracy = self.salt_and_pepper_attack(x, y)
                self.log(
                    f"Adversarial Salt and Pepper Accuracy \n(eps=.3)",
                    robust_accuracy,
                )
                print("Adversarial Salt and Pepper Accuracy: ", robust_accuracy)
            
            self.set_preprocess_during_forward(False)


    def untargeted_pgd_attack(self, x, y):
        """ 
        Performs an untargeted projected gradient descent attack on the model as described in
        https://arxiv.org/abs/1706.06083
        """
        x_adv = projected_gradient_descent(
            self, x, self.adv_eps, 0.007, self.pgd_steps, np.inf
        )
        y_hat_adv = self(x_adv)
        loss_adv = F.cross_entropy(y_hat_adv, y)
        return loss_adv, y_hat_adv, x_adv
    
    def targeted_pgd_attack(self, x, y):
        """ 
        Performs a targeted projected gradient descent attack on the model as described in
        https://arxiv.org/abs/1706.06083
        """
        y_target = (y + 1) % self.num_classes
        x_adv = projected_gradient_descent(
            self, x, self.adv_eps, 0.007, self.pgd_steps, np.inf, y=y_target, targeted=True
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

    def square_attack(self, x, y):
        """
        Runs a black box square attack with AutoAttack.
        """
        adversary = AutoAttack(self, norm='Linf', eps=.3, version='rand')
        adversary.attacks_to_run = ['square']
        _, y_hat_adv = adversary.run_standard_evaluation(x, y, return_labels=True)
        return y_hat_adv
    
    def randomized_attacks(self, x, y):
        """
        Runs APGD-CE (no restarts, 20 iterations for EoT) and APGD-DLR (no restarts, 20 iterations for EoT),
        designed especially for randomized attacks.
        """
        adversary = AutoAttack(self, norm='Linf', eps=.3, version='rand')
        adversary.attacks_to_run = ['apgd-ce']
        _, y_hat_adv_ce = adversary.run_standard_evaluation(x, y, return_labels=True)
        adversary.attacks_to_run = ['apgd-dlr']
        _, y_hat_adv_dlr = adversary.run_standard_evaluation(x, y, return_labels=True)
        return y_hat_adv_ce, y_hat_adv_dlr

    def salt_and_pepper_attack(self, x, y):
        """
        Uses Foolbox to perform a salt and pepper attack.
        """
        # get images and don't use dataloaders
        if self.dset_name == "CIFAR10":
            preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
            self.set_preprocess_during_forward(False)
            fmodel = fb.PyTorchModel(self, bounds=(0, 1), preprocessing=preprocessing)
            images, labels = fb.utils.samples(fmodel, dataset='cifar10', batchsize=16, data_format='channels_first', bounds=(0, 1))
            clean_acc = fb.accuracy(fmodel, images, labels)
            attack = fb.attacks.saltandpepper.SaltAndPepperNoiseAttack()
            raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=None)            
            robust_accuracy = 1 - success.sum()/len(success)
            self.set_preprocess_during_forward(True)
            return robust_accuracy
        else:
            raise NotImplementedError("Salt and pepper attack not implemented for this dataset.")
