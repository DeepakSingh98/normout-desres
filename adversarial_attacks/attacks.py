from abc import ABC
import numpy as np
from pytorch_lightning.loggers import WandbLogger

from robustbench.eval import benchmark
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from .cw_attacks import carlini_wagner_l2
from robustbench.data import load_cifar10c, load_cifar10
from robustbench.utils import clean_accuracy
import torchvision.transforms as transforms

import torch
import torch.nn.functional as F
import torchvision
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
        no_cw_l2_ce,
        no_cw_l2_t,
        no_fab,
        no_fab_t,
        no_square_attack,
        no_randomized_attack,
        no_robustbench_linf,
        no_robustbench_l2,
        no_salt_and_pepper_attack,
        corruption_types,
        corruption_severity,
        adv_eps,
        pgd_steps,
        all_attacks_off,
        log_adversarial_examples,
        # catch other kwargs
        **kwargs
    ):
        # set attributes
        self.use_adversarial_fgm = not no_fgm
        self.use_adversarial_pgd_ce = not no_pgd_ce
        self.use_adversarial_pgd_t = not no_pgd_t
        self.use_cw_l2_ce = not no_cw_l2_ce
        self.use_cw_l2_t = not no_cw_l2_t
        self.use_fab_attack = not no_fab
        self.use_fab_t_attack = not no_fab_t
        self.use_square_attack = not no_square_attack
        self.use_randomized_attack = not no_randomized_attack
        self.use_robustbench_linf = not no_robustbench_linf
        self.use_robustbench_l2 = not no_robustbench_l2
        self.use_salt_and_pepper_attack = not no_salt_and_pepper_attack
        self.adv_eps = adv_eps
        self.pgd_steps = pgd_steps
        self.corruption_types = corruption_types
        self.corruption_severity = corruption_severity
        self.log_adversarial_examples = log_adversarial_examples

        self.all_attacks_off = all_attacks_off
        
        if self.all_attacks_off:
            self.use_adversarial_fgm = False
            self.use_adversarial_pgd_ce = False
            self.use_adversarial_pgd_t = False
            self.use_cw_l2_ce = False
            self.use_cw_l2_t = False
            self.use_square_attack = False
            self.use_randomized_attack = False
            self.use_robustbench_linf = False
            self.use_robustbench_l2 = False
            self.use_salt_and_pepper_attack = False

    def on_val_epoch_end(self):
        """
        `pytorch_lightning` hook override to insert attacks at end of test epoch.
        """
        if self.current_epoch % 10 == 0 and not self.all_attacks_off:

            if self.dataset_name != 'CIFAR10':
                raise NotImplementedError('Only CIFAR-10 is supported for now.')

            x, y = load_cifar10(n_examples=256)
            x = x.to(self.device); y = y.to(self.device)
            acc = clean_accuracy(self.forward_with_preprocessing, x, y)
            print("Clean Accuracy", acc)
            self.log(f"Clean Accuracy", acc)

            # do ablation to make sure this needs to be here
            torch.set_grad_enabled(True)
            x.requires_grad = True

            if self.use_robustbench_linf:
                print("Evaluating RobustBenchmark Linf...")
                clean_acc, robust_acc = benchmark(self,
                                                  dataset='cifar10',
                                                  threat_model='Linf', 
                                                  eps=8/255,
                                                  n_examples=40,
                                                  device=self.device,
                                                  ) 
                self.log("RobustBench Linf Clean Accuracy", clean_acc)
                self.log("RobustBench Linf Robust Accuracy (Linf 8/255)", robust_acc)
            
            if self.use_robustbench_l2:
                print("Evaluating RobustBenchmark L2...")
                clean_acc, robust_acc = benchmark(self,
                                                  dataset='cifar10',
                                                  threat_model='L2', 
                                                  eps=0.5,
                                                  n_examples=40,
                                                  device=self.device,
                                                  ) 
                self.log("RobustBench L2 Clean Accuracy", clean_acc)
                self.log("RobustBench L2 Robust Accuracy (L2 0.5)", robust_acc)

            if self.corruption_types is not None:
                self.test_corruption()

            if self.use_adversarial_fgm:
                self.fgm_attack(x, y)
                
            if self.use_adversarial_pgd_ce:
                self.untargeted_pgd_attack(x, y)
            
            if self.use_adversarial_pgd_t:
                accuracies = torch.tensor([self.targeted_pgd_attack(x, y, i) for i in range(10)])
                self.log(f"PGD Targeted Accuracy", accuracies)
            
            if self.use_cw_l2_ce and self.current_epoch != 0:
                self.untargeted_cw_l2_attack(x, y)
            
            if self.use_cw_l2_t and self.current_epoch != 0:
                self.targeted_cw_l2_attack(x, y)
            
            if self.use_fab_attack:
                self.untargeted_fab_attack(x, y)
            
            if self.use_fab_t_attack:
                self.targeted_fab_attack(x, y)

            if self.use_square_attack:
                self.square_attack(x, y)
            
            if self.use_randomized_attack:
                self.random_apgd_dlr_attack(x, y)
                self.random_apgd_ce_attack(x, y)

            if self.use_salt_and_pepper_attack:
                self.salt_and_pepper_attack(x, y)
                        
    def test_corruption(self):
        x, y = load_cifar10c(n_examples=256, corruptions=self.corruption_types, severity=self.corruption_severity)
        acc = clean_accuracy(self.forward_with_preprocessing, x, y)
        print(f'Model: {self.model_name}, CIFAR-10-C accuracy: {acc:.1%}')
        self.log(
                    f"Corruption Accuracy (corruption types={self.corruption_types}, severity={self.corruption_severity})", acc
                )

    def log_attack_stats(self, x_adv, y_hat_adv: torch.Tensor, y, attack_name):
        self.logger: WandbLogger
        self.log(
            f"{attack_name} Loss", F.cross_entropy(y_hat_adv.to(torch.float), y)
        )
        self.log(
            f"{attack_name} Accuracy", (y_hat_adv.argmax(dim=1) == y).float().mean()
        )
        print(f"{attack_name} Accuracy", (y_hat_adv.argmax(dim=1) == y).float().mean().item())
        if self.log_adversarial_examples:
            image_grid = torchvision.utils.make_grid(x_adv[:5, :, :, :], nrow=5, normalize=True)
            self.logger.log_image(
                caption=f"{attack_name} Examples, (labels, y_hat): {[(i.item(), j.item()) for i, j in zip(y[:5], y_hat_adv.argmax(dim=1)[:5])]}",
                images=[image_grid],
            )

    def fgm_attack(self, x, y):
        """
        Performs a fast gradient method attack on the model as described in
        https://arxiv.org/abs/1412.6572
        """
        x_adv = fast_gradient_method(self.forward_with_preprocessing, x, self.adv_eps, norm=np.inf)
        y_hat_adv = self.forward_with_preprocessing(x_adv)
        self.log_attack_stats(x_adv, y_hat_adv, y, "FGSM")
        
    def untargeted_pgd_attack(self, x, y):
        """ 
        Performs an untargeted projected gradient descent attack on the model as described in
        https://arxiv.org/abs/1706.06083
        """
        x_adv = projected_gradient_descent(self.forward_with_preprocessing, x, self.adv_eps, 0.007, self.pgd_steps, np.inf)
        y_hat_adv = self.forward_with_preprocessing(x_adv)
        self.log_attack_stats(x_adv, y_hat_adv, y, "Untargeted PGD")
    
    def targeted_pgd_attack(self, x, y, i):
        """ 
        Performs a targeted projected gradient descent attack on the model as described in
        https://arxiv.org/abs/1706.06083
        """
        y_target = torch.full((self.batch_size,), i) # (y + 1) % self.num_classes
        y_target = y_target.to(self.device)
        x_adv = projected_gradient_descent(self.forward_with_preprocessing, x, self.adv_eps, 0.007, self.pgd_steps, np.inf, y=y_target, targeted=True)
        y_hat_adv = self.forward_with_preprocessing(x_adv)
        self.log_attack_stats(x_adv, y_hat_adv, y, f"Targeted PGD i={i}")
        return (y_hat_adv.argmax(dim=1) == y).float().mean()
    
    def untargeted_cw_l2_attack(self, x, y):
        """ 
        Performs an untargeted Carlini-Wagner L2 attack on the model as described in
        https://arxiv.org/pdf/1608.04644.pdf
        """
        x_adv = carlini_wagner_l2(self.forward_with_preprocessing, x, self.num_classes)
        y_hat_adv = self.forward_with_preprocessing(x_adv)
        self.log_attack_stats(x_adv, y_hat_adv, y, "Untargeted CW L2")
    
    def targeted_cw_l2_attack(self, x, y):
        """ 
        Performs a targeted Carlini-Wagner L2 attack on the model as described in
        https://arxiv.org/pdf/1608.04644.pdf
        """
        #y_target = torch.full((self.batch_size,), i) # 
        y_target = (y + 1) % self.num_classes
        y_target = y_target.to(self.device)
        x_adv = carlini_wagner_l2(self.forward_with_preprocessing, x, self.num_classes, y=y_target, targeted=True)
        y_hat_adv = self.forward_with_preprocessing(x_adv)
        self.log_attack_stats(x_adv, y_hat_adv, y, "Targeted CW L2")

    def square_attack(self, x, y):
        """
        Runs a black box square attack with AutoAttack.
        """
        adversary = AutoAttack(self.forward_with_preprocessing, norm='Linf', eps=.3, version='rand')
        adversary.attacks_to_run = ['square']
        x_adv = adversary.run_standard_evaluation(x, y)
        y_hat_adv = self.forward_with_preprocessing(x_adv)
        self.log_attack_stats(x_adv, y_hat_adv, y, "Square")
    
    def random_apgd_ce_attack(self, x, y):
        """
        Runs an EOI APGD-CE attack with AutoAttack (no restarts, 20 iterations for EoT).
        """
        adversary = AutoAttack(self.forward_with_preprocessing, norm='Linf', eps=.3, version='rand')
        adversary.attacks_to_run = ['apgd-ce']
        x_adv = adversary.run_standard_evaluation(x, y)
        y_hat_adv = self.forward_with_preprocessing(x_adv)
        self.log_attack_stats(x_adv, y_hat_adv, y, "EOI APGD-CE")
    
    def random_apgd_dlr_attack(self, x, y):
        """
        Runs an EOI APGD-DLR attack with AutoAttack (no restarts, 20 iterations for EoT).
        """
        adversary = AutoAttack(self.forward_with_preprocessing, norm='Linf', eps=.3, version='rand')
        adversary.attacks_to_run = ['apgd-dlr']
        x_adv = adversary.run_standard_evaluation(x, y)
        y_hat_adv = self.forward_with_preprocessing(x_adv)
        self.log_attack_stats(x_adv, y_hat_adv, y, "EOI APGD-DLR")
    
    def untargeted_fab_attack(self, x, y):
        adversary = AutoAttack(self.forward_with_preprocessing, norm='Linf', eps=self.adv_eps, version='standard')
        adversary.attacks_to_run = ['fab']
        x_adv = adversary.run_standard_evaluation(x, y, self.batch_size)
        y_hat_adv = self.forward_with_preprocessing(x_adv)
        self.log_attack_stats(x_adv, y_hat_adv, y, "Untargeted FAB")
        
    def targeted_fab_attack(self, x, y):
        adversary = AutoAttack(self.forward_with_preprocessing, norm='Linf', eps=self.adv_eps, version='standard')
        adversary.attacks_to_run = ['fab-t']
        x_adv = adversary.run_standard_evaluation(x, y, self.batch_size)
        y_hat_adv = self.forward_with_preprocessing(x_adv)
        self.log_attack_stats(x_adv, y_hat_adv, y, "Targeted FAB")

    def salt_and_pepper_attack(self, x, y):
        """
        Uses Foolbox to perform a salt and pepper attack.
        """
        fmodel = fb.PyTorchModel(self.forward_with_preprocessing, bounds=(0, 1))
        images, labels = fb.utils.samples(fmodel, dataset='cifar10', batchsize=16, data_format='channels_first', bounds=(0, 1))
        clean_acc = fb.accuracy(fmodel, images, labels)
        attack = fb.attacks.saltandpepper.SalßtAndPepperNoiseAttack()
        raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=None)            
        robust_accuracy = 1 - success.sum()/len(success)
        self.log(
            "Salt and Pepper Attack Accuracy", robust_accuracy
        )
        self.log(
            "Salt and Pepper Attack Clean Accuracy", clean_acc
        )
        image_grid = torchvision.utils.make_grid(clipped_advs, nrow=5, normalize=True)
        self.logger.log_image(
            image_grid,
            caption=f"Salt and Pepper Attack, (labels): {labels}",
        )
