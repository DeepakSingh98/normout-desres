import numpy as np
import tqdm
import wandb
import argparse
from datetime import datetime

import torchvision.transforms as transforms
import torch
from torchvision.models import vgg16_bn
from robustbench.data import load_cifar10, load_imagenet
from robustbench.utils import clean_accuracy

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

from models.robustbench_model import robustbench_model
from autoattack import AutoAttack



class Benchmarker():
    def __init__(
        self, 
        model_name, 
        dset_name, 
        weights_path, 
        preprocess_means=[0.4914, 0.4822, 0.4465], 
        preprocess_stds = [0.2023, 0.1994, 0.2010], 
        special_name=None,
        device='cuda', 
        batch_size=128,
        log_to_wandb=False,
    ):

        self.model_name = model_name
        self.dataset = dset_name
        self.weights_path = weights_path
        self.preprocess_means = preprocess_means
        self.preprocess_stds = preprocess_stds
        self.device = device
        self.batch_size = batch_size
        self.log_to_wandb = log_to_wandb
    
        # load data
        if self.dataset == "CIFAR10":
            self.test_x, self.test_y = load_cifar10()
            self.n_classes = 10
        elif self.dataset == "ImageNet":
            self.test_x, self.test_y = load_imagenet()
            self.n_classes = 1000
        self.test_x = self.test_x.to(self.device)
        self.test_y = self.test_y.to(self.device)
        self.n_batches = int(np.ceil(self.test_x.shape[0] / self.batch_size))

        # get model
        if model_name == "VGG16" and dset_name == "CIFAR10":
            self.model = vgg16_bn(False, num_classes=self.n_classes)
        elif model_name in [
            "Carmon2019Unlabeled",
            "Standard",
            "Rebuffi2021Fixing_70_16_cutmix_extra"
            ]:
            self.model = robustbench_model(model_name, False)
            self.preprocess_means = None
            self.preprocess_stds = None
        else:
            raise NotImplementedError("model type not implemented")
        
        # load weights
        if weights_path is not None and weights_path != "":            
            loaded = torch.load(weights_path)
            self.model.load_state_dict(loaded['state_dict'])

        # initialize wandb
        if self.log_to_wandb:
            name = special_name if (special_name is not None and special_name != "") else self.model_name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb.init(
                name=(name + "-" + timestamp),
                project="normout",
                entity="normout",
                tags=["benchmark_run"],
                config=args
                )       

    def forward_with_preprocessing(self, x):
        if self.preprocess_means is not None and self.preprocess_stds is not None:
            x = transforms.Normalize(self.preprocess_means, self.preprocess_stds)(x)
        return self.model(x)

    def benchmark(self):
        """
        Evaluates all attacks on the entire validation set
        """

        # set up model
        self.model.to(self.device)
        self.model.eval()

        # report clean accuracy
        clean_acc = clean_accuracy(self.forward_with_preprocessing, self.test_x, self.test_y)
        print(f"Clean accuracy: {clean_acc}")
        if self.log_to_wandb:
            wandb.log({"Clean Accuracy": clean_acc})

        # fgm
        self.fgsm_attack(self.test_x, self.test_y, eps=0.03, norm=np.inf)
        self.fgsm_attack(self.test_x, self.test_y, eps=0.03, norm=2)

        # untargeted pgd
        self.untargeted_pgd_attack(self.test_x, self.test_y, eps=(32/255), eps_iter=0.007, nb_iter=40, norm=np.inf)
        self.untargeted_pgd_attack(self.test_x, self.test_y, eps=(32/255), eps_iter=0.007, nb_iter=40, norm=2)

        # # carlini wagner
        # self.carlini_wagner_l2_attack(self.test_x, self.test_y, self.n_classes, targeted=True)
        # self.carlini_wagner_l2_attack(self.test_x, self.test_y, self.n_classes, targeted=False)

        # targeted pgd
        self.targeted_pgd_attack(self.test_x, self.test_y, eps=0.03, eps_iter=0.007, nb_iter=40, norm=np.inf)
        self.targeted_pgd_attack(self.test_x, self.test_y, eps=0.03, eps_iter=0.007, nb_iter=40, norm=2)

        # square 
        self.square_attack(self.test_x, self.test_y, eps=0.03, norm='Linf')
        self.square_attack(self.test_x, self.test_y, eps=0.03, norm='L2')

    def log_attack_results(self, acc, attack_name):
        if self.log_to_wandb:
            wandb.log(
                {f"{attack_name} Accuracy": acc}
            )
        print(f"{attack_name} Accuracy: {acc}")

    def fgsm_attack(self, x, y, eps, norm=np.inf):
        """
        Performs a fast gradient method attack on the model as described in
        https://arxiv.org/abs/1412.6572
        """
        accs = []
        for i in [0]: # tqdm.tqdm(range(self.n_batches), desc=f"FGSM, norm={norm}"):
            x_batch = x[i * self.batch_size:(i + 1) * self.batch_size]
            y_batch = y[i * self.batch_size:(i + 1) * self.batch_size]
            x_adv = fast_gradient_method(self.forward_with_preprocessing, x_batch, eps, norm=norm)
            y_hat_adv = self.forward_with_preprocessing(x_adv)
            accs.append(
                (y_hat_adv.argmax(dim=1) == y_batch).float().mean().item()
            )
        # print("FGSM accs per batch", [acc.item() for acc in accs])
        self.log_attack_results(sum(accs)/len(accs), f"FGSM, norm={norm}")
        
    def untargeted_pgd_attack(self, x, y, eps, eps_iter, nb_iter, norm=np.inf):
        """ 
        Performs an untargeted projected gradient descent attack on the model as described in
        https://arxiv.org/abs/1706.06083
        """
        accs = []
        # first half is  0.888888909961238, second half is
        for i in [0]: # tqdm.tqdm(range(self.n_batches//2, self.n_batches), desc=f"Untargeted PGD, norm={norm}"):
            x_batch = x[i * self.batch_size:(i + 1) * self.batch_size]
            y_batch = y[i * self.batch_size:(i + 1) * self.batch_size]
            x_adv = projected_gradient_descent(self.forward_with_preprocessing, x_batch, eps, eps_iter, nb_iter, norm=norm)
            y_hat_adv = self.forward_with_preprocessing(x_adv)
            accs.append(
                (y_hat_adv.argmax(dim=1) == y_batch).float().mean().item()
            )
        self.log_attack_results(sum(accs)/len(accs), f"Untargeted PGD, norm={norm}")
    
    def targeted_pgd_attack(self, x, y, eps, eps_iter, nb_iter, norm=np.inf):
        """ 
        Performs a targeted projected gradient descent attack on the model as described in
        https://arxiv.org/abs/1706.06083
        """
        accs = []
        for i in [0]: # tqdm.tqdm(range(self.n_batches), desc=f"Targeted PGD, norm={norm}"):
            x_batch = x[i * self.batch_size:(i + 1) * self.batch_size]
            y_batch = y[i * self.batch_size:(i + 1) * self.batch_size]
            x_adv = projected_gradient_descent(self.forward_with_preprocessing, x_batch, eps, eps_iter, nb_iter, norm, targeted=True, y=(y_batch+1)%self.n_classes)
            y_hat_adv = self.forward_with_preprocessing(x_adv)
            accs.append(
                (y_hat_adv.argmax(dim=1) == y_batch).float().mean().item()
            )
        self.log_attack_results(sum(accs)/len(accs), f"Targeted PGD, norm={norm}")
    
    def carlini_wagner_l2_attack(self, x, y, n_classes, targeted=False):
        """ 
        Performs a Carlini Wagner attack on the model as described in
        https://arxiv.org/abs/1608.04644
        """
        accs = []
        for i in [0]: # tqdm.tqdm(range(self.n_batches), desc=f"Carlini Wagner L2, targeted={targeted}"):
            x_batch = x[i * self.batch_size:(i + 1) * self.batch_size]
            y_batch = y[i * self.batch_size:(i + 1) * self.batch_size]
            x_adv = carlini_wagner_l2(self.forward_with_preprocessing, x_batch, n_classes, targeted=targeted)
            y_hat_adv = self.forward_with_preprocessing(x_adv)
            accs.append(
                (y_hat_adv.argmax(dim=1) == y_batch).float().mean().item()
            )
        self.log_attack_results(sum(accs)/len(accs), f"Carlini Wagner L2, targeted={targeted}")
    
    def square_attack(self, x, y, eps, norm='Linf'):
        """
        Runs a black box square attack with AutoAttack.
        """
        accs = []
        for i in [0]: # tqdm.tqdm(range(self.n_batches), desc=f"Carlini Wagner L2, targeted={targeted}"):
            x_batch = x[i * self.batch_size:(i + 1) * self.batch_size]
            y_batch = y[i * self.batch_size:(i + 1) * self.batch_size]
            adversary = AutoAttack(self.forward_with_preprocessing, norm=norm, eps=eps)
            adversary.attacks_to_run = ['square']
            x_adv = adversary.run_standard_evaluation(x, y)
            y_hat_adv = self.forward_with_preprocessing(x_adv)
            accs.append(
                (y_hat_adv.argmax(dim=1) == y_batch).float().mean().item()
            )
        self.log_attack_results(sum(accs)/len(accs), f"Square")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial benchmarking")
    parser.add_argument("--model-name", type=str, default="VGG16")
    parser.add_argument("--dset-name", type=str, default="CIFAR10", help="dataset name (default CIFAR10, also supports MNIST-Fashion)")
    parser.add_argument("--weights-path", type=str, default="", help="path to model weights")
    parser.add_argument("--device", type=str, default="cuda", help="device (default cuda)")
    parser.add_argument("--batch-size", type=int, default=100, help="batch size (default 128)") # change this if get out of memory error!
    parser.add_argument("--preprocess-means", type=float, nargs="+", default=[0.4914, 0.4822, 0.4465], help="preprocess means (default [0.4914, 0.4822, 0.4465])")
    parser.add_argument("--preprocess-stds", type=float, nargs="+", default=[0.2023, 0.1994, 0.2010], help="preprocess stds (default [0.2023, 0.1994, 0.2010])")
    parser.add_argument("--special-name", type=str, default="", help="special name (default '')")

    args = parser.parse_args()
    benchmarker = Benchmarker(**vars(args))
    benchmarker.benchmark()

    # sample usage
    # python evaluate.py --weights-path /home/ald339/normout/normout/1cse78k1/checkpoints/epoch=99-step=19600.ckpt