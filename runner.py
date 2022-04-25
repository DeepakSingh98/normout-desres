import argparse
from datetime import datetime
from tkinter import FALSE
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from custom_model import CustomModel
from utils import set_tags
import wandb

# parse arguments
# general settings
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs (default 100)")
parser.add_argument("--batch-size", type=int, default=256, help="batch size (default 256)")
parser.add_argument("--num-workers", type=int, default=4, help="number of workers used for data loading (default 4)")
parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus to use (default 1)")
parser.add_argument("--dset-name", type=str, default="CIFAR10", help="dataset name (default CIFAR10, also supports MNIST-Fashion)")
parser.add_argument("--no-data-augmentation", default=False, action="store_true", help="Don't use data augmentation (default False)")
parser.add_argument("--optimizer", type=str, default="SGDM", help="optimizer (default SGDM, also supports Adam)")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default 0.01)")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum value (default 0.9)")
parser.add_argument("--weight-decay", type=float, default=0.0001, help="weight decay value (default 0.0001)")
parser.add_argument("--custom-tag", type=str, default=None, help="custom tag to be added to wandb log")
# model settings
parser.add_argument("--model-name", type=str, default="VGG16", help="model name (default VGG16)")
parser.add_argument("--custom-layer-name", type=str, default=None, help="custom layer (default None, supports 'ReLU', 'NormOut', 'AlwaysDropout', 'DeterministicNormOut', 'NormOutBlock', 'DetNormOutBlock , and 'TopK')")
parser.add_argument("--topk-k", type=int, default=10, help="k value for TopK")
parser.add_argument("--dropout-p", type=float, default=0.5, help="p value for Dropout (probability of neuron being dropped)")
parser.add_argument("--no-batch-norm", default=False, action="store_true", help="Don't use batch norm (default False)")
parser.add_argument("--replace-layers", type=int, nargs="+", default=None, help="layer indices at which the layer is placed with the custom layer (NOTE: happens after removal and insertion)")
parser.add_argument("--remove-layers", type=int, nargs="+", default=None, help="layer indices at which the layer is removed from the model; give vals in ascending order")
parser.add_argument("--insert-layers", type=int, nargs="+", default=None, help="layer indices at which a custom layer is inserted (NOTE: happens after removal)")
# NormOut settings
parser.add_argument("--no-abs", default=False, action="store_true", help="Don't use absolute value of input during NormOut (default False)")
parser.add_argument("--max-type", typer=str, default="spatial", help="Type of max to use in NormOut (default spatial, supports channel, global)")
parser.add_argument("--on-at-inference", default=False, action="store_true", help="Turn NormOut on at inference time (default False)")
parser.add_argument("--normout-delay-epochs", type=int, default=0, help="number of epochs to delay using normout")
# attacks
parser.add_argument("--no-fgm", default=False, action="store_true", help="Don't use adversarial fgm (default False)")
parser.add_argument("--no-pgd", default=False, action="store_true", help="Don't use adversarial pgd (default False)")
parser.add_argument("--no-square-attack", default=False, action="store_true", help="Don't use square attack (default False)")
parser.add_argument("--no-randomized-attack", default=False, action="store_true", help="Don't use randomized attacks (default False)")
parser.add_argument("--no-robustbench", default=False, action="store_true", help="Don't use robustbench autoattack (default False)")
parser.add_argument("no-salt-and-pepper-attack", default=False, action="store_true", help="Don't use salt and pepper attack (default False)")
parser.add_argument("--adv-eps", type=float, default=0.03, help="adversarial epsilon (default 0.03)")
parser.add_argument("--pgd-steps", type=int, default=10, help="number of steps for PGD (default 10)")
args = parser.parse_args()

# get model
model = CustomModel(**vars(args))

# wandb
tags = set_tags(args)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb_logger = WandbLogger(
    project="normout",
    name=(("-").join(tags) + "-" + timestamp),
    tags=tags,
    entity="normout",
    config=args,
    settings=wandb.Settings(start_method="thread")
)
wandb_logger.watch(model)

# train
trainer = Trainer(gpus=args.num_gpus, logger=wandb_logger, max_epochs=args.epochs)
trainer.fit(model)
