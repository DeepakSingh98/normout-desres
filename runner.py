import argparse
from model import NormOutModel

from datetime import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

import wandb

wandb.init(entity="normout")

# accept command line arguments for epochs, batch size, number of workers, normout_fc1, normout_fc2
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs (default 100)")
parser.add_argument("--batch-size", type=int, default=128, help="batch size (default 64)")
parser.add_argument("--num-workers", type=int, default=4, help="number of workers used for data loading (default 4)")
parser.add_argument("--normout-fc1", action="store_true", default=False, help="use normout for the fc1 layer (default False)")
parser.add_argument("--normout-fc2", action="store_true", default=False, help="use normout for the fc2 layer (default False)")
parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus to use (default 1)")
parser.add_argument("--adversarial-fgm", action="store_true", default=True, help="run fgm attack (default False)")
parser.add_argument("--adversarial-pgd", action="store_true", default=True, help="run pgd attack (default False)")
parser.add_argument("--adv-eps", type=float, default=0.1, help="epsilon for fgm and pgd attacks (default 0.03)")
parser.add_argument("--pgd-steps", type=int, default=40, help="pgd steps (default 40)")
parser.add_argument("--optimizer", type=str, default="SGDM", help="optimizer (default SGDM)")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default 0.001)")
parser.add_argument("--dset-name", type=str, default="MNIST-Fashion", help="dataset name (default MNIST-Fashion, also supports CIFAR10)")
parser.add_argument("--normout-delay-epochs", type=int, default=0, help="normout delay epochs (default 0)")
args = parser.parse_args()

# get model
model = NormOutModel(**vars(args))

# initialize model name/logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tags = []
if model.normout_fc1:
    tags.append("normout_fc1")
if model.normout_fc2:
    tags.append("normout_fc2")

wandb_logger = WandbLogger(
    project="normout",
    name=(("-").join(tags) + "-" + timestamp) if len(tags) > 0 else f"baseline-{timestamp}",
    tags=tags,
)

wandb_logger.watch(model)

trainer = Trainer(gpus=args.num_gpus, logger=wandb_logger, max_epochs=args.epochs)

trainer.fit(model)

