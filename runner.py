import argparse
from model import *

from datetime import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

# accept command line arguments for epochs, batch size, number of workers, normout_fc1, normout_fc2
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs (default 100)")
parser.add_argument("--batch-size", type=int, default=128, help="batch size (default 64)")
parser.add_argument("--num-workers", type=int, default=4, help="number of workers used for data loading (default 4)")
parser.add_argument("--abs-normout-fc1", action="store_true", default=False, help="use abs-normout for the fc1 layer (default False)")
parser.add_argument("--abs-normout-fc2", action="store_true", default=False, help="use abs-normout for the fc2 layer (default False)")
parser.add_argument("--exp-normout-fc1", action="store_true", default=False, help="use abs-normout for the fc1 layer (default False)")
parser.add_argument("--exp-normout-fc2", action="store_true", default=False, help="use abs-normout for the fc2 layer (default False)")
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
## topk baseline
parser.add_argument("--topk-baseline", action="store_true", default=False, help="use topk baseline instead of normout (default False)")
parser.add_argument("--topk-k", type=int, default=10, help="topk k (default 10)")
parser.add_argument("--topk-fc1", action="store_true", default=False, help="use topk for the fc1 layer (default True)")
parser.add_argument("--topk-fc2", action="store_true", default=False, help="use topk for the fc2 layer (default True)")
#dropout
parser.add_argument("--dropout-baseline", action="store_true", default=False, help="use dropout baseline instead of normout (default False)")
parser.add_argument("--dropout-p", type=float, default=0.5, help="dropout p (default 0.5)")
parser.add_argument("--dropout-fc1", action="store_true", default=False, help="use dropout after the fc1 layer (default True)")
parser.add_argument("--dropout-fc2", action="store_true", default=False, help="use dropout after the fc2 layer (default True)")
args = parser.parse_args()

# get model
if args.topk_baseline:
    print("Using topk baseline")
    assert args.topk_fc1 or args.topk_fc2
    model = NormOutTopK(**vars(args))

if args.dropout_baseline:
    print("Using dropout baseline")
    assert args.dropout_fc1 or args.dropout_fc2
    model = DropoutModel(**vars(args))

else:
    model = NormOutModel(**vars(args))

# initialize model name/logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tags = []
if model.exp_normout_fc1:
    tags.append("exp_normout_fc1")
if model.exp_normout_fc2:
    tags.append("exp_normout_fc2")
if model.abs_normout_fc1:
    tags.append("abs_normout_fc1")
if model.abs_normout_fc2:
    tags.append("abs_normout_fc2")
if model.normout_fc1:
    tags.append("normout_fc1")
if model.normout_fc2:
    tags.append("normout_fc2")
if model.normout_delay_epochs > 0 and (model.normout_fc1 or model.normout_fc2):
    tags.append("normout_delay_epochs_%d" % model.normout_delay_epochs)
if args.topk_baseline:
    tags.append("k=%d" % model.topk_k)
    if args.topk_fc1:
        tags.append("topk_fc1")
    if args.topk_fc2:
        tags.append("topk_fc2")
if args.dropout_baseline:
    #tags.append("p=%d", model.dropout_p)
    if args.dropout_fc1:
        tags.append("dropout_fc1")
    if args.dropout_fc2:
        tags.append("dropout_fc2")
if not model.normout_fc1 and not model.normout_fc2:
    tags.append("baseline")
tags.append(model.dset_name)
tags.append(model.optimizer)

wandb_logger = WandbLogger(
    project="normout",
    name=(("-").join(tags) + "-" + timestamp),
    tags=tags,
    entity="normout"
)

wandb_logger.watch(model)

trainer = Trainer(gpus=args.num_gpus, logger=wandb_logger, max_epochs=args.epochs)

trainer.fit(model)

