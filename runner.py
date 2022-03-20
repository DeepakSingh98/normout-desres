import argparse
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from models.vgg16 import VGG16NormOut

# parse command line inputs
parser = argparse.ArgumentParser()
# basic
parser.add_argument("--epochs", type=int, default=100, help="number of epochs (default 100)")
parser.add_argument("--batch-size", type=int, default=128, help="batch size (default 128)")
parser.add_argument("--num-workers", type=int, default=4, help="number of workers used for data loading (default 4)")
parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus to use (default 1)")
parser.add_argument("--dset-name", type=str, default="CIFAR10", help="dataset name (default CIFAR10, also supports MNIST-Fashion)")
parser.add_argument("--use-cifar-data-augmentation", default=False, action="store_true", help="use data augmentation from CIFAR10 (default False)")
parser.add_argument("--optimizer", type=str, default="SGDM", help="optimizer (default SGDM, also supports Adam)")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default 0.01)")
# model settings
parser.add_argument("--model", type=str, default="VGG16", help="model name (default VGG16)")
parser.add_argument("--dropout-style", type=str, default="NormOut", help="dropout style (default NormOut, supports 'None', 'Dropout', 'NormOut', and 'TopK')")
parser.add_argument("--vgg-no-batch-norm", action="store_true", default=False, help="don't use batch norm (default False)")
# attack params
parser.add_argument("--no-adversarial-fgm", action="store_true", default=False, help="don't use FGM (default False)")
parser.add_argument("--no-adversarial-pgd", action="store_true", default=False, help="don't use PGD (default False)")
parser.add_argument("--no-autoattack", action="store_true", default=False, help="don't use AutoAttack (default False)")
parser.add_argument("--adv-eps", type=float, default=0.03, help="adversarial epsilon (default 0.03)")
parser.add_argument("--pgd-steps", type=int, default=40, help="number of steps for PGD (default 40)")
args = parser.parse_args()

# get model
if args.model == "VGG16":
    model = VGG16NormOut(**vars(args))
else:
    raise NotImplementedError("model not implemented")

# wandb setup
tags = [args.model, args.dropout_style, args.optimizer, args.dset_name]
if args.use_cifar_data_augmentation:
    tags.append("DataAug")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

wandb_logger = WandbLogger(
    project="normout",
    name=(("-").join(tags) + "-" + timestamp),
    tags=tags,
    entity="normout",
    config=args,
)
wandb_logger.watch(model)

# train
trainer = Trainer(gpus=args.num_gpus, logger=wandb_logger, max_epochs=args.epochs)
trainer.fit(model)
