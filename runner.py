import argparse
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from models.vgg16 import VGG16NormOut

# parse command line inputs
parser = argparse.ArgumentParser()
# basic
parser.add_argument("--epochs", type=int, default=200, help="number of epochs (default 100)")
parser.add_argument("--batch-size", type=int, default=128, help="batch size (default 64)")
parser.add_argument("--num-workers", type=int, default=4, help="number of workers used for data loading (default 4)")
parser.add_argument("--dset-name", type=str, default="MNIST-Fashion", help="dataset name (default MNIST-Fashion, also supports CIFAR10)")
parser.add_argument("--optimizer", type=str, default="SGDM", help="optimizer (default SGDM)")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default 0.01)")
# model settings
parser.add_argument("--model", type=str, default="VGG16", help="model name (default VGG16)")
parser.add_argument("--dropout-style", type=str, default="NormOut", help="dropout style (default NormOut, supports 'None', 'Dropout', 'NormOut', and 'TopK')")
parser.add_argument("--vgg-no-batch-norm", action="store_true", default=False, help="don't use batch norm (default False)")
args = parser.parse_args()

# get model
if args.model == "VGG16":
    model = VGG16NormOut(**vars(args))
else:
    raise NotImplementedError("model not implemented")

# wandb setup
tags = [parser.model, parser.dropout_style, parser.optimizer, parser.dset_name]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

wandb_logger = WandbLogger(
    project="normout",
    name=(("-").join(tags) + "-" + timestamp),
    tags=tags,
    entity="normout"
)
wandb_logger.watch(model)

# train
trainer = Trainer(gpus=args.num_gpus, logger=wandb_logger, max_epochs=args.epochs)
trainer.fit(model)
