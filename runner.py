import argparse
from model import NormOutModel

from datetime import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

# accept command line arguments for epochs, batch size, number of workers, normout_fc1, normout_fc2
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs (default 100)")
parser.add_argument("--batch-size", type=int, default=128, help="batch size (default 64)")
parser.add_argument("--num-workers", type=int, default=2, help="number of workers used for data loading (default 4)")
parser.add_argument("--normout-fc1", action="store_true", default=False, help="use normout for the fc1 layer (default False)")
parser.add_argument("--normout-fc2", action="store_true", default=False, help="use normout for the fc2 layer (default False)")
parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus to use (default 1)")
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



