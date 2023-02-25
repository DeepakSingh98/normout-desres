import argparse
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from custom_model import CustomModel
from utils import set_tags
import wandb

# parse arguments
parser = argparse.ArgumentParser()
general_params = parser.add_argument_group('General')
general_params.add_argument("--num-workers", type=int, default=4, help="number of workers used for data loading (default 4)")
general_params.add_argument("--num-gpus", type=int, default=1, help="number of gpus to use (default 1)")
general_params.add_argument("--dataset-name", type=str, default="CIFAR10", help="dataset_name (default CIFAR10, also supports MNIST-Fashion, SplitCIFAR10)")
general_params.add_argument("--no-data-augmentation", default=False, action="store_true", help="Don't use data augmentation (default False)")
general_params.add_argument("--custom-tag", type=str, default=None, help="custom tag to be added to wandb log")
general_params.add_argument("--seed", type=int, default=1234, help="Random seed (default 1234)")

hyperparams = parser.add_argument_group('Hyperparameters and Optimiizer')
hyperparams.add_argument("--epochs", type=int, default=100, help="number of epochs (default 100)")
hyperparams.add_argument("--batch-size", type=int, default=256, help="batch size (default 256)")
hyperparams.add_argument("--optimizer", type=str, default="SGDM", help="optimizer (default SGDM, also supports Adam)")
hyperparams.add_argument("--lr", type=float, default=0.01, help="learning rate (default 0.01)")
hyperparams.add_argument("--momentum", type=float, default=0.9, help="momentum value (default 0.9)")
hyperparams.add_argument("--weight-decay", type=float, default=0.0001, help="weight decay value (default 0.0001)")

model_params = parser.add_argument_group('Model Settings')
model_params.add_argument("--model-name", type=str, default="VGG16", help="model name (default VGG16, supports resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2)")
model_params.add_argument("--use-robustbench-data", default=False, action="store_true", help="Use robustbench data")
model_params.add_argument("--get-robustbench-layers", default=False, action="store_true", help="Get robustbench model layers (default False)")
model_params.add_argument("--pretrained", default=False, action="store_true", help="Use pretrained model")
model_params.add_argument("--no-batch-norm", default=False, action="store_true", help="Don't use batch norm (default False)")

custom_layer_params = parser.add_argument_group('Custom Layer Settings')
custom_layer_params.add_argument("--custom-layer-name", type=str, default=None, help="custom layer (default None, supports 'ReLU', 'NormOut', 'Dropout', 'SigmoidOut', and 'TopK')")
custom_layer_params.add_argument("--topk-k", type=int, default=10, help="k value for TopK")
custom_layer_params.add_argument("--dropout-p", type=float, default=0.5, help="p value for Dropout (probability of neuron being dropped)")
custom_layer_params.add_argument("--replace-layers", type=int, nargs="+", default=None, help="layer indices at which the layer is placed with the custom layer (NOTE: happens after removal and insertion)")
custom_layer_params.add_argument("--remove-layers", type=int, nargs="+", default=None, help="layer indices at which the layer is removed from the model; give vals in ascending order")
custom_layer_params.add_argument("--insert-layers", type=int, nargs="+", default=None, help="layer indices at which a custom layer is inserted (NOTE: happens after removal)")

normout_params = parser.add_argument_group('NormOut Settings')
normout_params.add_argument("--no-abs", default=False, action="store_true", help="Don't use absolute value of input during NormOut (default False)")
normout_params.add_argument("--normout-delay-epochs", type=int, default=0, help="number of epochs to delay using normout")
normout_params.add_argument("--normalization-type", type=str, default="SpatiotemporalMax", help="type of normalization to use (default SpatiotemporalMax), supports SpatialMax, TemporalMax, SpatiotemporalMax")
normout_params.add_argument("--temperature", type=int, default=1,help="Temperature to use in NormOut (default 1)")
normout_params.add_argument("--softmax", default=False, action="store_true", help="use softmax in normalization operation")

adv_params = parser.add_argument_group('Adversarial Attack Settings')
adv_params.add_argument("--all-attacks-off", default=False, action="store_true", help="Turn all attacks off (default False)")
adv_params.add_argument("--no-fgm", default=False, action="store_true", help="Don't use adversarial fgm (default False)")
adv_params.add_argument("--no-pgd-ce", default=False, action="store_true", help="Don't use adversarial pgd-ce (default False)")
adv_params.add_argument("--no-pgd-t", default=False, action="store_true", help="Don't use adversarial pgd-t (default False)")
adv_params.add_argument("--no-cw-l2-ce", default=True, action="store_true", help="Don't use untargeted Carlini wagner L2 attacks (default False)")
adv_params.add_argument("--no-cw-l2-t", default=True, action="store_true", help="Don't use targeted Carlini wagner L2 attacks (default False)")
adv_params.add_argument("--no-fab",  default=True, action="store_true", help="Don't use Untargeted FAB attack (default False)")
adv_params.add_argument("--no-fab-t",  default=True, action="store_true", help="Don't use FAB-T attack (default False)")
adv_params.add_argument("--no-square-attack", default=True, action="store_true", help="Don't use square attack (default False)")
adv_params.add_argument("--no-randomized-attack", default=True, action="store_true", help="Don't use randomized attacks (default False)") #TODO make False default
adv_params.add_argument("--no-robustbench-linf", default=True, action="store_true", help="Don't use robustbench Linf autoattack (default False)")
adv_params.add_argument("--no-robustbench-l2", default=True, action="store_true", help="Don't use robustbench L2 autoattack (default False)")
adv_params.add_argument("--no-salt-and-pepper-attack", default=True, action="store_true", help="Don't use salt and pepper attack (default False)")
adv_params.add_argument("--adv-eps", type=float, default=0.03, help="adversarial epsilon (default 0.03)")
adv_params.add_argument("--pgd-steps", type=int, default=10, help="number of steps for PGD (default 10)")
adv_params.add_argument("--corruption-types", type=str, nargs="+", default=None, help="type of corruption to add (supports shot_noise, motion_blur, snow, pixelate, gaussian_noise, defocus_blur, brightness, fog, zoom_blur, frost, glass_blur, impulse_noise, contrast, jpeg_compression, elastic_transform")
adv_params.add_argument("--corruption-severity", type=int, default=1, help="Severity of corruption, supports ints 1 through 5 (default 1)")
adv_params.add_argument("--log-adversarial-examples", default=False, action="store_true", help="Log adversarial examples (default False)")

logging_params = parser.add_argument_group('Logging Settings')
logging_params.add_argument("--no-log-sparsity", default=False, action="store_true", help="Don't log sparsity (default False")
logging_params.add_argument("--no-log-stats", default=False, action="store_true", help="Log input stats (default False)")

continual_params = parser.add_argument_group('Continual Learning Settings')
continual_params.add_argument("--num_tasks", type=int, default=5, help="Number of tasks for continual learning (default 5)")
continual_params.add_argument("--ewc_lambda", type=int, default=0, help="hyperparam: how strong to weigh EWC-loss ('regularisation strength')")
continual_params.add_argument("--gamma", type=float, default=1., help="hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term")
continual_params.add_argument("--online", type=bool, default=True, help="'online' (=single quadratic term) or 'offline' (=quadratic term per task) EWC")
continual_params.add_argument("--fisher_n", type=int, default=None, help="sample size for estimating FI-matrix (if 'None', full pass over dataset)")
continual_params.add_argument("--emp-FI", type=bool, default=False, help="if True, use provided labels to calculate FI ('empirical FI'); else predicted labels")

args = parser.parse_args()

# wandb
tags = set_tags(args)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb_logger = WandbLogger(
    project="normout-continual",
    name=(("-").join(tags) + "-" + timestamp),
    tags=tags,
    entity="deepaksingh",
    config=args,
    settings=wandb.Settings(start_method="thread")
)
config = wandb.config

# update args with wandb config
for k, v in config.items():
    if k in args:
        args.__setattr__(k, v)

# get model
model = CustomModel(**vars(args))

wandb_logger.watch(model)

# train
if args.dataset_name == "SplitCIFAR10":
    trainer = Trainer(gpus=args.num_gpus, logger=wandb_logger, max_epochs=args.epochs, reload_dataloaders_every_n_epochs=model.epochs_per_task, num_sanity_val_steps=0)
else:
    trainer = Trainer(gpus=args.num_gpus, logger=wandb_logger, max_epochs=args.epochs)

trainer.fit(model)