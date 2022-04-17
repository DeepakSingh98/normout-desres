# NormOut: Divisive Normalization as Dropout Probabilities
*Deepak Singh & Xander Davies*

## Files 

- `attacks.py` is where the class Attacks uses the `on_validation_epoch_end` method to run our adversarial attacks used to evaluate robustness.
- `base_model.py` is an abstract base class for a basic `pl.LightningModule` with `Attacks` inherited (called `BasicLightningModel`). It configures the optimizer and dataloaders, but requires that the `forward` method be defined by a child.
- `editable_model.py` inherits `BasicLightningModel` and defines a model architecture specified by `model-name` where specified layers can be removed or replaced with `custom-layer-name` layers, and `custom-layer-name` layers can be inserted at specified indices.
- `custom_layers/` stores custom layers (`nn.Module`) like `NormOut`.
- `models/` stores functions which return a `nn.Module` list corresponding to specified model architectures. These functions correspond to `model-name` specified in `EditableModel`.
- `runner.py` is the runner file for training. Use `python runner.py -h` to see runner specifications. 

## Usage 

### Example Usage
- To run a baseline VGG16: `python runner.py`
- To run VGG16 with Abs NormOut layers replacing both layers 47 and 48, as well as 50 and 51 (both of which are (ReLU, Dropout) pairs): `python runner.py --custom-layer-name "NormOut" --replace-layers 47 50 --remove-layers 48 51`
- To run VGG16 with all ReLU and BatchNorm layers replaced with NormOut: `python runner.py --custom-layer-name "NormOut" --replace-layers 1 3 6 8 11 13 15 18 20 22 25 27 29 34 37 --remove-layers 35 38 --custom-tag AllNormOut --vgg-no-batch-norm`


### HMS O2 Interactive Usage

Request a node with `srun -n 6 --mem 40G --pty -t 10:00:00 -p gpu --gres=gpu:1 bash`, activate your relevant environment (Xander: `source activate sdm_env`, Deepak: `conda activate env_pytorch`), then call `module load gcc/9.2.0`. Run with `python runner.py`.  Use `python runner.py -h` for command line arguments. 

*Note, AutoAttack should be installed via `pip install git+https://github.com/fra31/auto-attack`.*