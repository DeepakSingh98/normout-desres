# NormOut: Divisive Normalization as Dropout Probabilities
*Deepak Singh & Xander Davies*

## Files 

- `attacks.py` defines the abstract base class `Attacks` which uses the `on_validation_epoch_end` method to run our adversarial attacks used to evaluate robustness.
- `base_model.py` defines the abstract base class `BasicLightningModel` for a basic `pl.LightningModule` with `Attacks` inherited. It configures the optimizer and dataloaders, but requires that the `forward` method be defined by a child.
- `custom_model.py` inherits `BasicLightningModel` and defines a base model architecture specified by `model-name`. Models can be customized as follows:
    1. Replacing existing layers with custom layers (e.g., `--custom-layer-name NormOut --replace-layers 47 50` replaces the layers at indices 47 and 50 with `NormOut` layers).
    2. Removing layers (e.g., `--remove-layers 20` removes the layer at index 20)
    3. Inserting custom layers (e.g., `--custom-layer-name NormOut --insert-layer 53` inserts a `NormOut` layer at index 53). 
    *Note: The editing is performed in the following order: replace, remove, insert. All indices should be listed in increasing order.*
- `custom_layers/` stores custom layers (`nn.Module`) like `NormOut`.
- `models/` stores functions which return a `nn.Module` list corresponding to specified model architectures. These functions are called in `CustomModel` based on `model-name`. We also include `.txt` files of each model to assist with model customization.
- `runner.py` is the runner file for training. Use `python runner.py -h` to see runner specifications. 

## Usage 

### Example Usage
- To run a baseline VGG16: `python runner.py`
- To run VGG16 with Abs NormOut layers replacing both layers 47 and 48, as well as 50 and 51 (both of which are (ReLU, Dropout) pairs): `python runner.py --custom-layer-name NormOut --replace-layers 47 50 --remove-layers 48 51`
- To run VGG16 with all ReLU and BatchNorm layers replaced with NormOut: `python runner.py --custom-layer-name NormOut --replace-layers 1 3 6 8 11 13 15 18 20 22 25 27 29 34 37 --remove-layers 35 38 --custom-tag AllNormOut --batch-norm False`


### HMS O2 Interactive Usage

Request a node with `srun -n 6 --mem 40G --pty -t 10:00:00 -p gpu --gres=gpu:1 bash`, activate your relevant environment (Xander: `source activate sdm_env`, Deepak: `conda activate env_pytorch`), then call `module load gcc/9.2.0`. Run with `python runner.py`.  Use `python runner.py -h` for command line arguments.

*Note, RobustBench should be installed via `pip install git+https://github.com/RobustBench/robustbench`. Make sure you are running python >= 3.7.1\**
