# NormOut: Divisive Normalization as Dropout Probabilities
*Deepak Singh & Xander Davies*

## Usage

To use, request a node with `srun -n 6 --mem 50G --pty -t 10:00:00 -p gpu --gres=gpu:1 bash`, call `module load gcc/9.2.0`, then run with `python runner.py`.  Use `python runner.py -h` for command line arguments. In Xander's case, also call `source activate sdm_env` prior to run command to activate correct conda environment.

*Note, AutoAttack should be installed via `pip install git+https://github.com/fra31/auto-attack`.*

## Files

- `utils.py` is where we define NormOut and TopK layers (both are `nn.Module`).
- `basic_lightning_model.py` is where the class BasicLightningModel is defined, which we use for setting up the optimizer and dataloaders.
- `attacks.py` is where the class Attacks uses the `on_validation_epoch_end` method to run our adversarial attacks used to evaluate robustness.
- `models/` contains definitions of relevant models which we evaluate. Models need only define the `forward` method (along with corresponding definitions in the `__init__`), as they inherit everything else from BasicLightningModel and Attacks.
- `runner.py` is the runner file used to train networks; for more information call `python runner.py -h`.