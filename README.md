# NormOut: Divisive Normalization as Dropout Probabilities
*Deepak Singh & Xander Davies*

To run, call `module load gcc/9.2.0`, request a node with `srun -n 6 --mem 10G --pty -t 10:00:00 -p gpu --gres=gpu:1 bash` and then `python runner.py`. Use `python runner.py -h` for command line arguments. In Xander's case, also call `source activate sdm_env` to activate correct conda environment.
