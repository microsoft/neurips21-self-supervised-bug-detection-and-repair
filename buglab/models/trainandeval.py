#!/usr/bin/env python
"""
Usage:
    train.py [options] MODEL_NAME TRAIN_DATA_PATH VALID_DATA_PATH TEST_DATA_PATH MODEL_FILENAME

Options:
    --aml                         Run this in Azure ML
    --amp                         Use AMP
    --azure-info=<path>           Azure authentication information file (JSON). Used to load data from Azure storage.
    --limit-num-elements=<num>    Limit the number of elements to evaluate on.
    --max-num-epochs=<epochs>     The maximum number of epochs to run training for. [default: 100]
    --max-files-per-fold=<n>      The maximum number of files to include in each fold.
    --minibatch-size=<size>       The minibatch size. [default: 300]
    --restore-path=<path>         The path to previous model file for starting from previous checkpoint.
    --validate-after=<n_samples>  Run the validation after seen n_samples. [default: 1000000]
    --sequential                  Do not parallelize data loading. Makes debugging easier.
    --quiet                       Do not show progress bar.
    -h --help                     Show this screen.
    --debug                       Enable debug routines. [default: False]
"""
from docopt import docopt
from dpu_utils.utils import run_and_debug

from buglab.models import evaluate, train

if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: train.run(args), args.get("--debug", False))
    run_and_debug(lambda: evaluate.run(args), args.get("--debug", False))
