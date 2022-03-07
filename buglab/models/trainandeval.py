#!/usr/bin/env python
"""
Usage:
    train.py [options] MODEL_NAME TRAIN_DATA_PATH VALID_DATA_PATH TEST_DATA_PATH METADATA_PATH MODEL_FILENAME

Options:
    --aml                         Run this in Azure ML
    --amp                         Use AMP
    --azure-info=<path>           Azure authentication information file (JSON). Used to load data from Azure storage.
    --limit-num-elements=<num>    Limit the number of elements to evaluate on.
    --max-num-epochs=<epochs>     The maximum number of epochs to run training for. [default: 100]
    --reduce-kind=<rdc>           The type of reduction on nodes and edges for the hypergnn message passing mechanism. [default: max]
    --tie-weights                 Tie the weights of the hypergnn layers. [default: False]
    --update-edge-embedding       Update edges types embedding with the aggregated features after each message passing layer. [default: False]
    --no-use-arg-names            Do not use the information of the argument names in the model. [default: False]
    --max-files-per-fold=<n>      The maximum number of files to include in each fold.
    --max-memory=<mm>             The maximum amount of memory to use in a batch. [default: 700_000]
    --minibatch-size=<size>       The minibatch size. [default: 300]
    --restore-path=<path>         The path to previous model file for starting from previous checkpoint.
    --validate-after=<n_samples>  Run the validation after seen n_samples. [default: 1000000]
    --sequential                  Do not parallelize data loading. Makes debugging easier.
    --seed=<int>                  Set random seed. [default: 0]
    --profile=<path>              Enable profiling at target path.
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
