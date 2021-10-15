#!/usr/bin/env python
"""
Usage:
    ensemble.py [options] OUT_MODEL_FILENAME ENSEMBLE_KIND MODEL_FILENAMES...

Options:
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import gzip
from pathlib import Path

import torch
from docopt import docopt
from dpu_utils.utils import run_and_debug
from ptgnn.baseneuralmodel import AbstractNeuralModel

from buglab.models.ensemble.wrapper import EnsembleModuleWrapper, EnsembleWrapper


def run(arguments):
    models, nns = [], []

    for path in arguments["MODEL_FILENAMES"]:
        print(f"Loading {path}...")
        model_path = Path(path)
        model, nn = AbstractNeuralModel.restore_model(model_path, "cpu")
        models.append(model)
        nns.append(nn)

    print(f"Loaded {len(models)} models. Saving...")

    model = EnsembleWrapper(models, arguments["ENSEMBLE_KIND"])
    nn = EnsembleModuleWrapper(nns)
    with gzip.open(arguments["OUT_MODEL_FILENAME"], "wb") as f:
        torch.save((model, nn), f)


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get("--debug", False))
