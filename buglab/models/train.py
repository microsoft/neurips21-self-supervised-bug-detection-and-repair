#!/usr/bin/env python
"""
Usage:
    train.py [options] MODEL_NAME TRAIN_DATA_PATH VALID_DATA_PATH MODEL_FILENAME

Options:
    --aml                         Run this in Azure ML
    --amp                         Use AMP
    --azure-info=<path>           Azure authentication information file (JSON). Used to load data from Azure storage.
    --max-num-epochs=<epochs>     The maximum number of epochs to run training for. [default: 100]
    --max-files-per-fold=<n>      The maximum number of files to include in each fold.
    --minibatch-size=<size>       The minibatch size. [default: 300]
    --validate-after=<n_samples>  Run the validation after seen n_samples. [default: 1000000]
    --restore-path=<path>         The path to previous model file for starting from previous checkpoint.
    --sequential                  Do not parallelize data loading. Makes debugging easier.
    --quiet                       Do not show progress bar.
    -h --help                     Show this screen.
    --debug                       Enable debug routines. [default: False]
"""
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug
from ptgnn.baseneuralmodel import ModelTrainer
from ptgnn.baseneuralmodel.utils.amlutils import configure_logging, log_run
from ptgnn.baseneuralmodel.utils.data import LazyDataIterable

from buglab.models.modelregistry import load_model
from buglab.models.utils import LinearWarmupScheduler, optimizer
from buglab.representations.data import BugLabData
from buglab.utils.msgpackutils import load_all_msgpack_l_gz

LOGGER = logging.getLogger(__name__)


def construct_data_loading_callable(
    data_path: RichPath,
    shuffle: bool = False,
    max_files_per_fold: Optional[int] = None,
    limit_num_yielded_elements: Optional[int] = None,
) -> Callable[[], Iterator[BugLabData]]:
    return lambda: load_all_msgpack_l_gz(
        data_path,
        shuffle=shuffle,
        take_only_first_n_files=max_files_per_fold,
        limit_num_yielded_elements=limit_num_yielded_elements,
    )


def run(arguments):
    if arguments["--aml"]:
        import torch
        from azureml.core.run import Run

        aml_ctx = Run.get_context()
        assert torch.cuda.is_available(), "No CUDA available. Aborting training."
    else:
        aml_ctx = None

    log_path = configure_logging(aml_ctx)
    logging.getLogger("azure.storage.blob").setLevel(logging.ERROR)
    logging.getLogger("azure.core").setLevel(logging.ERROR)

    azure_info_path = arguments.get("--azure-info", None)

    max_files_per_fold = arguments["--max-files-per-fold"]
    if max_files_per_fold is not None:
        max_files_per_fold = int(max_files_per_fold)

    training_data_path = RichPath.create(arguments["TRAIN_DATA_PATH"], azure_info_path)
    arguments: dict
    training_data = LazyDataIterable(
        construct_data_loading_callable(
            training_data_path,
            shuffle=True,
            max_files_per_fold=max_files_per_fold,
            limit_num_yielded_elements=int(arguments["--validate-after"]),
        )
    )

    validation_data_path = RichPath.create(arguments["VALID_DATA_PATH"], azure_info_path)
    validation_data = LazyDataIterable(
        construct_data_loading_callable(
            validation_data_path,
            max_files_per_fold=max_files_per_fold,
        )
    )

    model_path = Path(arguments["MODEL_FILENAME"])
    restore_path = arguments.get("--restore-path", None)
    model_spec = {"modelName": arguments["MODEL_NAME"]}
    model, nn, initialize_metadata = load_model(model_spec, model_path, restore_path, arguments["--aml"])

    trainer = ModelTrainer(
        model,
        model_path,
        max_num_epochs=int(arguments["--max-num-epochs"]),
        minibatch_size=int(arguments["--minibatch-size"]),
        optimizer_creator=optimizer,
        clip_gradient_norm=0.5,
        scheduler_creator=lambda o: LinearWarmupScheduler(o),
        enable_amp=arguments["--amp"],
    )
    if nn is not None:
        trainer.neural_module = nn

    trainer.register_train_epoch_end_hook(
        lambda model, nn, epoch, metrics: log_run(aml_ctx, "train", model, epoch, metrics)
    )
    trainer.register_validation_epoch_end_hook(
        lambda model, nn, epoch, metrics: log_run(aml_ctx, "valid", model, epoch, metrics)
    )

    if initialize_metadata:
        metadata_data_path = RichPath.create(arguments["TRAIN_DATA_PATH"], azure_info_path)
        data_for_metadata = LazyDataIterable(
            construct_data_loading_callable(metadata_data_path, shuffle=True, limit_num_yielded_elements=250_000)
        )
        trainer.load_metadata_and_create_network(
            data_for_metadata, not arguments["--sequential"], not arguments["--quiet"]
        )

    trainer.train(
        training_data,
        validation_data,
        show_progress_bar=not arguments["--quiet"],
        initialize_metadata=False,
        parallelize=not arguments["--sequential"],
        patience=10,
    )

    if aml_ctx is not None:
        aml_ctx.upload_file(name="model.pkl.gz", path_or_stream=str(model_path))
        aml_ctx.upload_file(name="full.log", path_or_stream=log_path)


if __name__ == "__main__":
    args = docopt(__doc__)

    run_and_debug(lambda: run(args), args.get("--debug", False))
