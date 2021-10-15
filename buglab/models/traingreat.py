#!/usr/bin/env python
"""
Usage:
    train.py [options] TRAIN_DATA_PATH VALID_DATA_PATH MODEL_FILENAME

Options:
    --aml                         Run this in Azure ML
    --amp                         Use AMP
    --azure-info=<path>           Azure authentication information file (JSON). Used to load data from Azure storage.
    --max-num-epochs=<epochs>     The maximum number of epochs to run training for. [default: 100]
    --max-files-per-fold=<n>      The maximum number of files to include in each fold.
    --minibatch-size=<size>       The minibatch size. [default: 30]
    --validate-after=<n_samples>  Run the validation after seen n_samples. [default: 1000000]
    --restore-path=<path>         The path to previous model file for starting from previous checkpoint.
    --sequential                  Do not parallelize data loading. Makes debugging easier.
    --quiet                       Do not show progress bar.
    -h --help                     Show this screen.
    --debug                       Enable debug routines. [default: False]
"""
import logging
import random
from pathlib import Path
from typing import Callable, Iterator, Optional

from docopt import docopt
from dpu_utils.utils import RichPath, load_jsonl_gz, run_and_debug
from ptgnn.baseneuralmodel import AbstractNeuralModel, ModelTrainer
from ptgnn.baseneuralmodel.utils.amlutils import configure_logging, log_run
from ptgnn.baseneuralmodel.utils.data import LazyDataIterable

from buglab.models.greatreimplementation import GreatVarMisuse
from buglab.models.utils import LinearWarmupScheduler, optimizer

LOGGER = logging.getLogger(__name__)


def load_all_json_l_gz(
    path: RichPath,
    shuffle: bool = False,
    take_only_first_n_files: Optional[int] = None,
    limit_num_yielded_elements: Optional[int] = None,
) -> Iterator:
    all_files = sorted(path.iterate_filtered_files_in_dir("*.jsonl.gz"))
    if take_only_first_n_files is not None:
        all_files = all_files[:take_only_first_n_files]
    if shuffle:
        random.shuffle(all_files)

    sample_idx = 0
    for jsonlgz_file in all_files:
        try:
            for element in jsonlgz_file.read_as_jsonl():
                if element is not None:
                    sample_idx += 1
                    yield element
                if limit_num_yielded_elements is not None and sample_idx > limit_num_yielded_elements:
                    return
        except Exception as e:
            print(f"Error loading {jsonlgz_file}: {e}.")


def construct_data_loading_callable(
    data_path: RichPath,
    shuffle: bool = False,
    max_files_per_fold: Optional[int] = None,
    limit_num_yielded_elements: Optional[int] = None,
) -> Callable[[], Iterator]:
    return lambda: load_all_json_l_gz(
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
    initialize_metadata = True
    if arguments.get("--restore-path", None) is not None:
        import torch

        LOGGER.info("Resuming training from %s." % model_path)
        initialize_metadata = False
        model, nn = AbstractNeuralModel.restore_model(
            model_path, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
    else:
        nn = None
        model = GreatVarMisuse(
            {
                "num_layers": 10,
                "num_heads": 8,
                "intermediate_dimension": 2048,
                "dropout_rate": 0.1,
                "rezero_mode": "off",
                "normalization_mode": "prenorm",
            },
            embedding_dim=512,
            vocab_size=10240,  # Closest multiple of 64
        )

    trainer = ModelTrainer(
        model,
        model_path,
        max_num_epochs=int(arguments["--max-num-epochs"]),
        minibatch_size=int(arguments["--minibatch-size"]),
        optimizer_creator=optimizer,
        clip_gradient_norm=0.25,  # https://github.com/VHellendoorn/ICLR20-Great/blob/472069aba236244f8e06f9f98c565df7f9b2ea64/running/run_model.py#L73
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

    trainer.train(
        training_data,
        validation_data,
        show_progress_bar=not arguments["--quiet"],
        initialize_metadata=initialize_metadata,
        parallelize=not arguments["--sequential"],
        patience=10,
    )

    if aml_ctx is not None:
        aml_ctx.upload_file(name="model.pkl.gz", path_or_stream=str(model_path))
        aml_ctx.upload_file(name="full.log", path_or_stream=log_path)


if __name__ == "__main__":
    args = docopt(__doc__)

    run_and_debug(lambda: run(args), args.get("--debug", False))
