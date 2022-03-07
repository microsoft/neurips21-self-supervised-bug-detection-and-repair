#!/usr/bin/env python
"""
Usage:
    train.py [options] MODEL_NAME TRAIN_DATA_PATH VALID_DATA_PATH METADATA_PATH MODEL_FILENAME

Options:
    --aml                         Run this in Azure ML
    --amp                         Use AMP
    --azure-info=<path>           Azure authentication information file (JSON). Used to load data from Azure storage.
    --max-num-epochs=<epochs>     The maximum number of epochs to run training for. [default: 100]
    --reduce-kind=<rdc>           The type of reduction on nodes and edges for the hypergnn message passing mechanism. [default: max]
    --tie-weights                 Tie the weights of the hypergnn layers. [default: False]
    --update-edge-embedding       Update edges types embedding with the aggregated features after each message passing layer. [default: False]
    --no-use-arg-names            Do not use the information of the argument names in the model. [default: False]
    --max-files-per-fold=<n>      The maximum number of files to include in each fold.
    --max-memory=<mm>             The maximum amount of memory to use in a batch. [default: 500_000]
    --minibatch-size=<size>       The minibatch size. [default: 300]
    --validate-after=<n_samples>  Run the validation after seen n_samples. [default: 1000000]
    --restore-path=<path>         The path to previous model file for starting from previous checkpoint.
    --sequential                  Do not parallelize data loading. Makes debugging easier.
    --seed=<int>                  Set random seed. [default: 0]
    --profile=<path>              Enable profiling at target path.
    --quiet                       Do not show progress bar.
    -h --help                     Show this screen.
    --debug                       Enable debug routines. [default: False]
"""
from typing import Callable, Iterator, Optional

import logging
import yaml
from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug
from pathlib import Path
from ptgnn.baseneuralmodel import ModelTrainer
from ptgnn.baseneuralmodel.utils.amlutils import configure_logging, log_run
from ptgnn.baseneuralmodel.utils.data import LazyDataIterable

from buglab.models.modelregistry import load_model
from buglab.models.utils import LinearWarmupScheduler, optimizer
from buglab.representations.data import BugLabData
from buglab.utils.msgpackutils import load_all_msgpack_l_gz
from buglab.utils.randomutils import set_seed

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

    set_seed(int(arguments["--seed"]))

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

    model_name = arguments["MODEL_NAME"]
    if model_name.endswith(".yml"):
        with open(model_name) as f:
            run_spec = yaml.safe_load(f)["runConfig"]
        LOGGER.info("Loaded YAML configuration:\n%s", yaml.dump(run_spec))
    else:
        run_spec = {
            "detector": {
                "modelName": model_name,
                "type_name_exclude_set": frozenset(),
                "node_reduce_kind": arguments["--reduce-kind"],
                "tie_weights": arguments["--tie-weights"],
                "update_edge_embedding": arguments["--update-edge-embedding"],
                "use_arg_names": not arguments["--no-use-arg-names"],
                "max_memory": int(arguments["--max-memory"]),
                "localization_module_type": "CandidateQuery",
            },
            "training": {},
        }

    model_path = Path(arguments["MODEL_FILENAME"])
    restore_path = arguments.get("--restore-path", None)
    model, nn, initialize_metadata = load_model(run_spec["detector"], model_path, restore_path, arguments["--aml"])

    trainer = ModelTrainer(
        model,
        model_path,
        max_num_epochs=int(arguments["--max-num-epochs"]),
        minibatch_size=run_spec["training"].get("minibatchSize", int(arguments["--minibatch-size"])),
        optimizer_creator=lambda p: optimizer(p, float(run_spec["training"].get("learningRate", 2e-5))),
        clip_gradient_norm=run_spec["training"].get("gradientClipNorm", 0.1),
        scheduler_creator=lambda o: LinearWarmupScheduler(
            o,
            num_warmup_steps=1000,
        ),
        enable_amp=arguments["--amp"],
        catch_cuda_ooms=True,
    )

    if nn is not None:
        trainer.neural_module = nn

    trainer.register_train_epoch_end_hook(
        lambda model, nn, epoch, metrics: log_run(aml_ctx, "train", model, epoch, metrics)
    )
    trainer.register_validation_epoch_end_hook(
        lambda model, nn, epoch, metrics: log_run(aml_ctx, "valid", model, epoch, metrics)
    )

    if arguments["--profile"]:
        LOGGER.warning("Running in Profiling Mode...")
        from torch.profiler import profile, schedule, tensorboard_trace_handler

        with profile(
            profile_memory=False,
            with_stack=True,
            schedule=schedule(wait=20, warmup=10, skip_first=10, active=10, repeat=3),
            on_trace_ready=tensorboard_trace_handler(dir_name=arguments["--profile"]),
        ) as profiler:
            trainer._create_optimizer = lambda p: optimizer(
                p, float(run_spec["training"].get("learningRate", 1e-4)), profiler=profiler
            )
            trainer.train(
                training_data,
                validation_data,
                show_progress_bar=not arguments["--quiet"],
                initialize_metadata=initialize_metadata,
                parallelize=not arguments["--sequential"],
                patience=run_spec["training"].get("patience", 10),
                validate_on_start=False,
            )
    else:
        if initialize_metadata:
            metadata_data_path = RichPath.create(arguments["METADATA_PATH"], azure_info_path)
            data_for_metadata = LazyDataIterable(
                construct_data_loading_callable(metadata_data_path, shuffle=True, limit_num_yielded_elements=750_000)
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
            patience=run_spec["training"].get("patience", 10),
        )

    if aml_ctx is not None:
        aml_ctx.upload_file(name="model.pkl.gz", path_or_stream=str(model_path))
        aml_ctx.upload_file(name="full.log", path_or_stream=log_path)
        if arguments["--profile"]:
            aml_ctx.upload_folder(name="tb-logs", path=arguments["--profile"])


if __name__ == "__main__":
    args = docopt(__doc__)

    run_and_debug(lambda: run(args), args.get("--debug", False))
