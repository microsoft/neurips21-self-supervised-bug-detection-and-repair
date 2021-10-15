import argparse
import logging
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import NoReturn

import msgpack
import yaml
import zmq
from dpu_utils.utils import RichPath, run_and_debug
from ptgnn.baseneuralmodel import ModelTrainer
from ptgnn.baseneuralmodel.utils.data import LazyDataIterable

from buglab.data.modelsync import ModelSyncServer
from buglab.models.modelregistry import load_model
from buglab.models.utils import LinearWarmupScheduler, optimizer
from buglab.utils.iteratorutils import limited_queue_iterator
from buglab.utils.logging import MetricProvider, configure_logging
from buglab.utils.msgpackutils import load_all_msgpack_l_gz

metric_provider = MetricProvider("BugSelectorTraining")

LOGGER = logging.getLogger(__name__)


def bug_detector_scored_data_queue_server(queue: Queue, address: str) -> NoReturn:
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(address)

    incoming_counter = metric_provider.new_counter(counter_name="incoming_messages")
    delay_measure = metric_provider.new_latency_measure(measure_name="incoming_latency")

    while True:
        with delay_measure:
            data = socket.recv()
        socket.send(bytes(True))
        received = msgpack.loads(data)
        queue.put(received)
        incoming_counter.inc()


def run(arguments):
    LOGGER.info("Run args: %s", arguments)

    if arguments.model_config.endswith(".yml"):
        with open(arguments.model_config) as f:
            run_spec = yaml.safe_load(f)
        LOGGER.info("Loaded YAML configuration:\n%s", yaml.dump(run_spec))
    else:
        run_spec = {"selector": {"modelName": arguments.model_config}, "training": {}}

    # Create the queue that will store the training data.
    training_queue = metric_provider.new_queue("training_queue", maxsize=100_000)
    # Create a thread that listens for bug detector scored data and inserts them into a queue.
    queue_subscription_thread = Thread(
        target=lambda: bug_detector_scored_data_queue_server(training_queue, arguments.bug_detector_scored_data_server),
        daemon=True,
        name="detector_scored_data_queue_server",
    )
    queue_subscription_thread.start()

    # Define the data iterables for training the generator.
    training_data = LazyDataIterable(lambda: limited_queue_iterator(training_queue, arguments.num_samples_between_eval))
    # Note: we are not using validation data for the generator at the moment.
    validation_data = LazyDataIterable(lambda: iter([]))

    # Load in the generator.
    model_path = Path(arguments.model_save_path)
    model, nn, initialize_metadata = load_model(
        run_spec["selector"], model_path, arguments.restore_path, restore_if_model_exists=True
    )

    # Start the model training.
    trainer = ModelTrainer(
        model,
        model_path,
        max_num_epochs=int(arguments.max_num_epochs),
        minibatch_size=run_spec["training"].get("minibatchSize", int(arguments.minibatch_size)),
        optimizer_creator=lambda p: optimizer(p, float(run_spec["training"].get("learningRate", 1e-4))),
        clip_gradient_norm=run_spec["training"].get("gradientClipNorm", None),
        scheduler_creator=lambda o: LinearWarmupScheduler(o),
        enable_amp=arguments.amp,
    )
    if nn is not None:
        trainer.neural_module = nn

    model_sync_server: ModelSyncServer = None

    # Define and register hooks:
    model_publish_counter = metric_provider.new_counter("models_published")

    def publish_updated_model(model, nn):
        nonlocal model_sync_server
        model_sync_server.update_parameters(nn)
        model_publish_counter.inc()

    train_loss_metric = metric_provider.new_measure("training_loss")
    trainer.register_train_epoch_end_hook(lambda _, __, ___, metrics: train_loss_metric.record(metrics["Loss"]))

    # TODO: when we run validation, log the validation loss.
    # validation_loss_metric = metric_provider.new_measure("validation_loss")
    # trainer.register_validation_epoch_end_hook(
    #     lambda _, __, ___, metrics: validation_loss_metric.record(metrics["Loss"])
    # )

    def publish_created_model(model, nn, optimizer):
        nonlocal model_sync_server
        assert model_sync_server is None
        model_sync_server = ModelSyncServer(arguments.model_sync_server_address, model, nn)
        model_publish_counter.inc()

    trainer.register_training_start_hook(publish_created_model)

    if initialize_metadata:
        initialize_metadata = False

        data = LazyDataIterable(
            lambda: load_all_msgpack_l_gz(
                RichPath.create(arguments.load_metadata_from),
                shuffle=False,
            )
        )
        trainer.load_metadata_and_create_network(data, parallelize=not arguments.sequential, show_progress_bar=True)

    # Hijack trainer to ignore validation
    def _run_validation(
        validation_tensors,
        epoch,
        best_target_metric,
        device,
        parallelize,
        show_progress_bar,
    ):
        dummy_validation_metric = 42.0
        target_metric_improved = True
        publish_updated_model(trainer.model, trainer.neural_module)
        return dummy_validation_metric, target_metric_improved

    trainer._run_validation = _run_validation

    with model._tensorize_all_location_rewrites():
        trainer.train(
            training_data,
            validation_data,
            initialize_metadata=initialize_metadata,
            parallelize=not arguments.sequential,
            use_multiprocessing=False,
            patience=10,
        )


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Train a bug selector model.")

    parser.add_argument("model_config", type=str, help="the type of the model to train")
    parser.add_argument(
        "load_metadata_from",
        type=str,
        help="Load metadata from the data found in the selected folder.",
    )

    parser.add_argument(
        "model_save_path",
        type=str,
        help="The target path to store the trained model.",
    )

    parser.add_argument(
        "--model-sync-server-address",
        type=str,
        default="tcp://*:6001",
        help="The zmq address serve requests for serving the latest versions of the generator model.",
    )

    parser.add_argument(
        "--bug-detector-scored-data-server",
        type=str,
        default="tcp://*:5559",
        help="The zmq address for the server listening for the bug detector-scored data.",
    )

    parser.add_argument(
        "--prometheus-server-port",
        type=int,
        default=8001,
        help="The address to the prometheus server.",
    )

    parser.add_argument(
        "--enable-tracing",
        action="store_true",
        help="Set to enable recording tracing information.",
    )

    parser.add_argument(
        "--restore-path",
        type=str,
        default=None,
        help="The path from which to restore a model.",
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Should the model training run in parallel?",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug on exception.",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable AMP.",
    )

    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=300,
        help="The minibatch size",
    )

    parser.add_argument(
        "--num-samples-between-eval",
        type=int,
        default=1000,
        help="The number of samples to train on between reporting metrics and updating the model.",
    )

    parser.add_argument(
        "--max-num-epochs",
        type=int,
        default=100_000,
        help="The maximum number of training epochs to run for.",
    )

    args = parser.parse_args()
    metric_provider.start_server(args.prometheus_server_port)
    metric_provider.set_tracing(args.enable_tracing)
    run_and_debug(lambda: run(args), args.debug)
