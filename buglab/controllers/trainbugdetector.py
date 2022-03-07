import argparse
import logging
import msgpack
import time
import yaml
import zmq
from dpu_utils.utils import RichPath, run_and_debug
from pathlib import Path
from ptgnn.baseneuralmodel import ModelTrainer
from ptgnn.baseneuralmodel.utils.data import LazyDataIterable
from queue import Queue
from threading import Thread

from buglab.data.modelsync import ModelSyncServer
from buglab.models.modelregistry import load_model
from buglab.models.train import construct_data_loading_callable
from buglab.models.utils import LinearWarmupScheduler, optimizer
from buglab.utils.iteratorutils import limited_queue_iterator
from buglab.utils.loggingutils import MetricProvider, configure_logging
from buglab.utils.msgpackutils import load_all_msgpack_l_gz

LOGGER = logging.getLogger(__name__)
metric_provider = MetricProvider("BugDetectorTraining")


def training_queue_filler(queue: Queue, data_buffer_address: str):
    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.connect(data_buffer_address)

    while True:
        socket.send(b"")
        queue.put(msgpack.loads(socket.recv()))


def run(arguments):
    LOGGER.info("Run args: %s", arguments)

    if arguments.model_config.endswith(".yml"):
        with open(arguments.model_config) as f:
            run_spec = yaml.safe_load(f)
        LOGGER.info("Loaded YAML configuration:\n%s", yaml.dump(run_spec))
    else:
        run_spec = {"detector": {"modelName": arguments.model_config}, "training": {}}

    # Kick-off training of the model and validate at regular intervals
    training_data_queue = metric_provider.new_queue(
        "training_queue_from_buffer", maxsize=1000, description="The number of elements loaded from the data buffer."
    )
    training_data = LazyDataIterable(lambda: limited_queue_iterator(training_data_queue, 400_000))

    data_buffer_to_training_thread = Thread(
        target=lambda: training_queue_filler(training_data_queue, arguments.training_data_buffer_address),
        name="data_buffer_to_training_thread",
        daemon=True,
    )
    data_buffer_to_training_thread.start()

    validation_data_path = RichPath.create(arguments.validation_data_path)
    validation_data = LazyDataIterable(lambda: load_all_msgpack_l_gz(validation_data_path))

    model_path = Path(arguments.model_save_path)
    restore_path = arguments.restore_path
    model, nn, initialize_metadata = load_model(
        run_spec["detector"], model_path, restore_path, restore_if_model_exists=True
    )

    # Write out config file.
    model_path.parent.mkdir(exist_ok=True)
    (model_path.parent / "config.yml").write_text(yaml.dump(run_spec))

    trainer = ModelTrainer(
        model,
        model_path,
        max_num_epochs=int(arguments.max_num_epochs),
        minibatch_size=run_spec["training"].get("minibatchSize", int(arguments.minibatch_size)),
        optimizer_creator=lambda p: optimizer(p, float(run_spec["training"].get("learningRate", 1e-4))),
        clip_gradient_norm=run_spec["training"].get("gradientClipNorm", None),
        scheduler_creator=lambda o: LinearWarmupScheduler(o),
        enable_amp=arguments.amp,
        catch_cuda_ooms=True,
    )
    if nn is not None:
        trainer.neural_module = nn

    model_sync_server: ModelSyncServer = None
    sync_counter = metric_provider.new_counter("parameter_updates")

    # Register training hooks.
    def publish_updated_model(model, nn, epoch, metrics):
        model_sync_server.update_parameters(nn)
        sync_counter.inc()

    trainer.register_validation_epoch_end_hook(publish_updated_model)

    def checkpoint_model(model, nn, epoch, metrics):
        target_path = model_path.parent / f"detector-ckpt-{int(time.time())}.pkl.gz"
        model.save(target_path, nn)

    train_loss_metric = metric_provider.new_measure("training_loss")
    trainer.register_train_epoch_end_hook(lambda _, __, ___, metrics: train_loss_metric.record(metrics["Loss"]))

    def publish_created_model(model, nn, optimizer):
        nonlocal model_sync_server
        assert model_sync_server is None
        model_sync_server = ModelSyncServer(arguments.model_sync_server, model, nn)
        sync_counter.inc()

    trainer.register_training_start_hook(publish_created_model)

    if initialize_metadata:
        trainer.load_metadata_and_create_network(
            construct_data_loading_callable(
                RichPath.create(args.initial_data_hydration_path),
                shuffle=False,
            )(),
            not arguments.sequential,
            False,
        )

    if not run_spec["training"].get("useDetectorValidation", True):

        def _run_validation(
            validation_tensors,
            epoch,
            best_target_metric,
            device,
            parallelize,
            show_progress_bar,
        ):
            dummy_validation_metric = 32.0
            target_metric_improved = True
            publish_updated_model(None, trainer.neural_module, None, None)
            checkpoint_model(trainer.model, trainer.neural_module, None, None)
            return dummy_validation_metric, target_metric_improved, {}

        trainer._run_validation = _run_validation

    trainer.train(
        training_data,
        validation_data,
        initialize_metadata=False,
        parallelize=not arguments.sequential,
        use_multiprocessing=False,
        shuffle_training_data=False,  # Replay buffer takes care of this
        patience=run_spec["training"].get("patience", 10),
        show_progress_bar=False,
    )


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Subscribe to a data generating pipeline and use input data to train a bug detector model."
    )

    parser.add_argument(
        "initial_data_hydration_path",
        type=str,
        help="The path to hydrate the replay buffer upon startup.",
    )
    parser.add_argument("model_config", type=str, help="the type of the model to train or a path to a .yaml config")

    # TODO: For now use fixed validation data, later fix that...
    parser.add_argument("validation_data_path", type=str, help="the path to fixed validation data.")

    parser.add_argument("model_save_path", type=str, help="the target path to store the trained model.")

    parser.add_argument(
        "--training-data-buffer-address",
        type=str,
        default="tcp://localhost:5560",
        help="The zmq address to the data pipeline.",
    )

    parser.add_argument(
        "--evaluate-every-n-samples",
        type=int,
        default=150_000,
        help="The number of elements to train on during a single epoch.",
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
        help="Enable automatic mixed precision.",
    )

    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=300,
        help="The minibatch size",
    )

    parser.add_argument(
        "--max-num-epochs",
        type=int,
        default=10_000,
        help="The maximum number of training epochs to run for.",
    )

    parser.add_argument(
        "--model-sync-server",
        type=str,
        default="tcp://*:6000",
        help="The address to listen for clients asking to use an update discriminator model.",
    )

    parser.add_argument(
        "--prometheus-server-port",
        type=int,
        default=8002,
        help="The address to the prometheus server.",
    )

    parser.add_argument(
        "--enable-tracing",
        action="store_true",
        help="Set to enable recording tracing information.",
    )

    args = parser.parse_args()
    metric_provider.start_server(args.prometheus_server_port)
    metric_provider.set_tracing(args.enable_tracing)
    run_and_debug(lambda: run(args), args.debug)
