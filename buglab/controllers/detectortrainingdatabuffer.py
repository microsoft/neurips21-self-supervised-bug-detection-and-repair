import argparse as argparse
import logging
import msgpack
import zmq
from dpu_utils.utils import RichPath, run_and_debug
from tempfile import TemporaryDirectory
from threading import Thread

from buglab.utils.loggingutils import MetricProvider, configure_logging
from buglab.utils.msgpackutils import load_all_msgpack_l_gz
from buglab.utils.replaybuffer import ReplayBuffer

LOGGER = logging.getLogger(__name__)
metric_provider = MetricProvider("DetectorTrainingDataBuffer")


def hydrate_replay_buffer(replay_buffer, path):
    for sample in load_all_msgpack_l_gz(
        RichPath.create(path),
        shuffle=True,
    ):
        replay_buffer.add(sample)


def connect_buffer_to_publisher(replay_buffer: ReplayBuffer, pipeline_address: str):
    context = zmq.Context.instance()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(pipeline_address)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    message_counter = metric_provider.new_counter("incoming_messages")
    graph_counter = metric_provider.new_counter("incoming_graphs")
    delay_measure = metric_provider.new_latency_measure("incoming_latency")

    while True:
        with delay_measure:
            msg = msgpack.loads(subscriber.recv())
        message_counter.inc()
        for graph_data, bug_prob in msg["rewrites"].values():
            if graph_data is not None:
                replay_buffer.add(graph_data)
                graph_counter.inc()


def run(arguments):
    LOGGER.info("Run args: %s", arguments)

    # Create the replay buffer and hydrate it with initial data
    tmp = TemporaryDirectory()
    replay_gauge = metric_provider.new_gauge("replay_buffer_incoming_queue")
    replay_buffer = ReplayBuffer(backing_dir=tmp.name, gauge=replay_gauge, ttl=arguments.sample_ttl)
    hydration_thread = Thread(
        target=lambda: hydrate_replay_buffer(replay_buffer, args.initial_data_hydration_path),
        name="replay_buffer_hydration_thread",
        daemon=True,
    )
    hydration_thread.start()

    # Create a thread that subscribes to the data generating pipeline and updates itself
    buffer_subscription_thread = Thread(
        target=lambda: connect_buffer_to_publisher(
            replay_buffer,
            arguments.data_pipeline_address,
        ),
        name="data_pipeline_to_replay_buffer_thread",
        daemon=True,
    )
    buffer_subscription_thread.start()

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{arguments.out_port}")

    while True:
        next_element = msgpack.dumps(next(replay_buffer.iter_batch(1)))
        _ = socket.recv()
        socket.send(next_element)


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Subscribe to a data generating pipeline and create a replay-like buffer for training."
    )

    parser.add_argument(
        "initial_data_hydration_path",
        type=str,
        help="The path to hydrate the replay buffer upon startup.",
    )

    parser.add_argument(
        "--data-pipeline-address",
        type=str,
        default="tcp://localhost:5558",
        help="The zmq address to the data pipeline.",
    )

    parser.add_argument(
        "--out-port",
        type=int,
        default=5560,
        help="The address to the prometheus server.",
    )

    parser.add_argument(
        "--sample-ttl",
        type=int,
        default=4,
        help="The number of times to show each sample to the trainer.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug on exception.",
    )

    parser.add_argument(
        "--prometheus-server-port",
        type=int,
        default=8003,
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
