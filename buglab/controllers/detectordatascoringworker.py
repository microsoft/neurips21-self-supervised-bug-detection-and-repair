import argparse
import logging
import math
import time
from itertools import cycle, islice
from tempfile import TemporaryDirectory
from threading import Thread
from typing import NoReturn

import msgpack
import torch
import zmq
from dpu_utils.utils import run_and_debug
from tqdm import tqdm

from buglab.controllers.helper.dummydatageneratingpipeline import get_data_from_folder
from buglab.data.modelsync import MockModelSyncClient, ModelSyncClient
from buglab.utils.logging import MetricProvider, configure_logging
from buglab.utils.replaybuffer import ReplayBuffer

metric_provider = MetricProvider("DetectorDataScoringWorker")


def queue_loader(replay_buffer: ReplayBuffer, data_pipeline_address: str) -> NoReturn:
    """A thread adding the received data into a queue."""
    context = zmq.Context.instance()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(data_pipeline_address)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    incoming_counter = metric_provider.new_counter(counter_name="incoming_messages")
    delay_measure = metric_provider.new_latency_measure(measure_name="incoming_latency")
    tracer = metric_provider.get_tracer()

    while True:
        with tracer.start_as_current_span("enqueue"), delay_measure:
            data = subscriber.recv()
        incoming_counter.inc()
        replay_buffer.add(msgpack.loads(data))


def hydrate_replay_buffer(replay_buffer, path, num_elements):
    for sample in tqdm(
        islice(get_data_from_folder(path), num_elements),
        desc="Hydrating replay buffer",
    ):
        replay_buffer.add(sample)


LOGGER = logging.getLogger(__name__)


def run(args):
    tracer = metric_provider.get_tracer()

    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {model_device}.")
    if args.fixed_model_path is not None:
        model_sync_client = MockModelSyncClient(args.fixed_model_path)
    else:
        model_sync_client = ModelSyncClient(args.model_server_address, do_not_update_before_sec=5 * 60)
    model, nn = model_sync_client.ask(model_device)
    nn.eval()

    tmp = TemporaryDirectory()
    replay_gauge = metric_provider.new_gauge("replay_buffer_incoming_queue")
    replay_buffer = ReplayBuffer(backing_dir=tmp.name, gauge=replay_gauge, ttl=10)

    data_input_thread = Thread(target=lambda: queue_loader(replay_buffer, args.data_pipeline_address), daemon=True)
    data_input_thread.start()

    hydration_thread = Thread(
        target=lambda: hydrate_replay_buffer(replay_buffer, args.initial_data_hydration_path, args.replay_buffer_size),
        name="replay_buffer_hydration_thread",
        daemon=True,
    )
    hydration_thread.start()
    time.sleep(20)  # Wait for buffer to be filled a bit.

    # The results are placed in a queue of another process. We model this as a synchronous request.
    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.connect(args.target_queue_address)

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    outgoing_counter = metric_provider.new_counter(counter_name="outgoing_messages")
    queue_delay_measure = metric_provider.new_latency_measure("outgoing_latency")
    processing_time_measure = metric_provider.new_latency_measure("processing_time")

    for _ in tqdm(cycle(range(10))):  # while True:
        with tracer.start_as_current_span("Processing data point."):
            with tracer.start_as_current_span("Waiting for data."):
                data = list(replay_buffer.iter_batch(1))[0]

            processing_time_measure.start()
            with tracer.start_as_current_span("Getting ready for scoring."):
                graphs, rewrite_idxs = [], []
                for rewrite_idx, (graph, _) in data["rewrites"].items():
                    if rewrite_idx == "NO_BUG":
                        rewrite_idxs.append(-1)
                    else:
                        rewrite_idxs.append(int(rewrite_idx))
                    if graph is None:
                        LOGGER.error(f"None element for graph. Rewrite_idx: {rewrite_idx}")
                        continue
                    graphs.append(graph)
                assert len(rewrite_idxs) == len(set(rewrite_idxs))

                original_graph = data["original"]
                num_rewrite_candidates = len(original_graph["graph"]["reference_nodes"])
                discriminator_rewrite_logprobs = [-math.inf] * (num_rewrite_candidates + 1)  # +1 for the NO_BUG case

            with tracer.start_as_current_span("Scoring."):
                model_sync_client.update_params_if_needed(nn, model_device)
                predictions = model.predict(graphs, nn, model_device, parallelize=not args.sequential)
                predictions = list(predictions)

                with tracer.start_as_current_span("Processing scores."):
                    for i, (datapoint, location_logprobs, rewrite_logprobs) in enumerate(predictions):
                        target_fix_action_idx = datapoint["target_fix_action_idx"]
                        if target_fix_action_idx is None:
                            target_logprob = location_logprobs[-1]
                        else:
                            ground_node_idx = datapoint["graph"]["reference_nodes"][target_fix_action_idx]
                            target_logprob = (
                                location_logprobs[ground_node_idx] + rewrite_logprobs[target_fix_action_idx]
                            )

                        discriminator_rewrite_logprobs[rewrite_idxs[i]] = float(
                            target_logprob
                        )  # Can't serialise numpy.float32 objects.

                original_graph["candidate_rewrite_logprobs"] = discriminator_rewrite_logprobs

            processing_time_measure.stop()

        with queue_delay_measure:
            with tracer.start_as_current_span("Sending detection scores data."):
                serialized_graph = msgpack.dumps(original_graph)
                socket.send(serialized_graph)
            outgoing_counter.inc()
            with tracer.start_as_current_span("Waiting for confirmation of receipt."):
                if poller.poll(60 * 1000):  # 1min
                    if not bool(socket.recv()):
                        LOGGER.warning(f"Error accepting data from {__package__}.")
                else:
                    LOGGER.error(f"Timeout for sending data.")
                    socket.close()
                    socket = context.socket(zmq.REQ)
                    socket.connect(args.target_queue_address)

                    poller = zmq.Poller()
                    poller.register(socket, zmq.POLLIN)


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Score original/buggy code adding the bug detection probabilities and forward the data to the bug selector training."
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
        help="The zmq address to the data generating pipeline.",
    )

    parser.add_argument(
        "--replay-buffer-size",
        type=int,
        default=10000,
        help="The number of elements to pre-load in the replay buffer.",
    )

    parser.add_argument(
        "--target-queue-address",
        type=str,
        default="tcp://localhost:5559",
        help="The zmq address to the data accumulator queue.",
    )

    parser.add_argument(
        "--model-server-address",
        type=str,
        default="tcp://localhost:6000",
        help="The address to the bug detector model server.",
    )

    parser.add_argument(
        "--fixed-model-path",
        type=str,
        help="Use a fixed discriminator model, instead of asking for one from the server.",
    )

    parser.add_argument(
        "--prometheus-server-port",
        type=int,
        default=8000,
        help="The address to the prometheus server.",
    )

    parser.add_argument(
        "--enable-tracing",
        action="store_true",
        help="Set to enable recording tracing information.",
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Should any computations happen sequentially?",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug on exception.",
    )

    args = parser.parse_args()
    metric_provider.start_server(args.prometheus_server_port)
    metric_provider.set_tracing(args.enable_tracing)
    run_and_debug(lambda: run(args), args.debug)
