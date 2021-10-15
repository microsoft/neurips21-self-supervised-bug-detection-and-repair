import argparse
import json
import logging
from collections import Counter, defaultdict
from threading import Event, Thread
from typing import Dict, List

import msgpack
import numpy as np
import torch
import zmq
from dpu_utils.utils import run_and_debug

from buglab.controllers.helper.randombugselectorserver import random_bug_selector_server, select_random_rewrites
from buglab.data.modelsync import MockModelSyncClient, ModelSyncClient
from buglab.utils.logging import MetricProvider, configure_logging

LOGGER = logging.getLogger(__name__)
metric_provider = MetricProvider("BugSelectorServer")


def calculate_selection_distribution(logprobs: List[float], temperature: float = 1, epsilon: float = 0) -> List[float]:
    if np.random.rand() < epsilon:
        return np.ones(len(logprobs)) * (1 / len(logprobs))
    unnormalised_selection_distribution = [np.exp(l / temperature) for l in logprobs]
    z = sum(unnormalised_selection_distribution)
    return unnormalised_selection_distribution / z


class BugSelectionStats:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.available_rewrite_frequency: Dict[str, float] = defaultdict(float)
        self.selected_rewrite_frequency: Dict[str, float] = defaultdict(float)
        self.entropy_sum = 0.0
        self.uniform_baseline_entropy_sum = 0.0
        self.total_samples = 0

    def add(self, sample, distribution, selected) -> None:
        self.total_samples += 1
        self.entropy_sum += -np.sum(distribution * np.log(distribution))
        self.uniform_baseline_entropy_sum += np.log(distribution.shape[0])  # == -log(1/N)

        rewrites = Counter(rewrite_type for rewrite_type, _ in sample["candidate_rewrite_metadata"])
        rewrites["NO_REWRITE"] = 1
        num_rewrites = sum(rewrites.values())

        for rewrite_type, rewrite_count in rewrites.items():
            self.available_rewrite_frequency[rewrite_type] += rewrite_count / num_rewrites

        for rewrite_idx in selected.keys():
            if rewrite_idx == "NO_BUG":
                rewrite_type = "NO_REWRITE"
            else:
                rewrite_type = sample["candidate_rewrite_metadata"][int(rewrite_idx)][0]
            self.selected_rewrite_frequency[rewrite_type] += 1.0 / len(selected)

    def report(self) -> Dict:
        entropy = self.entropy_sum / self.total_samples
        uniform_baseline_entropy = self.uniform_baseline_entropy_sum / self.total_samples
        print(f"Avg Entropy: {entropy:.3f}")
        print(f"Avg Uniform Entropy (Baseline): {uniform_baseline_entropy:.3f}")
        for rewrite_type in sorted(self.available_rewrite_frequency):
            print(
                f"{rewrite_type} {self.selected_rewrite_frequency[rewrite_type] / self.total_samples :.2%} "
                f"(in-data {self.available_rewrite_frequency[rewrite_type] / self.total_samples :.2%})"
            )
        self.reset()
        return {"entropy": entropy, "uniform_baseline_entropy": uniform_baseline_entropy}


def run(arguments):
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(arguments.bug_selector_server_address)

    # Start a random bug selector to keep things going until we have a model.
    terminate_random_bug_selector = Event()
    random_bug_selector_thread = Thread(
        target=lambda: random_bug_selector_server(
            arguments.bug_selector_server_address, terminate_random_bug_selector, context, socket
        ),
        name="random_bug_selector_thread",
    )
    random_bug_selector_thread.start()

    if args.fixed_model_path is not None:
        model_sync_client = MockModelSyncClient(args.fixed_model_path)
    else:
        model_sync_client = ModelSyncClient(args.model_sync_server, do_not_update_before_sec=5 * 60)
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, nn = model_sync_client.ask(model_device)
    nn.eval()
    LOGGER.info("Loaded a generator model from the training loop.")

    # Stop the random bug selector one the nn has been set.
    terminate_random_bug_selector.set()
    random_bug_selector_thread.join()

    # Set up metrics for the main model.
    incoming_message_counter = metric_provider.new_counter("incoming_messages")
    outgoing_message_counter = metric_provider.new_counter("outgoing_messages")
    processing_timer = metric_provider.new_latency_measure("processing_time")
    waiting_timer = metric_provider.new_latency_measure("incoming_latency")
    outgoing_timer = metric_provider.new_latency_measure("outgoing_latency")
    selection_distribution_entropy_gauge = metric_provider.new_gauge("selection_entropy")
    selection_distribution_uniform_entropy_gauge = metric_provider.new_gauge("selection_uniform_entropy")

    stats = BugSelectionStats()

    # Start serving on the trained model
    while True:
        model_sync_client.update_params_if_needed(nn, model_device)
        with waiting_timer:
            data = [msgpack.loads(socket.recv())]
        incoming_message_counter.inc()
        with processing_timer:
            try:
                predictions = model.predict(data, nn, model_device, parallelize=False)
                datapoint, location_logprobs, rewrite_logprobs = next(predictions)

                generator_rewrite_logprobs = []
                for rewrite_logprob, rewrite_reference_node_idx in zip(
                    rewrite_logprobs, datapoint["graph"]["reference_nodes"]
                ):
                    generator_rewrite_logprobs.append(rewrite_logprob + location_logprobs[rewrite_reference_node_idx])

                # Append the "NO_BUG" logprob.
                generator_rewrite_logprobs.append(location_logprobs[-1])

                rewrite_selection_distribution = calculate_selection_distribution(
                    logprobs=generator_rewrite_logprobs, temperature=arguments.temperature_scaling, epsilon=args.epsilon
                )
                candidate_rewrites = datapoint["candidate_rewrites"]
                selected_rewrites_idxs = np.random.choice(
                    range(len(candidate_rewrites) + 1),
                    size=arguments.num_rewrites_per_sample,
                    replace=False,
                    p=rewrite_selection_distribution,
                )
                selected_rewrites = {
                    "NO_BUG" if i == len(candidate_rewrites) else str(i): float(generator_rewrite_logprobs[i])
                    for i in selected_rewrites_idxs
                }
                # Since there is no computational cost, always include the NO_BUG
                if "NO_BUG" not in selected_rewrites:
                    selected_rewrites["NO_BUG"] = float(generator_rewrite_logprobs[len(candidate_rewrites)])
                stats.add(datapoint, rewrite_selection_distribution, selected_rewrites)

            except StopIteration:  # Emitted by predict
                selected_rewrites = select_random_rewrites(data[0]["candidate_rewrites"])

        with outgoing_timer:
            socket.send(msgpack.dumps(selected_rewrites))
        outgoing_message_counter.inc()

        if stats.total_samples % 200 == 199:
            summary = stats.report()
            selection_distribution_entropy_gauge.set(summary["entropy"])
            selection_distribution_uniform_entropy_gauge.set(summary["uniform_baseline_entropy"])
            print(json.dumps(summary, indent=2))
            stats = BugSelectionStats()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="A bug selection server.")

    parser.add_argument(
        "--bug-selector-server-address",
        type=str,
        default="tcp://*:5556",
        help="The address where this process will serve requests for selecting the bugs to be generators.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug on exception.",
    )

    parser.add_argument(
        "--model-sync-server",
        type=str,
        default="tcp://localhost:6001",
        help="The address to get updates on the generator model.",
    )

    parser.add_argument(
        "--prometheus-server-port",
        type=int,
        default=8003,
        help="The address to the prometheus server.",
    )

    parser.add_argument(
        "--num-rewrites-per-sample",
        type=int,
        default=4,
        help="The number of rewrites to sample for each data sample.",
    )

    parser.add_argument(
        "--enable-tracing",
        action="store_true",
        help="Set to enable recording tracing information.",
    )

    parser.add_argument(
        "--fixed-model-path",
        type=str,
        help="Use a fixed model, instead of asking for one from the server.",
    )

    parser.add_argument(
        "--temperature-scaling",
        type=float,
        help="The temperature used to scale the bug selection distribution. "
        "Higher temperature gives closer to uniform selection. [Default = 1.0]",
        default=1.0,
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        help="The epsilon-greedy used to use a fully random selection. [Default = 0.02]",
        default=0.02,
    )

    args = parser.parse_args()
    metric_provider.start_server(args.prometheus_server_port)
    metric_provider.set_tracing(args.enable_tracing)
    run_and_debug(lambda: run(args), args.debug)
