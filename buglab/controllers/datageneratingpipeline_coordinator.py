import argparse
import logging
import random
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Thread

import zmq

from buglab.data.deduplication import DuplicationIndex
from buglab.utils.logging import MetricProvider, configure_logging

LOGGER = logging.getLogger(__name__)
metric_provider = MetricProvider("DataGeneratingPipelineCoordinator")


def data_pipeline_proxy():
    # This follows http://zguide.zeromq.org/py:chapter2#The-Dynamic-Discovery-Problem
    context = zmq.Context.instance()

    # Socket facing producers
    frontend = context.socket(zmq.XPUB)
    frontend.bind("tcp://*:5558")

    # Socket facing consumers
    backend = context.socket(zmq.XSUB)
    backend.bind("tcp://*:5557")

    zmq.proxy(frontend, backend)

    raise Exception("Should never get here.")


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Orchestrator to extract graphs and publish data.")
    parser.add_argument(
        "package_list_path",
        type=str,
        help="the path to a txt file containing the names of the packages to be considered",
    )

    parser.add_argument(
        "--prometheus-server-port",
        type=int,
        default=8004,
        help="The port where Prometheus metrics can be accessed.",
    )

    parser.add_argument(
        "--enable-tracing",
        action="store_true",
        help="Set to enable recording tracing information.",
    )

    parser.add_argument(
        "--work-distribution-server-port",
        type=int,
        default=5550,
        help="Work distribution port.",
    )

    args = parser.parse_args()
    LOGGER.info("Run args: %s", args)
    metric_provider.start_server(args.prometheus_server_port)
    metric_provider.set_tracing(args.enable_tracing)

    proxy_thread = Thread(target=data_pipeline_proxy, name="data_publishing_proxy", daemon=True)
    proxy_thread.start()

    f = NamedTemporaryFile()
    duplication_index = DuplicationIndex(Path(f.name))
    duplication_server_thread = Thread(target=lambda: duplication_index.server(address="tcp://*:5555"), daemon=True)
    duplication_server_thread.start()

    all_packages = []
    with open(args.package_list_path) as f:
        for line in f.readlines():
            pkg_name = line.strip()
            if len(pkg_name) > 0:
                all_packages.append(pkg_name)

    package_counter = metric_provider.new_counter("published_packages")
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.work_distribution_server_port}")

    while True:  # Keep publishing forever
        random.shuffle(all_packages)
        for package in all_packages:
            worker_id = socket.recv_string()
            LOGGER.info(f"Worker `{worker_id}` asked for the next package to process. Sending `{package}`.")
            socket.send_string(package)
            package_counter.inc()

        duplication_index.clear()  # Reset duplication index
        LOGGER.info(f"All packages have been distributed. Restarting from scratch.")
