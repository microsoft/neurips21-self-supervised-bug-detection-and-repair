import argparse
import logging
import socket as s

import msgpack
import zmq

from buglab.controllers.buggydatacreation import extract_for_package
from buglab.data.deduplication import DuplicationClient
from buglab.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Orchestrator to extract (static graphs) across multiple packages.")
    parser.add_argument("--debug", action="store_true", help="Enter debugging mode when an exception is thrown.")
    parser.add_argument(
        "--deduplication-server",
        type=str,
        default="tcp://localhost:5555",
        help="The zmq address to the deduplication server.",
    )
    parser.add_argument(
        "--bug-selector-server",
        type=str,
        default="tcp://localhost:5556",
        help="The zmq address to the bug selector server.",
    )
    parser.add_argument(
        "--data-publishing-proxy-address",
        type=str,
        default="tcp://localhost:5557",
        help="The zmq address to publish the extracted data.",
    )

    parser.add_argument(
        "--push-gateway-address",
        type=str,
        default=None,
        help="The address of the Prometheus push gateway.",
    )

    parser.add_argument(
        "--work-coordinator-address",
        type=str,
        default="tcp://localhost:5550",
        help="The address of the Prometheus push gateway.",
    )

    parser.add_argument(
        "--num-semantics-preserving-transforms",
        type=int,
        default=1,
        help="The number of semantics-preserving transformation per input file.",
    )

    args = parser.parse_args()

    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.connect(args.data_publishing_proxy_address)

    work_receive_socket = context.socket(zmq.REQ)
    work_receive_socket.connect(args.work_coordinator_address)

    LOGGER.info(f"Asking {args.work_coordinator_address} for work...")
    host_name = s.gethostname()
    work_receive_socket.send_string(f"Worker {host_name}")
    package = work_receive_socket.recv_string()

    extacted_data_iter = extract_for_package(
        package,
        args.bug_selector_server,
        args.push_gateway_address,
        DuplicationClient(args.deduplication_server),
        num_semantics_preserving_transformations_per_file=args.num_semantics_preserving_transforms,
    )
    for extracted_function_code in extacted_data_iter:
        socket.send(msgpack.dumps(extracted_function_code))
