import argparse
import logging

import msgpack
import zmq
from tqdm import tqdm

from buglab.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Dummy data subscriber. Used mainly for debugging and observing the data generating pipeline."
    )

    parser.add_argument(
        "--data-generating-pipeline-address",
        type=str,
        default="tcp://localhost:5558",
        help="The zmq address to the data generating pipeline.",
    )
    args = parser.parse_args()

    context = zmq.Context.instance()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(args.data_generating_pipeline_address)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    def msg_yielder():
        while True:
            yield msgpack.loads(subscriber.recv())

    for _ in tqdm(msg_yielder()):
        LOGGER.info("Got data.")
        pass
