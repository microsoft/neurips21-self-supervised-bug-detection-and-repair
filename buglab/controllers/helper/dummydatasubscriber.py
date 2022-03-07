import argparse
import logging
import msgpack
import zmq
from dpu_utils.utils import ChunkWriter
from tqdm import tqdm

from buglab.utils.loggingutils import configure_logging

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

    parser.add_argument(
        "--store-data-at",
        type=str,
        default=None,
        help="Store the data in the given folder.",
    )

    args = parser.parse_args()

    context = zmq.Context.instance()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(args.data_generating_pipeline_address)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    def msg_yielder():
        while True:
            yield msgpack.loads(subscriber.recv())

    if args.store_data_at is not None:
        writer = ChunkWriter(args.store_data_at, "graphs-", 5000, ".msgpack.l.gz", mode="a")
    else:
        writer = None

    for data in tqdm(msg_yielder()):
        if writer is not None:
            for g, _ in data["rewrites"].values():
                writer.add(g)
