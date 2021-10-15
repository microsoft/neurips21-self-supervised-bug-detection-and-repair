import argparse
from typing import Dict, Optional

import msgpack
import zmq
from tqdm import tqdm

from buglab.utils.logging import configure_logging

if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Dummy bug selector client. Used mainly for debugging and observing the bug selector server."
    )

    parser.add_argument(
        "--data-generating-pipeline-address",
        type=str,
        default="tcp://localhost:5558",
        help="The zmq address to the data generating pipeline.",
    )

    parser.add_argument(
        "--bug-selector-address",
        type=str,
        default="tcp://localhost:5556",
        help="The zmq address to the bug selector server.",
    )

    args = parser.parse_args()

    context = zmq.Context.instance()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(args.data_generating_pipeline_address)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    def msg_yielder():
        while True:
            yield msgpack.loads(subscriber.recv())

    bug_selector = context.socket(zmq.REQ)
    bug_selector.connect(args.bug_selector_address)
    for data in tqdm(msg_yielder()):
        bug_selector.send(msgpack.dumps(data["original"]))
        selected_rewrites: Dict[Optional[str], float] = msgpack.loads(bug_selector.recv())
