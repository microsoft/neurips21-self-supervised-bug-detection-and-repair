from typing import Optional

import logging
import msgpack
import random
import sys
import zmq
from threading import Event

from buglab.utils.loggingutils import configure_logging

LOGGER = logging.getLogger(__name__)


def random_bug_selector_server(
    address: str,
    terminate_signal: Optional[Event] = None,
    context: Optional[zmq.Context] = None,
    socket: Optional[zmq.Socket] = None,
):
    if context is None:
        context = zmq.Context.instance()
    if socket is None:
        socket = context.socket(zmq.REP)
        socket.bind(address)

    if terminate_signal is None:
        terminate_signal = Event()

    while not terminate_signal.is_set():
        serialized_data_sample = msgpack.loads(socket.recv())
        LOGGER.info("Received Bug Selection Request")

        all_candidate_rewrites = serialized_data_sample["candidate_rewrites"]
        selected_rewrites = select_random_rewrites(all_candidate_rewrites)

        socket.send(msgpack.dumps(selected_rewrites))


def select_random_rewrites(all_candidate_rewrites, num_rewrites=4):
    random_prob = 1 / (len(all_candidate_rewrites) + 1)
    selected_rewrites = {"NO_BUG": random_prob}
    for selected_idx in random.sample(range(len(all_candidate_rewrites)), k=num_rewrites):
        selected_rewrites[str(selected_idx)] = random_prob
    return selected_rewrites


if __name__ == "__main__":
    configure_logging()
    address = sys.argv[1]
    random_bug_selector_server(address)
