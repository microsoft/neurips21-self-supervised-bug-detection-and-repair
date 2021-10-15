import gzip
import logging
import pickle
import time
from pathlib import Path
from typing import Set

import msgpack
import zmq
from datasketch import MinHash, MinHashLSH

LOGGER = logging.getLogger(__name__)


class DuplicationIndex:
    """A simple duplication index that checks for an overlap on the tokens of indexed sets."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        duplication_jaccard_threshold: float = 0.85,
        num_perm: int = 256,
        min_num_tokens: int = 10
    ):
        self.__duplication_jaccard_threshold = duplication_jaccard_threshold
        self.__num_perm = num_perm
        self.__min_num_tokens = min_num_tokens
        self.__checkpoint_path = checkpoint_path
        self.clear()

    def check_if_duplicate_and_add(self, filename: str, tokens: Set[str]) -> bool:
        if len(tokens) < self.__min_num_tokens:
            return False
        min_hash = MinHash(num_perm=self.__num_perm)
        for token in tokens:
            min_hash.update(token.encode())

        close_duplicates = self.__index.query(min_hash)
        if filename in self.__index.keys:
            LOGGER.warning("Duplicate key %s", filename)
            return True
        self.__index.insert(filename, min_hash)
        if len(close_duplicates) > 0:
            LOGGER.info("`%s` duplicate of: %s", filename, close_duplicates)
        return len(close_duplicates) > 0

    def server(self, address: str = "tcp://*:5555", save_state_every_sec: int = 5 * 60):
        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.bind(address)

        last_checkpoint = time.time()
        while True:
            r_bytes = socket.recv()
            filename, tokens = msgpack.loads(r_bytes)
            is_duplicate = self.check_if_duplicate_and_add(filename, set(tokens))
            socket.send(bytes(is_duplicate))
            if time.time() - last_checkpoint >= save_state_every_sec:
                LOGGER.info("Creating duplication index checkpoint.")
                self.__save_state()
                last_checkpoint = time.time()

    def __save_state(self) -> None:
        with gzip.GzipFile(self.__checkpoint_path, "wb") as outfile:
            pickle.dump(self, outfile)

    def clear(self) -> None:
        """Clear the index."""
        self.__index = MinHashLSH(threshold=self.__duplication_jaccard_threshold, num_perm=self.__num_perm)
