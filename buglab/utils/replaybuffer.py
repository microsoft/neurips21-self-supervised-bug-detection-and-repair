import os
import time
from os import PathLike
from queue import Queue
from random import Random
from threading import RLock, Thread
from typing import Generic, Iterator, Optional, TypeVar

from prometheus_client import Gauge

from buglab.utils.msgpackutils import load_msgpack_l_gz, save_msgpack_l_gz

T = TypeVar("T")


class ReplayBuffer(Generic[T]):
    def __init__(
        self,
        backing_dir: PathLike,
        gauge: Optional[Gauge] = None,
        ttl: int = 5,
        chunk_size: int = 100,
        rng: Optional[Random] = None,
    ):
        self.__ttl = ttl  # Time to live for each new entry: how many times will it be yielded before "expiring"?

        ### Queues
        # Queue of incoming elements
        self.__incoming_queue = Queue(maxsize=10000)
        # Queue of outgoing elements
        self.__outgoing_queue = Queue(maxsize=10000)
        # Elements ready to be stored in the files
        self.__to_store_queue = Queue(maxsize=1000)

        self.__buffer_size_gauge = gauge
        self.__backing_dir = backing_dir
        self.__chunk_size = chunk_size

        self.__storage_chunks_lock = RLock()
        # The following three fields are protected by the lock above.
        self.__available_storage_chunks = set()
        self.__num_active_chunks = 0
        self.__used_storage_chunks = set()

        self.__rng = rng
        if self.__rng is None:
            self.__rng = Random()

        self._start_threads()

    @property
    def ttl(self) -> int:
        return self.__ttl

    def __file_path_for(self, idx: int) -> PathLike:
        return os.path.join(self.__backing_dir, f"buffer.{idx}.msgpack.l.gz")

    def _acquire_random_available_chunk_to_read(self) -> int:
        while True:
            with self.__storage_chunks_lock:
                if len(self.__used_storage_chunks) > 0:
                    chunk_idx = self.__rng.choice(list(self.__used_storage_chunks))
                    self.__used_storage_chunks.remove(chunk_idx)
                    return chunk_idx
            # No chunk is available to be read. Wait a bit.
            time.sleep(0.05)

    def _acquire_random_available_chunk_to_write(self) -> int:
        while True:
            with self.__storage_chunks_lock:
                if len(self.__available_storage_chunks) > 0:
                    return self.__available_storage_chunks.pop()
                else:
                    self.__num_active_chunks += 1
                    writable_chunk = self.__num_active_chunks
                    return writable_chunk

    def _buffer_output_reader(self):
        """
        Keep filling the consumer queue with elements from the buffer.
        """
        while True:
            chunk_idx = self._acquire_random_available_chunk_to_read()
            out = list(load_msgpack_l_gz(self.__file_path_for(chunk_idx)))
            self.__rng.shuffle(out)
            for element, ttl in out:
                self.__outgoing_queue.put(element)
                if ttl > 1:
                    ttl1 = ttl - 1
                    self.__to_store_queue.put((element, ttl1))
                elif self.__buffer_size_gauge is not None:
                    self.__buffer_size_gauge.dec()
            with self.__storage_chunks_lock:
                # Release chunk to be re-used.
                self.__available_storage_chunks.add(chunk_idx)

    def _buffer_input_output_thread(self) -> None:
        """
        Read from the buffer producer queue into the buffer.
        """
        while True:
            element = self.__incoming_queue.get()
            if self.__buffer_size_gauge is not None:
                self.__buffer_size_gauge.inc()
            self.__to_store_queue.put((element, self.__ttl))

    def _store_buffer_queue(self) -> None:
        while True:
            chunk_to_write = self._acquire_random_available_chunk_to_write()

            def write_chunk():
                for _ in range(self.__chunk_size):
                    yield self.__to_store_queue.get()

            save_msgpack_l_gz(write_chunk(), self.__file_path_for(chunk_to_write))

            with self.__storage_chunks_lock:
                # Record chunk as ready to be consumed.
                self.__used_storage_chunks.add(chunk_to_write)

    def _start_threads(self):
        buffer_input_output_thread = Thread(
            target=self._buffer_input_output_thread,
            daemon=True,
            name="buffer_input_output_thread",
        )
        buffer_input_output_thread.start()

        buffer_output_from_storage_thread = Thread(
            target=self._buffer_output_reader,
            daemon=True,
            name="buffer_output_from_storage_thread",
        )
        buffer_output_from_storage_thread.start()

        storage_buffer_queue = Thread(
            target=self._store_buffer_queue,
            daemon=True,
            name="storage_buffer_queue_thread",
        )
        storage_buffer_queue.start()

    def add(self, element: T) -> None:
        """Thread-safe adding to the queue."""
        self.__incoming_queue.put(element)

    def iter_batch(self, max_num_elements: int) -> Iterator[T]:
        for _ in range(max_num_elements):
            yield self.__outgoing_queue.get()
