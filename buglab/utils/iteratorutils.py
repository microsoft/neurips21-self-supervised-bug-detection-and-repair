from itertools import islice
from queue import Queue
from typing import Iterator

import numpy as np


def limited_queue_iterator(queue: Queue, max_num_elements: int) -> Iterator:
    """Construct an iterator from a queue. The iterator will stop after max_num_elements."""
    for _ in range(max_num_elements):
        yield queue.get()


def sampled_iterator(input_iter, num_elements: int, sampling_rate: float):
    if sampling_rate == 1.0:
        yield from islice(input_iter, num_elements)
    else:
        num_taken = 0
        for element in input_iter:
            if np.random.rand() < sampling_rate:
                yield element
                num_taken += 1
                if num_taken >= num_elements:
                    break
