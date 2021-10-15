import time
import unittest
from collections import Counter
from tempfile import TemporaryDirectory
from threading import Thread

from buglab.utils.replaybuffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def test_all_elements_added_sequential(self):
        with TemporaryDirectory() as tmp_dir:
            ttl = 5
            num_elements = 20000
            buffer = ReplayBuffer(tmp_dir, ttl=ttl, chunk_size=10)
            for i in range(num_elements):
                buffer.add(i)

            c = Counter(buffer.iter_batch(num_elements * ttl))
            for i in range(num_elements):
                self.assertIn(i, c, f"Element {i} not present in the output.")
                self.assertEqual(c[i], 5, "Incorrect count of elements")

    def test_all_elements_added_parallel(self):
        with TemporaryDirectory() as tmp_dir:
            ttl = 5
            num_elements = 20000
            buffer = ReplayBuffer(tmp_dir, ttl=ttl, chunk_size=10)

            def add():
                time.sleep(1)
                for i in range(num_elements):
                    if i % 131 == 0:
                        time.sleep(0.01)
                    buffer.add(i)

            add_thread = Thread(target=add, daemon=True)

            def consumer():
                for i in range(num_elements * ttl):
                    yield from buffer.iter_batch(1)
                    if i % 231 == 0:
                        time.sleep(0.01)

            add_thread.start()
            c = Counter(consumer())
            for i in range(num_elements):
                self.assertIn(i, c, f"Element {i} not present in the output.")
                self.assertEqual(c[i], 5, "Incorrect count of elements")


if __name__ == "__main__":
    unittest.main()
