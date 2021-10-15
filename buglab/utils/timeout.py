from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


def call_with_timeout(fn: Callable[[], Any], *, timout_sec: float):
    """Run fn within timeout_sec. Otherwise raise TimeoutError"""
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(fn)
        return future.result(timeout=timout_sec)
