import os
import time
import logging
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def _log_time_usage(prefix=""):
    """log the time usage in a code block
    prefix: the prefix text to show
    """
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        elapsed_seconds = float("%.2f" % (end - start))
        logging.info("%s: elapsed seconds: %s", prefix, elapsed_seconds)


def get_cache_dir():
    custom_dir = os.environ.get("ATOMIC_CACHE")
    if custom_dir is not None and custom_dir != "":
        return custom_dir
    return Path(Path.home(), ".cache", "atomic")

def get_cache_dir():
    custom_dir = os.environ.get("ATOMIC_CACHE")
    if custom_dir is not None and custom_dir != '':
        return custom_dir
    return Path(Path.home(), '.cache', "atomic")