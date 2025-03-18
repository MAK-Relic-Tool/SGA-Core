import logging
import os
import random
import tempfile
from io import StringIO
from typing import Optional

import pytest
from relic.core import CLI

from dummy_essencefs import write_random_essencefs, register_randomfs_opener


def random_nums(a, b, count=1, seed: Optional[int] = None):
    if seed is not None:
        random.seed = seed
    for _ in range(count):
        yield random.randint(a, b)


class TempFileHandle:
    def __init__(self):
        with tempfile.NamedTemporaryFile("x", delete=False) as h:
            self._filename = h.name

    @property
    def path(self):
        return self._filename

    def open(self, mode: str):
        return open(self._filename, mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            os.unlink(self._filename)
        except Exception as e:
            print(e)


SEEDS = random_nums(0, 8675309, 8, seed="DEADBEEF".__hash__())


@pytest.mark.parametrize(["seed"], list((v,) for v in SEEDS))
def test_tree(seed: int):
    with StringIO() as logFile:
        logging.basicConfig(
            stream=logFile, level=logging.DEBUG, format="%(message)s", force=True
        )
        logger = logging.getLogger()

        register_randomfs_opener()
        with TempFileHandle() as h:
            with h.open("wb") as w:
                write_random_essencefs(w, seed)
            CLI.run_with("relic", "sga", "tree", h.path, logger=logger)
        print("\nLOG:")
        print(logFile.getvalue())
