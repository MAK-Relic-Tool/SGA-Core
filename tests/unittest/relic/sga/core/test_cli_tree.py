import logging
import random
from io import StringIO
from typing import Optional

import pytest
from relic.core import CLI

from tests.dummy_essencefs import write_random_essencefs, register_randomfs_opener
from tests.util import TempFileHandle


def random_nums(a, b, count=1, seed: Optional[int] = None):
    if seed is not None:
        random.seed = seed
    for _ in range(count):
        yield random.randint(a, b)


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
