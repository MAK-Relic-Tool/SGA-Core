r"""
TestCases for more explicit errors when providing invalid path arguments.
https://github.com/MAK-Relic-Tool/Issue-Tracker/issues/40
"""

import io
import os
from typing import Iterable
from contextlib import redirect_stderr

import pytest

_ARGS = [
    (
        ["sga", "unpack", "nonexistant.sga", "."],
        f"error: argument src_sga: The given path '{os.path.abspath('nonexistant.sga')}' does not exist!",
    ),
    (
        ["sga", "unpack", __file__, __file__],
        rf"error: argument out_dir: The given path '{__file__}' is not a directory!",
    ),
]


@pytest.mark.parametrize(["args", "msg"], _ARGS)
def test_argparse_error(args: Iterable[str], msg: str):
    from relic.core import CLI

    with io.StringIO() as f:
        with redirect_stderr(f):
            status = CLI.run_with(*args)
            assert status == 2
        f.seek(0)
        err = f.read()
        print(err)
        assert msg in err
