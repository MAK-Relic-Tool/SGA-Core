r"""TestCases for more explicit errors when providing invalid path arguments.

https://github.com/MAK-Relic-Tool/Issue-Tracker/issues/40
"""

import os
from argparse import ArgumentError
from typing import Iterable

import pytest


def _ArgumentError(name, message):
    _ = ArgumentError(None, message)
    _.argument_name = name
    return _


_ARGS = [
    (
        ["sga", "unpack", "nonexistant.sga", "."],
        _ArgumentError(
            "src_sga",
            f"The given path '{os.path.abspath('nonexistant.sga')}' does not exist!",
        ),
    ),
    (
        ["sga", "unpack", __file__, __file__],
        _ArgumentError("out_dir", f"The given path '{__file__}' is not a directory!"),
    ),
]


@pytest.mark.parametrize(["args", "expected"], _ARGS)
def test_argparse_error(args: Iterable[str], expected: ArgumentError):
    from relic.core import CLI

    try:
        _ = CLI.run_with(*args)
    except ArgumentError as arg_err:
        assert arg_err.argument_name == expected.argument_name, (
            arg_err.argument_name,
            expected.argument_name,
        )
        assert arg_err.message == expected.message, (arg_err.message, expected.message)
    except Exception as exc:
        assert False, str(exc)
    else:
        assert False, str("Did not error!")
