import io
import subprocess
# Local testing requires running `pip install -e "."`
from contextlib import redirect_stdout
from typing import Sequence

import pytest


class CommandTests:
    def test_run(self, args: Sequence[str], output: str, exit_code: int):
        _args = ["relic", *args]
        cmd = subprocess.run(_args, capture_output=True, text=True)
        result = cmd.stdout
        status = cmd.returncode
        print(f"'{result}'")  # Visual Aid for Debugging
        assert output in result
        assert status == exit_code

    def test_run_with(self, args: Sequence[str], output: str, exit_code: int):
        from relic.core.cli import cli_root
        with io.StringIO() as f:
            with redirect_stdout(f):
                status = cli_root.run_with(*args)
            f.seek(0)
            result = f.read()
            print(f"'{result}'")  # Visual Aid for Debugging
            assert output in result
            assert status == exit_code


_SGA_HELP = ["sga", "-h"], """usage: relic sga [-h] {pack,unpack} ...""", 0
_SGA_PACK_HELP = ["sga", "pack", "-h"], """usage: relic sga pack [-h] {} ...""", 0
_SGA_UNPACK_HELP = ["sga", "unpack", "-h"], """usage: relic sga unpack [-h]""", 0

_TESTS = [_SGA_HELP, _SGA_PACK_HELP, _SGA_UNPACK_HELP]
_TEST_IDS = [' '.join(_[0]) for _ in _TESTS]


@pytest.mark.parametrize(["args", "output", "exit_code"], _TESTS, ids=_TEST_IDS)
class TestRelicSgaCli(CommandTests):
    ...
