import argparse
import logging
import os
import random
from io import StringIO
from tempfile import TemporaryDirectory
from typing import Optional
from typing import Type, Any

import pytest
from relic.core import CLI
from relic.core.cli import CliPluginGroup, CliPlugin

from relic.sga.core import Version, MAGIC_WORD
from relic.sga.core.cli import (
    _get_path_validator,
    _get_dir_type_validator,
    _get_file_type_validator,
    RelicSgaCli,
    RelicSgaUnpackCli,
    RelicSgaInfoCli,
    RelicSgaTreeCli,
    RelicSgaVersionCli,
    RelicSgaListCli,
)
from relic.sga.core.serialization import VersionSerializer
from tests.dummy_essencefs import write_random_essencefs, register_randomfs_opener
from tests.util import TempFileHandle

EXISTS_FILE_PATH = __file__
EXISTS_FOLD_PATH = os.path.join(__file__, "..")
INVALID_DIR_PATH = os.path.join(__file__, "doesnotexist.txt")
NONEXIST_PATH = os.path.join(__file__, "..\\doesnotexist.txt")


@pytest.mark.parametrize(
    ["exists", "path", "should_fail"],
    [
        (True, EXISTS_FILE_PATH, False),
        (False, EXISTS_FILE_PATH, False),
        (True, EXISTS_FOLD_PATH, False),
        (False, EXISTS_FOLD_PATH, False),
        (True, NONEXIST_PATH, True),
        (False, NONEXIST_PATH, False),
        (True, INVALID_DIR_PATH, True),
        (False, INVALID_DIR_PATH, True),
    ],
)
def test_get_path_validator(exists: bool, path: str, should_fail: bool):
    validator = _get_path_validator(exists)
    try:
        validator(path)
    except argparse.ArgumentTypeError:
        if not should_fail:
            pytest.fail("Validator failed when it was expected to pass")
    else:
        if should_fail:
            pytest.fail("Validator passed when it was expected to fail")


@pytest.mark.parametrize(
    ["exists", "path", "should_fail"],
    [
        (True, EXISTS_FILE_PATH, True),
        (False, EXISTS_FILE_PATH, True),
        (True, EXISTS_FOLD_PATH, False),
        (False, EXISTS_FOLD_PATH, False),
        (True, NONEXIST_PATH, True),
        (False, NONEXIST_PATH, False),
        (True, INVALID_DIR_PATH, True),
        (False, INVALID_DIR_PATH, True),
    ],
)
def test_get_dir_type_validator(exists: bool, path: str, should_fail: bool):
    validator = _get_dir_type_validator(exists)
    try:
        validator(path)
    except argparse.ArgumentTypeError:
        if not should_fail:
            pytest.fail("Validator failed when it was expected to pass")
    else:
        if should_fail:
            pytest.fail("Validator passed when it was expected to fail")


@pytest.mark.parametrize(
    ["exists", "path", "should_fail"],
    [
        (True, EXISTS_FILE_PATH, False),
        (False, EXISTS_FILE_PATH, False),
        (True, EXISTS_FOLD_PATH, True),
        (False, EXISTS_FOLD_PATH, True),
        (True, NONEXIST_PATH, True),
        (False, NONEXIST_PATH, False),
        (True, INVALID_DIR_PATH, True),
        (False, INVALID_DIR_PATH, True),
    ],
)
def test_get_file_type_validator(exists: bool, path: str, should_fail: bool):
    validator = _get_file_type_validator(exists)
    try:
        validator(path)
    except argparse.ArgumentTypeError:
        if not should_fail:
            pytest.fail("Validator failed when it was expected to pass")
    else:
        if should_fail:
            pytest.fail("Validator passed when it was expected to fail")


@pytest.mark.parametrize(
    "cli",
    [
        RelicSgaCli,
        RelicSgaUnpackCli,
        RelicSgaInfoCli,
        RelicSgaTreeCli,
        RelicSgaVersionCli,
        RelicSgaListCli,
    ],
)
@pytest.mark.parametrize("parent", [True, False])
def test_init_cli(cli: Type[CliPlugin | CliPluginGroup], parent: bool):
    parent_parser: Optional[Any] = None
    if parent:
        parent_parser = argparse.ArgumentParser().add_subparsers()

    cli(parent=parent_parser)


def random_nums(a, b, count=1, seed: Optional[int] = None):
    if seed is not None:
        random.seed = seed
    for _ in range(count):
        yield random.randint(a, b)


SEEDS = [8675309, 20040920, 20250318, 500500]


@pytest.mark.parametrize("seed", SEEDS)
def test_cli_tree(seed: int):
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


@pytest.mark.parametrize("add_plugin", [True, False])
def test_cli_list_plugins(add_plugin: bool):
    from relic.sga.core.essencefs.opener import registry

    if add_plugin:
        register_randomfs_opener()
    else:
        for key in list(registry._backing.keys()):
            del registry._backing[key]

    with StringIO() as logFile:
        logging.basicConfig(
            stream=logFile, level=logging.DEBUG, format="%(message)s", force=True
        )
        logger = logging.getLogger()
        CLI.run_with("relic", "sga", "list", logger=logger)
        print("\nLOG:")
        result = logFile.getvalue()
        print(result)
        if add_plugin:
            assert "No Plugins Found" not in result
        else:
            assert "No Plugins Found" in result


@pytest.mark.parametrize("version", [Version(0), Version(1), Version(2)])
@pytest.mark.parametrize("write_magic", [True, False])
def test_cli_version(version: Version, write_magic: bool):
    with StringIO() as logFile:
        logging.basicConfig(
            stream=logFile, level=logging.DEBUG, format="%(message)s", force=True
        )
        logger = logging.getLogger()

        with TempFileHandle() as h:
            with h.open("wb") as w:
                if write_magic:
                    MAGIC_WORD.write(w)
                VersionSerializer.write(w, version)

            CLI.run_with("relic", "sga", "version", h.path, logger=logger)
        print("\nLOG:")
        result = logFile.getvalue()
        print(result)

        if not write_magic:
            assert "File is not an SGA" in result
        else:
            assert str(version) in result


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("merge_flag", [None, "-m", "--merge", "-i", "--isolate"])
def test_cli_unpack(seed: int, merge_flag: Optional[str]):
    register_randomfs_opener()
    with StringIO() as logFile:
        logging.basicConfig(
            stream=logFile, level=logging.DEBUG, format="%(message)s", force=True
        )
        logger = logging.getLogger()
        with TempFileHandle() as h:
            with h.open("wb") as w:
                write_random_essencefs(w, seed)
            with TemporaryDirectory() as d:
                args = ["relic", "sga", "unpack", h.path, d]
                if merge_flag is not None:
                    args.append(merge_flag)

                CLI.run_with(*args, logger=logger)
                print("\nLOG:")
                print(logFile.getvalue())
