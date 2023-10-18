from __future__ import annotations

import argparse
import os.path
from argparse import ArgumentParser, Namespace
from typing import Optional

import fs.copy
from fs.base import FS
from relic.core.cli import CliPluginGroup, _SubParsersAction, CliPlugin


class RelicSgaCli(CliPluginGroup):
    GROUP = "relic.cli.sga"

    def _create_parser(
            self, command_group: Optional[_SubParsersAction] = None
    ) -> ArgumentParser:
        if command_group is None:
            return ArgumentParser("sga")
        else:
            return command_group.add_parser("sga")



def _arg_exists_err( value ):
    return argparse.ArgumentTypeError(f"The given path '{value}' does not exist!")


def _get_dir_type_validator(exists:bool):
    def _dir_type(path: str):
        if not os.path.exists(path):
            if exists:
                raise _arg_exists_err(path)
            else:
                return path

        if os.path.isdir(path):
            return path

        raise argparse.ArgumentTypeError(f"The given path '{path}' is not a directory!")

    return _dir_type


def _get_file_type_validator(exists:Optional[bool]):
    def _file_type(path: str):
        if not os.path.exists(path):
            if exists:
                raise _arg_exists_err(path)
            else:
                return path

        if os.path.isfile(path):
            return path

        raise argparse.ArgumentTypeError(f"The given path '{path}' is not a file!")

    return _file_type


class RelicSgaUnpackCli(CliPlugin):
    def _create_parser(
            self, command_group: Optional[_SubParsersAction] = None
    ) -> ArgumentParser:
        parser: ArgumentParser
        if command_group is None:
            parser = ArgumentParser("unpack")
        else:
            parser = command_group.add_parser("unpack")

        parser.add_argument("src_sga", type=_get_file_type_validator(exists=True), help="Source SGA File")
        parser.add_argument("out_dir", type=_get_dir_type_validator(exists=False), help="Output Directory")

        return parser

    def command(self, ns: Namespace) -> Optional[int]:
        infile: str = ns.src_sga
        outdir: str = ns.out_dir

        print(f"Unpacking `{infile}`")

        def _callback(_1: FS, srcfile: str, _2: FS, _3: str) -> None:
            print(f"\t\tUnpacking File `{srcfile}`")

        fs.copy.copy_fs(f"sga://{infile}", f"osfs://{outdir}", on_copy=_callback)

        return None  # To shut-up mypy


class RelicSgaPackCli(CliPluginGroup):
    GROUP = "relic.cli.sga.pack"

    def _create_parser(
            self, command_group: Optional[_SubParsersAction] = None
    ) -> ArgumentParser:
        parser: ArgumentParser
        if command_group is None:
            parser = ArgumentParser("pack")
        else:
            parser = command_group.add_parser("pack")

        # pack further delegates to version plugins

        return parser


class RelicSgaRepackCli(CliPluginGroup):
    """An alternative to pack which 'repacks' an SGA. Intended for testing purposes."""

    GROUP = "relic.cli.sga.repack"

    def _create_parser(
            self, command_group: Optional[_SubParsersAction] = None
    ) -> ArgumentParser:
        parser: ArgumentParser
        desc = "Debug Command; reads and repacks an SGA archive."
        if command_group is None:
            parser = ArgumentParser("repack", description=desc)
        else:
            parser = command_group.add_parser("repack", description = desc)

        # pack further delegates to version plugins

        return parser
