from __future__ import annotations

import argparse
import os.path
from argparse import ArgumentParser, Namespace
from typing import Optional, Callable

from fs import open_fs
from fs.base import FS
from fs.copy import copy_fs
from relic.core.cli import CliPluginGroup, _SubParsersAction, CliPlugin

from relic.sga.core.essencefs import EssenceFS

_SUCCESS = 0


class RelicSgaCli(CliPluginGroup):
    GROUP = "relic.cli.sga"

    def _create_parser(
            self, command_group: Optional[_SubParsersAction] = None
    ) -> ArgumentParser:
        if command_group is None:
            return ArgumentParser("sga")
        else:
            return command_group.add_parser("sga")


def _arg_exists_err(value: str) -> argparse.ArgumentTypeError:
    return argparse.ArgumentTypeError(f"The given path '{value}' does not exist!")


def _get_dir_type_validator(exists: bool) -> Callable[[str], str]:
    def _dir_type(path: str) -> str:
        path = os.path.abspath(path)
        if not os.path.exists(path):
            if exists:
                raise _arg_exists_err(path)
            else:
                return path

        if os.path.isdir(path):
            return path

        raise argparse.ArgumentTypeError(f"The given path '{path}' is not a directory!")

    return _dir_type


def _get_file_type_validator(exists: Optional[bool]) -> Callable[[str], str]:
    def _file_type(path: str) -> str:
        path = os.path.abspath(path)
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
        desc = """Unpack an SGA archive to the filesystem.
            If only one root is present in the SGA, '--merge' is implied.
            If multiple roots are in the SGA '--isolate' is implied. 
            Manually specify the flags to override this behaviour."""
        if command_group is None:
            parser = ArgumentParser("unpack", description=desc)
        else:
            parser = command_group.add_parser("unpack", description=desc)

        parser.add_argument(
            "src_sga",
            type=_get_file_type_validator(exists=True),
            help="Source SGA File",
        )
        parser.add_argument(
            "out_dir",
            type=_get_dir_type_validator(exists=False),
            help="Output Directory",
        )
        sga_root_flags = parser.add_mutually_exclusive_group()

        sga_root_flags.add_argument(
            "-m", "--merge",
            help="SGA roots will always write to the same folder; specified by out_dir",
            action="store_true",
        )
        sga_root_flags.add_argument(
            "-i", "--isolate",
            help="SGA roots will always write to separate folders, one per alias; located within out_dir",
            action="store_true",
        )

        return parser

    def command(self, ns: Namespace) -> Optional[int]:
        infile: str = ns.src_sga
        outdir: str = ns.out_dir
        merge: bool = ns.merge
        isolate: bool = ns.isolate

        print(vars(ns))
        print(f"Unpacking `{infile}`")

        def _callback(_1: FS, srcfile: str, _2: FS, _3: str) -> None:
            print(f"\t\tUnpacking File `{srcfile}`")

        if merge:  # we can short circuit the merge flag case
            copy_fs(
                f"sga://{infile}", f"osfs://{outdir}", on_copy=_callback, preserve_time=True
            )
            return _SUCCESS

        # we need to open the archive to 'isolate' or to determine if we implicit merge
        sga: EssenceFS
        with open_fs(infile, default_protocol="sga") as sga:
            roots = list(sga.iterate_fs())
            if not isolate and len(roots) == 1:
                copy_fs(
                    sga, f"osfs://{outdir}", on_copy=_callback, preserve_time=True
                )
                return _SUCCESS

            # Isolate or Implied Isolate
            with open_fs(outdir, writeable=True, create=True) as osfs:
                for alias, subfs in roots:
                    with osfs.makedir(alias, recreate=True) as osfs_subfs:
                        copy_fs(subfs, osfs_subfs, on_copy=_callback, preserve_time=True)

        return _SUCCESS


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

        return parser


