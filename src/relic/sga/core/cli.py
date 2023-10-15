from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Optional

import fs.copy
from fs.base import FS
from relic.core.cli import CliPluginGroup, _SubParsersAction, CliPlugin


class RelicSgaCli(CliPluginGroup):
    GROUP = "relic.cli.sga"

    def _create_parser(self, command_group: Optional[_SubParsersAction] = None) -> ArgumentParser:
        if command_group is None:
            return ArgumentParser("sga")
        else:
            return command_group.add_parser("sga")


class RelicSgaUnpackCli(CliPlugin):

    def _create_parser(self, command_group: Optional[_SubParsersAction] = None) -> ArgumentParser:
        parser: ArgumentParser
        if command_group is None:
            parser = ArgumentParser("unpack")
        else:
            parser = command_group.add_parser("unpack")

        parser.add_argument("src_sga", type=str, help="Source SGA File")
        parser.add_argument("out_dir", type=str, help="Output Directory")

        return parser

    def command(self, ns: Namespace) -> Optional[int]:
        infile: str = ns.src_sga
        outdir: str = ns.out_dir

        print(f"Unpacking `{infile}`")

        def _callback(_1: FS, srcfile: str, _2: FS, _3: str):
            print(f"\t\tUnpacking File `{srcfile}`")

        fs.copy.copy_fs(f"sga://{infile}", f"osfs://{outdir}", on_copy=_callback)

        return None  # To shut-up mypy


class RelicSgaPackCli(CliPluginGroup):
    GROUP = "relic.cli.sga.pack"

    def _create_parser(self, command_group: Optional[_SubParsersAction] = None) -> ArgumentParser:
        parser: ArgumentParser
        if command_group is None:
            parser = ArgumentParser("pack")
        else:
            parser = command_group.add_parser("pack")

        # pack further delegates to version plugins

        return parser

