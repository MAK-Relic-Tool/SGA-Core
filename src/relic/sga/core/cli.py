from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Optional

from relic.core.cli import CliPluginGroup, _SubParsersAction, CliPlugin


class RelicSgaCli(CliPluginGroup):
    GROUP = "relic.cli.sga"

    def __init__(self, parent: _SubParsersAction, **kwargs):
        super().__init__(parent, **kwargs)

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

        # TODO populate parser

        return parser

    def command(self, ns: Namespace) -> Optional[int]:
        raise NotImplementedError


class RelicSgaPackCli(CliPlugin):

    def _create_parser(self, command_group: Optional[_SubParsersAction] = None) -> ArgumentParser:
        parser: ArgumentParser
        if command_group is None:
            parser = ArgumentParser("pack")
        else:
            parser = command_group.add_parser("pack")

        # TODO populate parser

        return parser

    def command(self, ns: Namespace) -> Optional[int]:
        raise NotImplementedError
