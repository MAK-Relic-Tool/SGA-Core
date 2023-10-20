from __future__ import annotations

import argparse
import datetime
import os.path
from argparse import ArgumentParser, Namespace
from typing import Optional, Callable, Dict, List, Any, Tuple, Set, TextIO

import fs.copy
from fs.base import FS
from fs.multifs import MultiFS
from relic.core.cli import CliPluginGroup, _SubParsersAction, CliPlugin

from relic.sga.core.definitions import StorageType
from relic.sga.core.filesystem import EssenceFS, _EssenceDriveFS


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
            parser = command_group.add_parser("repack", description=desc)

        # pack further delegates to version plugins

        return parser


class RelicSgaInfoCli(CliPlugin):
    def _create_parser(
        self, command_group: Optional[_SubParsersAction] = None
    ) -> ArgumentParser:
        parser: ArgumentParser
        description = "Dumps metadata packed into an SGA file."
        if command_group is None:
            parser = ArgumentParser("info", description=description)
        else:
            parser = command_group.add_parser("info", description=description)

        parser.add_argument(
            "sga",
            type=_get_file_type_validator(exists=True),
            help="SGA File to inspect",
        )
        parser.add_argument(
            "log_file",
            nargs="?",
            type=_get_file_type_validator(exists=False),
            help="Optional file to write messages to",
            default=None,
        )
        parser.add_argument("-q", "--quiet", action="store_true", default=False)
        return parser

    def command(self, ns: Namespace) -> Optional[int]:
        sga: str = ns.sga
        log_file: str = ns.log_file
        quiet: bool = ns.quiet

        logger: Optional[TextIO] = None
        try:
            if log_file is not None:
                logger = open(log_file, "w")

            outputs: List[Optional[TextIO]] = []
            if quiet is False:
                outputs.append(None)  # None is a sentinel for stdout
            if logger is not None:
                outputs.append(logger)

            def _print(
                *msg: str, sep: Optional[str] = None, end: Optional[str] = None
            ) -> None:
                for output in outputs:
                    print(*msg, sep=sep, end=end, file=output)

            def _is_container(d: Any) -> bool:
                return isinstance(d, (Dict, List, Tuple, Set))  # type: ignore

            def _stringify(d: Any, indent: int = 0) -> None:
                _TAB = "\t"
                if isinstance(d, Dict):
                    for k, v in d.items():
                        if _is_container(v):
                            _print(f"{_TAB * indent}{k}:")
                            _stringify(v, indent + 1)
                        else:
                            _print(f"{_TAB * indent}{k}: {v}")
                elif isinstance(d, (List, Tuple, Set)):  # type: ignore
                    _print(f"{_TAB * indent}{', '.join(*d)}")
                else:
                    _print(f"{_TAB * indent}{d}")

            def _getessence(fs: FS, path: str = "/") -> Dict[str, Any]:
                return fs.getinfo(path, "essence").raw.get("essence", {})  # type: ignore

            _print(f"File: `{sga}`")
            sgafs: EssenceFS
            with fs.open_fs(f"sga://{sga}") as sgafs:  # type: ignore
                _print("Archive Metadata:")
                _stringify(sgafs.getmeta("essence"), indent=1)

                drive: _EssenceDriveFS
                for alias, drive in sgafs.iterate_fs():  # type: ignore
                    _print(f"Drive: `{drive.name}` (`{drive.alias}`)")
                    _print("\tDrive Metadata:")
                    info = _getessence(drive)
                    if len(info) > 0:
                        _stringify(info, indent=2)
                    else:
                        _print(f"\t\tNo Metadata")

                    _print("\tDrive Files Metadata:")
                    for f in drive.walk.files():
                        _print(f"\t\t`{f}`:")
                        finfo: Dict[str, Any] = _getessence(drive, f)
                        finfo = finfo.copy()
                        # We alter storage_type cause it *should* always be present, if its not, we dont do anything
                        key = "storage_type"
                        if key in finfo:
                            v: int = finfo[key]
                            st: StorageType = StorageType(v)
                            finfo[key] = f"{v} ({st.name})"

                        # We alter modified too, cause when it is present, its garbage
                        key = "modified"
                        if key in finfo:
                            v: int = finfo[key]
                            mt = datetime.datetime.fromtimestamp(
                                v, datetime.timezone.utc
                            )
                            finfo[key] = str(mt)

                        # And CRC32 if it's in bytes; this should be removed ASAP tho # I only put this in because its such a minor patch to V2
                        key = "crc32"
                        if key in finfo:
                            v: bytes = finfo[key]
                            if isinstance(v, bytes):
                                crc32 = int.from_bytes(v, "little", signed=False)
                                finfo[key] = crc32

                        if len(finfo) > 0:
                            _stringify(finfo, indent=3)
                        else:
                            _print(f"\t\t\tNo Metadata")

        finally:
            if logger is not None:
                logger.close()

        if log_file is not None:
            print(
                f"Saved to `{os.path.join(os.getcwd(), log_file)}`"
            )  # DO NOT USE _PRINT
        return None
