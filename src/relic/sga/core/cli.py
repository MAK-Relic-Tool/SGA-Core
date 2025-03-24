from __future__ import annotations

import dataclasses
import json
import logging
import os.path
from argparse import ArgumentParser, Namespace
from io import StringIO
from json import JSONEncoder
from logging import Logger
from typing import Optional, Any, Dict, Generator

import relic.core.cli
from fs import open_fs
from fs.base import FS
from fs.copy import copy_fs
from relic.core.cli import CliPluginGroup, _SubParsersAction, CliPlugin, RelicArgParser

from relic.sga.core.definitions import MAGIC_WORD
from relic.sga.core.essencefs import EssenceFS
from relic.sga.core.essencefs.opener import registry as sga_registry
from relic.sga.core.serialization import VersionSerializer
from relic.core.logmsg import BraceMessage
from relic.core.cli import (
    get_file_type_validator,
    get_dir_type_validator,
    get_path_validator,
)
from relic.core.errors import MagicMismatchError
from relic.sga.core.errors import VersionNotSupportedError
from contextlib import contextmanager

_SUCCESS = 0
_SGA_NOT_SUPPORTED = 1
_FILE_NOT_SGA = 2


def handle_version_not_supported_error(
    logger: logging.Logger, src: str, err: VersionNotSupportedError
) -> None:
    logger.debug(err, exc_info=True)
    logger.info(BraceMessage("Failed to open '{0}'", src))
    logger.info(BraceMessage("No SGA plugin found for '{0}'", err.received))
    logger.info("SGA Plugins:")
    for plugin in err.allowed:
        logger.info(BraceMessage("\t{0}", plugin))
    if len(err.allowed) == 0:
        logger.info("\tNone Found")


def handle_sga_magic_mismatch_error(
    logger: logging.Logger, src: str, err: MagicMismatchError
) -> None:
    logger.debug(err, exc_info=True)
    logger.info(BraceMessage("Failed to open '{0}'", src))
    logger.info("File is not an SGA")

    class PrettyBytesRepr:
        def __init__(self, v: bytes):
            self.v = v

        def __str__(self):
            return repr(self.v)[2:-1]  # trim b'' out of string

        def __repr__(self):
            return str(self)

    logger.info(
        BraceMessage(
            "Expected first {1} bytes to be '{0}', got '{2}'",
            PrettyBytesRepr(err.expected),
            len(err.expected),
            PrettyBytesRepr(err.received),
        )
    )


@contextmanager
def cli_open_sga(
    src: str, logger: logging.Logger, *, default_protocol: str = "sga"
) -> Generator[EssenceFS, None, None]:
    # Exception will cause a SystemExit:
    # run will cause the program to close
    # run_with will capture status code and continute
    try:
        with open_fs(src, default_protocol=default_protocol) as sga:
            yield sga  # type: ignore
    except VersionNotSupportedError as not_supported_error:
        handle_version_not_supported_error(logger, src, not_supported_error)

        raise SystemExit(_SGA_NOT_SUPPORTED) from not_supported_error
    except MagicMismatchError as mismatch_error:
        handle_sga_magic_mismatch_error(logger, src, mismatch_error)
        raise SystemExit(_FILE_NOT_SGA) from mismatch_error


class RelicSgaCli(CliPluginGroup):
    GROUP = "relic.cli.sga"

    def _create_parser(
        self, command_group: Optional[_SubParsersAction] = None
    ) -> ArgumentParser:
        name = "sga"
        if command_group is None:
            return RelicArgParser(name)
        return command_group.add_parser(name)


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
            parser = RelicArgParser("unpack", description=desc)
        else:
            parser = command_group.add_parser("unpack", description=desc)

        parser.add_argument(
            "src_sga",
            type=get_file_type_validator(exists=True),
            help="Source SGA File",
        )
        parser.add_argument(
            "out_dir",
            type=get_dir_type_validator(exists=False),
            help="Output Directory",
        )
        sga_root_flags = parser.add_mutually_exclusive_group()

        sga_root_flags.add_argument(
            "-m",
            "--merge",
            help="SGA roots will always write to the same folder; specified by out_dir",
            action="store_true",
        )
        sga_root_flags.add_argument(
            "-i",
            "--isolate",
            help="SGA roots will always write to separate folders, one per alias; located within out_dir",
            action="store_true",
        )

        return parser

    def command(self, ns: Namespace, *, logger: Logger) -> Optional[int]:
        infile: str = ns.src_sga
        outdir: str = ns.out_dir
        merge: bool = ns.merge
        isolate: bool = ns.isolate

        if merge and isolate:  # pragma: nocover
            # This error should be impossible
            raise relic.core.cli.RelicArgParserError(
                "Isolate and Merge flags are mutually exclusive"
            )

        logger.info(f"Unpacking `{infile}`")

        def _callback(_1: FS, srcfile: str, _2: FS, dstfile: str) -> None:
            logger.info(f"\t\tUnpacking File `{srcfile}`\n\t\tWrote to `{dstfile}`")

        # we need to open the archive to 'isolate' or to determine if we implicit merge
        with cli_open_sga(infile, logger=logger) as sga:
            roots = list(sga.iterate_fs())
            # Explicit and Implicit merge; we reuse sga to avoid reopening the filesystem
            if merge or (not isolate and len(roots) == 1):
                copy_fs(sga, f"osfs://{outdir}", on_copy=_callback, preserve_time=True)
                return _SUCCESS

            # Isolate or Implied Isolate
            with open_fs(outdir, writeable=True, create=True) as osfs:
                for alias, subfs in roots:
                    with osfs.makedir(alias, recreate=True) as osfs_subfs:
                        copy_fs(
                            subfs, osfs_subfs, on_copy=_callback, preserve_time=True
                        )
            return _SUCCESS


class EssenceInfoEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)  # type: ignore
        try:
            return super().default(o)
        except (
            TypeError
        ):  # Kinda bad; but we don't want to serialize, we want to logger.info; so i think this is an acceptable tradeoff
            return str(o)


class RelicSgaInfoCli(CliPlugin):
    _JSON_MINIFY_KWARGS: Dict[str, Any] = {"separators": (",", ":"), "indent": None}
    _JSON_MAXIFY_KWARGS: Dict[str, Any] = {"separators": (", ", ": "), "indent": 4}

    def _create_parser(
        self, command_group: Optional[_SubParsersAction] = None
    ) -> ArgumentParser:
        parser: ArgumentParser
        desc = """Reads an SGA Archive and extracts it's metadata to a json object.
            If out_json is a directory; the name of the file will be '[name of sga].json'
        """
        if command_group is None:
            parser = RelicArgParser("info", description=desc)
        else:
            parser = command_group.add_parser("info", description=desc)

        parser.add_argument(
            "src_sga",
            type=get_file_type_validator(exists=True),
            help="Source SGA File",
        )
        parser.add_argument(
            "out_json",
            type=get_path_validator(exists=False),
            help="Output File or Directory",
        )
        parser.add_argument(
            "-m",
            "--minify",
            action="store_true",
            default=False,
            help="Minifies the resulting json by stripping whitespace, newlines, and indentations. Reduces filesize",
        )

        return parser

    def command(self, ns: Namespace, *, logger: Logger) -> Optional[int]:
        infile: str = ns.src_sga
        outjson: str = ns.out_json
        minify: bool = ns.minify

        logger.info(f"Reading Info `{infile}`")

        # we need to open the archive to 'isolate' or to determine if we implicit merge
        with cli_open_sga(infile, logger=logger) as sga:
            info = sga.info_tree()

            outjson_dir, outjson_file = os.path.split(outjson)
            if len(outjson_file) == 0 or (
                os.path.exists(outjson) and os.path.isdir(outjson)
            ):  # Directory
                # Get name of sga without extension, then add .json extension
                outjson_dir = outjson
                outjson_file = os.path.splitext(os.path.split(infile)[1])[0] + ".json"

            os.makedirs(outjson_dir, exist_ok=True)
            outjson = os.path.join(outjson_dir, outjson_file)

            with open(outjson, "w", encoding=None) as info_h:
                json_kwargs: Dict[str, Any] = (
                    self._JSON_MINIFY_KWARGS if minify else self._JSON_MAXIFY_KWARGS
                )
                json.dump(info, info_h, cls=EssenceInfoEncoder, **json_kwargs)

        return _SUCCESS


class RelicSgaTreeCli(CliPlugin):
    def _create_parser(
        self, command_group: Optional[_SubParsersAction] = None
    ) -> ArgumentParser:
        parser: ArgumentParser
        desc = """Reads an SGA Archive and prints it's hierarchy"""
        if command_group is None:
            parser = RelicArgParser("tree", description=desc)
        else:
            parser = command_group.add_parser("tree", description=desc)

        parser.add_argument(
            "src_sga",
            type=get_file_type_validator(exists=True),
            help="SGA File",
        )

        return parser

    def command(self, ns: Namespace, *, logger: Logger) -> Optional[int]:
        infile: str = ns.src_sga

        logger.info(BraceMessage("Printing Tree `{0}`", infile))

        # we need to open the archive to 'isolate' or to determine if we implicit merge
        with cli_open_sga(infile, logger=logger) as sga:
            with StringIO() as writer:
                sga.tree(file=writer, with_color=True, dirs_first=True)
                writer.seek(0)
                logger.info(writer.read())
        return None


class RelicSgaVersionCli(CliPlugin):
    def _create_parser(
        self, command_group: Optional[_SubParsersAction] = None
    ) -> ArgumentParser:
        parser: ArgumentParser
        if command_group is None:
            parser = RelicArgParser("version")
        else:
            parser = command_group.add_parser("version")

        parser.add_argument(
            "sga",
            type=get_file_type_validator(exists=True),
            help="SGA File",
        )

        return parser

    def command(self, ns: Namespace, *, logger: logging.Logger) -> Optional[int]:
        sga_file: str = ns.sga
        logger.info("Sga Version")
        try:
            with open(sga_file, "rb") as sga:
                if not MAGIC_WORD.check(sga, advance=True):
                    logger.warning("File is not an SGA")
                else:
                    version = VersionSerializer.read(sga)
                    logger.info(version)
        except IOError:  # pragma: nocover
            # I don't know how to force an io error here for coverage testing
            # we safely handle bad file paths
            # So I believe this only occurs when a genuine fatal error occurs
            logger.error("Error reading file")
            raise
        return None


class RelicSgaListCli(CliPlugin):
    def _create_parser(
        self, command_group: Optional[_SubParsersAction] = None
    ) -> ArgumentParser:
        parser: ArgumentParser
        if command_group is None:
            parser = RelicArgParser("list")
        else:
            parser = command_group.add_parser("list")

        return parser

    def command(self, ns: Namespace, *, logger: logging.Logger) -> Optional[int]:
        logger.info("Installed SGA Plugins")
        keys = list(sga_registry)
        for key in keys:
            logger.info(key)
        if len(keys) == 0:
            logger.info("No Plugins Found!")

        return None
