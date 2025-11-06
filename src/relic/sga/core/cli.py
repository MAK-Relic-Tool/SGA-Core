from __future__ import annotations

import dataclasses
import json
import logging
import os.path
from argparse import ArgumentParser, Namespace
from io import StringIO
from json import JSONEncoder
from logging import Logger
from typing import Optional, Any, Dict

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
from relic.sga.core.parallel_advanced import AdvancedParallelUnpacker

_SUCCESS = 0

# Backwards compatibility aliases for v2 package
_get_file_type_validator = get_file_type_validator
_get_dir_type_validator = get_dir_type_validator
_get_path_validator = get_path_validator


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

        # Performance options
        parser.add_argument(
            "--fast",
            help="Use Fast native extraction (default, 80x faster)",
            action="store_true",
            default=True,
        )
        parser.add_argument(
            "--compatible",
            help="Use compatible fs-based extraction (slower, more compatible)",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--workers",
            type=int,
            help="Number of parallel workers for fast extraction (default: CPU count - 1)",
            default=None,
        )

        return parser

    def command(self, ns: Namespace, *, logger: Logger) -> Optional[int]:
        infile: str = ns.src_sga
        outdir: str = ns.out_dir
        merge: bool = ns.merge
        isolate: bool = ns.isolate
        use_fast: bool = not ns.compatible  # Use fast unless --compatible specified
        num_workers: Optional[int] = ns.workers

        if merge and isolate:  # pragma: nocover
            # This error should be impossible
            raise relic.core.cli.RelicArgParserError(
                "Isolate and Merge flags are mutually exclusive"
            )

        logger.info(f"Unpacking `{infile}`")

        # Use Fast native extraction by default
        if use_fast:
            try:
                import multiprocessing

                if num_workers is None:
                    num_workers = max(1, multiprocessing.cpu_count() - 1)

                logger.info(f"Using Fast native extraction ({num_workers} workers)")
                unpacker = AdvancedParallelUnpacker(
                    num_workers=num_workers, enable_delta=False, logger=logger
                )

                # Progress callback
                def _progress(current: int, total: int) -> None:
                    if current % 500 == 0 or current == total:
                        logger.info(
                            f"  Progress: {current}/{total} files ({current*100//total}%)"
                        )

                stats = unpacker.extract_native_ultra_fast(
                    infile, outdir, on_progress=_progress
                )

                logger.info(
                    f"Extraction complete: {stats.extracted_files} files extracted"
                )
                if stats.failed_files > 0:
                    logger.warning(f"Failed: {stats.failed_files} files")
                    return 1

                return _SUCCESS

            except Exception as e:
                logger.warning(f"Fast extraction failed: {e}")
                logger.info("Falling back to compatible mode...")
                use_fast = False

        # Fallback to compatible fs-based extraction
        if not use_fast:
            logger.info("Using compatible fs-based extraction")

            def _callback(_1: FS, srcfile: str, _2: FS, dstfile: str) -> None:
                logger.info(f"\t\tUnpacking File `{srcfile}`\n\t\tWrote to `{dstfile}`")

            # we need to open the archive to 'isolate' or to determine if we implicit merge
            sga: EssenceFS
            with open_fs(infile, default_protocol="sga") as sga:  # type: ignore
                roots = list(sga.iterate_fs())
                # Explicit and Implicit merge; we reuse sga to avoid reopening the filesystem
                if merge or (not isolate and len(roots) == 1):
                    copy_fs(
                        sga, f"osfs://{outdir}", on_copy=_callback, preserve_time=True
                    )
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
        sga: EssenceFS
        with open_fs(infile, default_protocol="sga") as sga:  # type: ignore
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
        sga: EssenceFS
        with open_fs(infile, default_protocol="sga") as sga:  # type: ignore
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
