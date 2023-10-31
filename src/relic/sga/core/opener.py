from __future__ import annotations

import abc
import os
from os.path import expanduser
from typing import (
    Generic,
    Dict,
    Optional,
    Any,
    runtime_checkable,
    Protocol,
    BinaryIO,
    Collection,
    TypeVar,
)

import fs.opener
import pkg_resources
from fs.base import FS
from fs.opener import Opener
from fs.opener.parse import ParseResult
from relic.core.errors import RelicToolError
from relic.core.lazyio import BinaryWrapper

from relic.sga.core.definitions import Version, MagicWord
from relic.sga.core.serialization import _validate_magic_word, VersionSerializer
from relic.sga.core.errors import VersionNotSupportedError
from relic.sga.core.essencefs import EssenceFS

TKey = TypeVar("TKey")
TValue = TypeVar("TValue")


class EntrypointRegistry(Generic[TKey, TValue]):
    def __init__(self, entry_point_path: str, autoload: bool = False):
        self._entry_point_path = entry_point_path
        self._mapping: Dict[TKey, TValue] = {}
        self._autoload = autoload

    def register(self, key: TKey, value: TValue) -> None:
        self._mapping[key] = value

    @abc.abstractmethod
    def auto_register(self, value: TValue) -> None:
        raise NotImplementedError

    def get(self, key: TKey, default: Optional[TValue] = None) -> Optional[TValue]:
        if key in self._mapping:
            return self._mapping[key]

        if self._autoload:
            try:
                key_root_path = self._entry_point_path
                key_path = self._key2entry_point_path(key)
                entry_points = pkg_resources.iter_entry_points(key_root_path, key_path)
                entry_point = next(entry_points)
            except StopIteration:
                entry_point = None
            if entry_point is None:
                return default
            self._auto_register_entrypoint(entry_point)
            if key not in self._mapping:
                raise NotImplementedError  # TODO specify autoload failed to load in a usable value
            return self._mapping[key]
        return default

    @abc.abstractmethod
    def _key2entry_point_path(self, key: TKey) -> str:
        raise NotImplementedError

    def _auto_register_entrypoint(self, entry_point: Any) -> None:
        # try:
        entry_point_result = entry_point.load()
        # except:  # Wrap in exception
        #     raise
        return self._register_entrypoint(entry_point_result)

    @abc.abstractmethod
    def _register_entrypoint(self, entry_point_result: Any) -> None:
        raise NotImplementedError


@runtime_checkable
class EssenceFSHandler(Protocol):
    version: Version

    def read(self, stream: BinaryIO) -> EssenceFS:
        raise NotImplementedError

    def write(self, stream: BinaryIO, essence_fs: EssenceFS) -> int:
        raise NotImplementedError


class EssenceFSFactory(EntrypointRegistry[Version, EssenceFSHandler]):
    def _key2entry_point_path(self, key: Version) -> str:
        return f"v{key.major}.{key.minor}"

    def _register_entrypoint(self, entry_point_result: Any) -> None:
        if isinstance(entry_point_result, EssenceFSHandler):
            self.auto_register(entry_point_result)
        elif isinstance(entry_point_result, (list, tuple, Collection)):
            version, handler = entry_point_result
            if not isinstance(handler, EssenceFSHandler):
                handler = handler()
            self.register(version, handler)
        else:
            # Callable; register nested result
            self._register_entrypoint(entry_point_result())

    def auto_register(self, value: EssenceFSHandler) -> None:
        self.register(value.version, value)

    def __init__(self, autoload: bool = True) -> None:
        super().__init__("relic.sga.handler", autoload)

    @staticmethod
    def _read_magic_and_version(sga_stream: BinaryIO) -> Version:
        # sga_stream.seek(0)
        jump_back = sga_stream.tell()
        _validate_magic_word(MagicWord, sga_stream, advance=True)
        version = VersionSerializer.read(sga_stream)
        sga_stream.seek(jump_back)
        return version

    def _get_handler(self, version: Version) -> EssenceFSHandler:
        handler = self.get(version)
        if handler is None:
            # This may raise a 'false positive' if a Null handler is registered
            raise VersionNotSupportedError(version, list(self._mapping.keys()))
        return handler

    def _get_handler_from_stream(
        self, sga_stream: BinaryIO, version: Optional[Version] = None
    ) -> EssenceFSHandler:
        if version is None:
            version = self._read_magic_and_version(sga_stream)
        return self._get_handler(version)

    def _get_handler_from_fs(
        self, sga_fs: EssenceFS, version: Optional[Version] = None
    ) -> EssenceFSHandler:
        if version is None:
            sga_version: Dict[str, int] = sga_fs.getmeta("essence").get("version")  # type: ignore
            version = Version(sga_version["major"], sga_version["minor"])
        return self._get_handler(version)

    def read(
        self, sga_stream: BinaryIO, version: Optional[Version] = None
    ) -> EssenceFS:
        handler = self._get_handler_from_stream(sga_stream, version)
        return handler.read(sga_stream)

    def write(
        self, sga_stream: BinaryIO, sga_fs: EssenceFS, version: Optional[Version] = None
    ) -> int:
        handler = self._get_handler_from_fs(sga_fs, version)
        return handler.write(sga_stream, sga_fs)


registry = EssenceFSFactory(True)


class EssenceFSOpener(Opener):
    def __init__(self, factory: Optional[EssenceFSFactory] = None):
        if factory is None:
            factory = registry
        self.factory = factory

    protocols = ["sga"]

    def open_fs(
        self,
        fs_url: str,
        parse_result: ParseResult,
        writeable: bool,
        create: bool,
        cwd: str,
    ) -> FS:
        # All EssenceFS should be writable; so we can ignore that

        # Resolve Path
        if fs_url == "sga://":
            if create:
                raise RelicToolError(
                    "Cannot create an SGA from fs.open_fs;"
                    " please manually create an empty FS object from an appropriate SGA Plugin."
                )
            raise fs.opener.errors.OpenerError(
                "No path was given and opener not marked for 'create'!"
            )

        _path = os.path.abspath(os.path.join(cwd, expanduser(parse_result.resource)))
        path = os.path.normpath(_path)

        # TODO, refactor this to be in the factory,
        sga_file = None
        handler = None
        try:
            _sga_file = open(path, "rb")
            sga_file = BinaryWrapper(_sga_file)
            handler = self.factory._get_handler_from_stream(sga_file)
            sga_fs = handler.read(sga_file)
            return sga_fs

        except (
            FileNotFoundError
        ):  # FNF ~ close file (it shouldn't exist, but better safe than sorry)
            if create:
                raise NotImplementedError("Cannot create a new SGA FS via open_fs")
                # return EssenceFS()

            raise

        finally:  # Close file if lazy-ness not required
            auto_close = True  # Default, force closure

            if handler is not None:
                # If autoclose is not defined, assume not lazy
                auto_close = not hasattr(handler, "autoclose") or getattr(
                    handler, "autoclose"
                )

            if sga_file is not None and auto_close:
                sga_file.close()


# fs_registry.install(EssenceFSOpener)
class _FakeSerializer:
    def read(self, stream: BinaryIO) -> EssenceFS:
        raise NotImplementedError

    def write(self, stream: BinaryIO, essence_fs: EssenceFS) -> int:
        raise NotImplementedError
