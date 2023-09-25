from __future__ import annotations

import abc
import os
from os.path import expanduser
from typing import (
    Optional,
    Dict,
    Any,
    BinaryIO,
    Text,
    Collection,
    Mapping,
    cast,
    Protocol,
    TypeVar,
    Generic,
    runtime_checkable,
)

import fs.opener.errors
import pkg_resources
from fs import ResourceType, errors
from fs.base import FS
from fs.info import Info
from fs.memoryfs import MemoryFS, _DirEntry, _MemoryFile
from fs.multifs import MultiFS
from fs.opener import Opener, registry as fs_registry
from fs.opener.parse import ParseResult
from fs.path import split

from relic.sga.core.definitions import Version, MagicWord, _validate_magic_word
from relic.sga.core.errors import VersionNotSupportedError

ESSENCE_NAMESPACE = "essence"

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
        try:
            entry_point_result = entry_point.load()
        except:  # Wrap in exception
            raise
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
        version = Version.unpack(sga_stream)
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


# @fs_registry.install
# Can't use decorator; it breaks subclassing for entrypoints
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
                return EssenceFS()
            else:
                raise fs.opener.errors.OpenerError(
                    "No path was given and opener not marked for 'create'!"
                )

        _path = os.path.abspath(os.path.join(cwd, expanduser(parse_result.resource)))
        path = os.path.normpath(_path)

        # Create will always create a new EssenceFS if needed
        try:
            with open(path, "rb") as sga_file:
                return self.factory.read(sga_file)
        except FileNotFoundError as e:
            if create:
                return EssenceFS()
            else:
                raise


fs_registry.install(EssenceFSOpener)


class _EssenceFile(_MemoryFile):
    ...  # I plan on allowing lazy file loading from the archive; I'll likely need to implement this to do that


class _EssenceDirEntry(_DirEntry):
    def __init__(self, resource_type: ResourceType, name: Text):
        super().__init__(resource_type, name)
        self.essence: Dict[str, object] = {}

    def to_info(self, namespaces=None):
        # type: (Optional[Collection[Text]]) -> Info
        info = super().to_info(namespaces)
        if (
            namespaces is not None
            and not self.is_dir
            and ESSENCE_NAMESPACE in namespaces
        ):
            info_dict = dict(info.raw)
            info_dict[ESSENCE_NAMESPACE] = self.essence.copy()
            info = Info(info_dict)
        return info


class _EssenceDriveFS(MemoryFS):
    def __init__(self, alias: str) -> None:
        super().__init__()
        self.alias = alias

    def _make_dir_entry(
        self, resource_type: ResourceType, name: str
    ) -> _EssenceDirEntry:
        return _EssenceDirEntry(resource_type, name)

    def validatepath(self, path: str) -> str:
        if ":" in path:
            parts = path.split(":", 1)
            if parts[0][0] == "/":
                parts[0] = parts[0][1:]
            if parts[0] != self.alias:
                raise fs.errors.InvalidPath(
                    path,
                    f"Alias `{parts[0]}` does not math the Drive's Alias `{self.alias}`",
                )
            fixed_path = parts[1]
        else:
            fixed_path = path
        return super().validatepath(fixed_path)

    def setinfo(self, path: str, info: Mapping[str, Mapping[str, object]]) -> None:
        _path = self.validatepath(path)
        with self._lock:
            dir_path, file_name = split(_path)
            parent_dir_entry = self._get_dir_entry(dir_path)

            if parent_dir_entry is None or file_name not in parent_dir_entry:
                raise errors.ResourceNotFound(path)

            resource_entry = cast(
                _EssenceDirEntry, parent_dir_entry.get_entry(file_name)
            )

            if "details" in info:
                details = info["details"]
                if "accessed" in details:
                    resource_entry.accessed_time = details["accessed"]  # type: ignore
                if "modified" in details:
                    resource_entry.modified_time = details["modified"]  # type: ignore

            if ESSENCE_NAMESPACE in info and not resource_entry.is_dir:
                essence = info[ESSENCE_NAMESPACE]
                resource_entry.essence.clear()
                resource_entry.essence.update(essence)

            # if LAZY_NAMESPACE in info and not resource_entry.is_dir:
            #     lazy

    def getessence(self, path: str) -> Info:
        return self.getinfo(path, [ESSENCE_NAMESPACE])


class EssenceFS(MultiFS):
    def __init__(self) -> None:
        super().__init__()
        self._sga_meta: Dict[str, object] = {}

    def getmeta(self, namespace: str = "standard") -> Mapping[str, object]:
        if namespace == ESSENCE_NAMESPACE:
            return self._sga_meta.copy()
        return super().getmeta(namespace)

    def setmeta(self, meta: Dict[str, Any], namespace: str = "standard") -> None:
        if namespace == ESSENCE_NAMESPACE:
            self._sga_meta = meta.copy()
        else:
            raise NotImplementedError

    def getessence(self, path: str) -> Info:
        return self.getinfo(path, [ESSENCE_NAMESPACE])

    def create_drive(self, name: str) -> _EssenceDriveFS:
        drive = _EssenceDriveFS(name)
        self.add_fs(name, drive)
        return drive

    def _delegate(self, path):
        # type: (Text) -> Optional[FS]
        # Resolve path's drive, if present,
        #   otherwise; use underlying FS
        if ":" in path:
            parts = path.split(":", 1)
            return self.get_fs(parts[0])

        return super()._delegate(path)


# if __name__ == "__main__":
#     test_file = File("test.txt", b"This is a Test!", StorageType.STORE, False, None)
#     test_folder = Folder("Test", [], [test_file])
#     data_folders = [test_folder]
#     data_files = []
#     data_drive = Drive("data", "", data_folders, data_files)
#     attr_drive = Drive("attr", "", [], [test_file])
#     archive = Archive("Test", None, [data_drive, attr_drive])
#
#     with SGAFS() as fs:
#         data_fs = MemoryFS()
#         fs.add_fs("data", data_fs)
#         data_dir = data_fs.makedir("Test Data")
#         with data_dir.open("sample_data.txt", "wb") as data_sample_text:
#             data_sample_text.write(b"Sample Data Text!")
#
#         attr_fs = MemoryFS()
#         fs.add_fs("attr", attr_fs)
#         attr_dir = attr_fs.makedir("Test Attr")
#         with attr_dir.open("sample_attr.txt", "wb") as attr_sample_text:
#             attr_sample_text.write(b"Sample Attr Text!")
#
#         for root, folders, files in fs.walk():
#             print(root, "\n\t", folders, "\n\t", files)
#
#         for name, sub_fs in fs.iterate_fs():
#             print(name)
#             for root, folders, files in sub_fs.walk():
#                 print("\t", root, "\n\t\t", folders, "\n\t\t", files)
#
#         print(fs.getinfo("/", ["basic", "access"]).raw)
#     pass

__all__ = [
    "ESSENCE_NAMESPACE",
    "EssenceFSHandler",
    "EssenceFSFactory",
    "_EssenceFile",
    "_EssenceDirEntry",
    "_EssenceDriveFS",
    "EssenceFS",
    "registry",
    "EssenceFSOpener",
]
