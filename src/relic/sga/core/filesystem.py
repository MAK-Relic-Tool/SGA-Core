from __future__ import annotations

import typing
from typing import Optional, Dict, Any, BinaryIO, Text, Collection

from fs import ResourceType, errors
from fs.base import FS
from fs.info import Info
from fs.memoryfs import MemoryFS, _DirEntry, _MemoryFile
from fs.multifs import MultiFS
from fs.osfs import OSFS
from fs.path import split

from relic.sga.core import Version, MagicWord
from relic.sga.core.errors import VersionNotSupportedError

ESSENCE_NAMESPACE = "essence"


class EssenceFSHandler(typing.Protocol):
    def read(self, sga_stream: BinaryIO) -> EssenceFS:
        raise NotImplementedError

    def write(self, sga_stream: BinaryIO, fs: EssenceFS) -> int:
        raise NotImplementedError


class EssenceFSFactory:
    def __init__(self):
        self.handler_map: Dict[Version, EssenceFSHandler] = {}

    def register_handler(self, version: Version, handler: EssenceFSHandler):
        if version is None:
            raise ValueError
        if handler is None:
            raise ValueError
        #     self.default_handler = handler
        # else:
        self.handler_map[version] = handler

    @staticmethod
    def _read_magic_and_version(sga_stream: BinaryIO) -> Version:
        sga_stream.seek(0)
        MagicWord.read_magic_word(sga_stream)
        return Version.unpack(sga_stream)

    def _get_handler(self, version: Version):
        handler = self.handler_map.get(version)
        if handler is None:
            # This may raise a 'false positive' if a Null handler is registered
            raise VersionNotSupportedError(version, list(self.handler_map.keys()))
        return handler

    def _get_handler_from_stream(
        self, sga_stream: BinaryIO, version: Optional[Version] = None
    ):
        if version is None:
            version = self._read_magic_and_version(sga_stream)
        return self._get_handler(version)

    def _get_handler_from_fs(
        self, sga_fs: EssenceFS, version: Optional[Version] = None
    ):
        if version is None:
            sga_version: Dict[str, int] = sga_fs.getmeta("essence").get("version")  # type: ignore
            version = Version(sga_version.get("major"), sga_version.get("minor"))
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


class _EssenceFile(_MemoryFile):
    ...  # I plan on allowing lazy file loading from the archive; I'll likely need to implement this to do that


class _EssenceDirEntry(_DirEntry):
    def __init__(self, resource_type: ResourceType, name: Text):
        super().__init__(resource_type, name)
        self.essence = {}

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
    def __init__(self, host: EssenceFS):
        super().__init__()
        self._hostfs = host

    def _make_dir_entry(self, resource_type, name):
        return _EssenceDirEntry(resource_type, name)

    def setinfo(self, path, info):
        _path = self.validatepath(path)
        with self._lock:
            dir_path, file_name = split(_path)
            parent_dir_entry = self._get_dir_entry(dir_path)

            if parent_dir_entry is None or file_name not in parent_dir_entry:
                raise errors.ResourceNotFound(path)

            resource_entry = typing.cast(
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

    def getessence(self, path):
        return self.getinfo(path, [ESSENCE_NAMESPACE])


class EssenceFS(MultiFS):
    def __init__(self):
        super().__init__()
        self._sga_meta = {}

    def getmeta(self, namespace="standard"):
        if namespace == ESSENCE_NAMESPACE:
            return self._sga_meta.copy()
        else:
            return super().getmeta(namespace)

    def setmeta(self, meta: Dict[str, Any], namespace="standard"):
        if namespace == ESSENCE_NAMESPACE:
            self._sga_meta = meta.copy()
        else:
            raise NotImplementedError

    def getessence(self, path):
        return self.getinfo(path, [ESSENCE_NAMESPACE])

    def create_drive(self, name: str) -> _EssenceDriveFS:
        drive = _EssenceDriveFS(self)
        self.add_fs(name, drive)
        return drive

    def _delegate(self, path):
        # type: (Text) -> Optional[FS]
        # Resolve path's drive, if present,
        #   otherwise; use underlying FS
        if ":" in path:
            parts = path.split(":", 1)
            return self.get_fs(parts[0])
        else:
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

if __name__ == "__main__":
    temp = OSFS("")
    print(temp.root_path)
    print(*temp.listdir("/"))
