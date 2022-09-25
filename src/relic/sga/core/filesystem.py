from __future__ import annotations
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable, BinaryIO

from fs import ResourceType
from fs.base import FS
from fs.info import Info
from fs.memoryfs import MemoryFS
from fs.multifs import MultiFS
from fs.permissions import Permissions

from relic.sga.core import StorageType, Version, MagicWord
from relic.sga.core.abstract import Archive, Drive, Folder, File
from relic.sga.core.errors import VersionNotSupportedError


class NamespaceAssembler:
    @staticmethod
    def _wrap(namespace: str, info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {namespace: info}

    @staticmethod
    def basic(name: str, is_dir: bool) -> Dict[str, Dict[str, Any]]:
        return NamespaceAssembler._wrap("basic", locals())

    @staticmethod
    def details(
            accessed: Optional[datetime] = None,
            created: Optional[datetime] = None,
            metadata_changed: Optional[datetime] = None,
            modified: Optional[datetime] = None,
            size: int = 0,
            type: ResourceType = ResourceType.unknown,
    ) -> Dict[str, Dict[str, Any]]:
        return NamespaceAssembler._wrap("details", locals())

    @staticmethod
    def access(
            gid: Optional[int] = None,
            group: Optional[str] = None,
            permissions: Optional[Permissions] = None,
            uid: Optional[int] = None,
            user: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        return NamespaceAssembler._wrap("access", locals())

    @staticmethod
    def link(
            target: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        return NamespaceAssembler._wrap("link", locals())

    @staticmethod
    def custom(namespace: str, **kwargs) -> Dict[str, Dict[str, Any]]:
        return NamespaceAssembler._wrap(namespace, kwargs)


class SGAFSHandler:
    def read(self, sga_stream: BinaryIO, fs: SGAFS, create_drive_backing_fs: Optional[Callable[[str], FS]] = None) -> None:
        raise NotImplementedError

    def write(self, sga_stream: BinaryIO, fs: SGAFS) -> int:
        raise NotImplementedError


class SGAFSFactory:
    def __init__(self):
        self.default_handler: Optional[SGAFSHandler] = None
        self.handler_map: Dict[Version, SGAFSHandler] = {}

    def register_handler(self, handler: SGAFSHandler, version: Optional[Version] = None):
        if version is None:
            self.default_handler = handler
        else:
            self.handler_map[version] = handler

    @staticmethod
    def _read_magic_and_version(sga_stream: BinaryIO) -> Version:
        sga_stream.seek(0)
        MagicWord.read_magic_word(sga_stream)
        return Version.unpack(sga_stream)

    def _get_handler_from_stream(self, sga_stream: BinaryIO, version: Optional[Version] = None):
        if version is None:
            version = self._read_magic_and_version(sga_stream)
        handler = self.handler_map.get(version, self.default_handler)
        if handler is None:
            # This may raise a 'false positive' if a Null handler is registered
            raise VersionNotSupportedError(version, list(self.handler_map.keys()))
        return handler

    def _get_handler_from_fs(self, sga_fs: SGAFS, version: Optional[Version] = None):
        if version is None:
            sga_version: Dict[str, int] = sga_fs.getmeta("SGA").get("version")  # type: ignore
            version = Version(sga_version.get("major"), sga_version.get("minor"))
        handler = self.handler_map.get(version, self.default_handler)
        if handler is None:
            # This may raise a 'false positive' if a Null handler is registered
            raise VersionNotSupportedError(version, list(self.handler_map.keys()))
        return handler

    @staticmethod
    def _default_create_drive_backing_fs(drive: str) -> FS:
        return MemoryFS()

    def read(self, sga_stream: BinaryIO, version: Optional[Version] = None, create_drive_backing_fs: Optional[Callable[[str], FS]] = None) -> SGAFS:
        handler = self._get_handler_from_stream(sga_stream, version)

        if create_drive_backing_fs is None:
            create_drive_backing_fs = self._default_create_drive_backing_fs

        sgafs = SGAFS()
        handler.read(sga_stream, sgafs, create_drive_backing_fs)

        return sgafs

    def write(self, sga_stream: BinaryIO, sga_fs: SGAFS, version: Optional[Version] = None) -> int:
        handler = self._get_handler_from_fs(sga_fs, version)

        return handler.write(sga_stream, sga_fs)


class SGAFS(MultiFS):
    def __init__(self):
        super().__init__()
        self._sga_meta = {}

    def getmeta(self, namespace="standard"):
        if namespace == "SGA":
            return self._sga_meta.copy()
        else:
            return super().getmeta(namespace)

    def getSGA(self, path):
        return self.getinfo(path, ["SGA"])


if __name__ == "__main__":
    test_file = File("test.txt", b"This is a Test!", StorageType.STORE, False, None)
    test_folder = Folder("Test", [], [test_file])
    data_folders = [test_folder]
    data_files = []
    data_drive = Drive("data", "", data_folders, data_files)
    attr_drive = Drive("attr", "", [], [test_file])
    archive = Archive("Test", None, [data_drive, attr_drive])

    with SGAFS() as fs:
        data_fs = MemoryFS()
        fs.add_fs("data", data_fs)
        data_dir = data_fs.makedir("Test Data")
        with data_dir.open("sample_data.txt","wb") as data_sample_text:
            data_sample_text.write(b"Sample Data Text!")

        attr_fs = MemoryFS()
        fs.add_fs("attr",attr_fs)
        attr_dir = attr_fs.makedir("Test Attr")
        with attr_dir.open("sample_attr.txt","wb") as attr_sample_text:
            attr_sample_text.write(b"Sample Attr Text!")

        for root, folders, files in fs.walk():
            print(root, "\n\t", folders, "\n\t", files)

        for name, sub_fs in fs.iterate_fs():
            print(name)
            for root, folders, files in sub_fs.walk():
                print("\t", root, "\n\t\t", folders, "\n\t\t", files)

        print(fs.getinfo("/",["basic","access"]).raw)
    pass
