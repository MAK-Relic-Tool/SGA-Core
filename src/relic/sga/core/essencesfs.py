from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import (
    Optional,
    Dict,
    BinaryIO,
    Mapping,
    TypeVar,
    Sequence,
    AnyStr,
    Iterable,
    Union,
    Any,
)

import fs.opener.errors
from fs import ResourceType, errors
from fs.base import FS
from fs.info import Info
from fs.mode import Mode
from fs.path import split, normpath, iteratepath, abspath, basename

from relic.sga.core import Version
from relic.sga.core._proxyfs import SgaFsFile, SgaFsFolder, SgaFsDrive, SgaFs
from relic.sga.core.lazyio import (
    BinaryWrapper,
    tell_end,
)

# ESSENCE_NAMESPACE = "essence"

TKey = TypeVar("TKey")
TValue = TypeVar("TValue")


def _ns_basic(name: str, is_dir: bool):
    return {"basic": {"name": name, "is_dir": is_dir}}


def __ns_creator(
    locals: Dict[str, Any],
    args_name: str = "args",
    kwargs_name: str = "kwargs",
    ignore: Iterable[str] = None,
):
    ns = locals.copy()

    for key in ignore or []:
        if key in ns:
            del ns[key]

    args = ns.get(args_name)
    if args is not None:
        raise NotImplementedError

    if kwargs_name in ns:
        kwargs = ns[kwargs_name]
        del ns[kwargs_name]
        ns.update(kwargs)

    return ns


def _ns_supports(
    *, archive_verification: bool = False, file_verification: bool = False, **kwargs
):
    return __ns_creator(locals())


def _ns_essence(*, version: Version, **kwargs):
    return __ns_creator(locals())


class _EssenceNode:
    def __init__(self, resource_type: ResourceType, name: str):
        self._resource_type = resource_type
        self.children: Dict[str, _EssenceNode] = (
            None if resource_type != ResourceType.directory else {}
        )
        self.name = name

    @property
    def is_dir(self):
        return self._resource_type == ResourceType.directory

    @property
    def is_file(self):
        return self._resource_type == ResourceType.file

    def get_child_node(self, child_path: str) -> Optional[_EssenceNode]:
        if not self.is_dir:
            raise fs.errors.DirectoryExpected("TODO")
        return self.children.get(child_path)

    def add_child_node(self, child_path: str, child_node: _EssenceNode):
        if not self.is_dir:
            raise fs.errors.DirectoryExpected("TODO")
        if child_path in self.children:
            raise fs.errors.DirectoryExists("TODO")
        self.children[child_path] = child_node

    def has_child(self, child_path):
        return child_path in self.children

    def get_info(self, namespaces: Optional[Sequence[str]]) -> Mapping[str, object]:
        result = {}
        result.update(_ns_basic(self.name, self.is_dir))
        return result

    def set_info(self, mapping: Mapping[str, object]):
        raise NotImplementedError

    def list_dir(self):
        if not self.is_dir:
            raise fs.errors.DirectoryExpected("TODO")
        return list(self.children.keys())

    def verify(self):
        raise NotImplementedError


class _EssenceFileHandle(BinaryWrapper):
    def __init__(
        self, parent: BinaryIO, mode: Mode, lock: RLock, name: Optional[str] = None
    ):
        super().__init__(parent, close_parent=False, name=name)
        self._mode = mode
        self._lock = lock
        self._now = 0 if not mode.appending else tell_end(parent)

    def close(self):
        return super().close()

    def readable(self) -> bool:
        return self._mode.reading

    def writable(self) -> bool:
        return self._mode.writing

    def mode(self) -> str:
        return self._mode.to_platform_bin()

    def __validate_readable(self):
        if not self.readable():
            raise IOError()  # TODO

    def __validate_writable(self):
        if not self.writable():
            raise IOError()  # TODO

    @contextmanager
    def __rw_ctx(self):
        with self._lock as lock:
            self.seek(self._now)
            yield lock
            self._now = super().tell()

    def read(self, __n: int = -1) -> AnyStr:
        self.__validate_readable()
        with self.__rw_ctx():
            return super(_EssenceFileHandle, self).read(__n)

    def readline(self, __limit: int = -1) -> AnyStr:
        self.__validate_readable()
        with self.__rw_ctx():
            return super(_EssenceFileHandle, self).readline(__limit)

    def readlines(self, __hint: int = -1) -> list[AnyStr]:
        self.__validate_readable()
        with self.__rw_ctx():
            return super(_EssenceFileHandle, self).readlines(__hint)

    def write(self, __s: AnyStr) -> int:
        self.__validate_writable()
        with self.__rw_ctx():
            return super(_EssenceFileHandle, self).write(__s)

    def writelines(self, __lines: Iterable[AnyStr]) -> None:
        self.__validate_writable()
        with self.__rw_ctx():
            return super(_EssenceFileHandle, self).writelines(__lines)

    def seek(self, _offset: int, _whence: int = 0) -> int:
        with self._lock:
            self._now = super(_EssenceFileHandle, self).seek(_offset, _whence)
            return self._now

    def tell(self) -> int:
        return self._now


class _EssenceFS(FS):
    STANDARD_NAMESPACE = "standard"
    ESSENCE_NAMESPACE = "essence"
    SUPPORTS_NAMESPACE = "supports"

    _meta = {}
    _supports = {}
    _essence = {}

    def __init__(self):
        super().__init__()
        self._roots: Dict[str, _EssenceNode] = {}  # self._create_dir_node("/")

    def getmeta(self, namespace: str = STANDARD_NAMESPACE):
        if namespace == self.ESSENCE_NAMESPACE:
            return self._essence.copy()
        elif namespace == self.SUPPORTS_NAMESPACE:
            return self._supports.copy()
        else:
            super().getmeta(namespace)

    def _create_node(self, resource_type: ResourceType, name: str, *args, **kwargs):
        raise NotImplementedError

    def _create_file_node(self, name: str, *args, **kwargs):
        return self._create_node(
            resource_type=ResourceType.file, name=name, *args, **kwargs
        )

    def _create_dir_node(self, name: str, *args, **kwargs):
        return self._create_node(
            resource_type=ResourceType.directory, name=name, *args, **kwargs
        )

    @staticmethod
    def _try_get_node_in_drive(root: _EssenceNode, path: str):
        current_node: Optional[_EssenceNode] = root
        for path_component in iteratepath(path):
            if current_node is None:
                return None
            if not current_node.is_dir:
                return None
            current_node = current_node.get_child_node(path_component)
        return current_node

    def _try_get_node(self, path: str) -> Optional[_EssenceNode]:
        drive, path = self._resolve_path(path)

        with self.lock():
            if drive is None:
                for root in self._roots.values():
                    node = self._try_get_node_in_drive(root, path)
                    if node is not None:
                        return node
                return None
            elif drive in self._roots:
                return self._try_get_node_in_drive(self._roots[drive], path)
            else:
                return None

    def _get_node(self, path: str) -> _EssenceNode:
        node = self._try_get_node(path)
        if node is None:
            raise fs.errors.ResourceNotFound(path)
        return node

    def _resolve_path(self, path: str):
        if ":" in path:
            drive, drive_path = path.split(":", 1)
        else:
            drive, drive_path = None, path
        return drive, abspath(normpath(drive_path)).replace("\\", "/")

    def _resolve_split(self, path: str):
        if ":" in path:
            drive, drive_path = path.split(":", 1)
            head, tail = split(path)
            return drive + ":" + head, tail
        else:
            return split(path)

    def setmeta(self, meta: Mapping[str, object], namespace: str = STANDARD_NAMESPACE):
        if namespace == self.ESSENCE_NAMESPACE:
            return self._set_essence_meta(meta)
        else:
            raise NotImplementedError

    def getinfo(self, path, namespaces=None):
        with self._lock:
            node = self._get_node(path)
            raw_info = node.get_info(namespaces)
            return Info(raw_info)

    def listdir(self, path):
        with self._lock:
            if path != "/":
                node = self._get_node(path)
                return node.list_dir()
            else:
                magic_root = set()
                for root in self._roots.values():
                    magic_root.update(root.list_dir())
                return list(magic_root)

    def makedir(self, path, permissions=None, recreate=False):
        with self._lock:
            _, _path = self._resolve_path(path)
            if _path == "/":
                if recreate:
                    return self.opendir(path)
                else:
                    raise errors.DirectoryExists(path)

            dir_path, dir_name = self._resolve_split(path)
            parent_dir = self._get_node(dir_path)

            dir_entry = parent_dir.get_child_node(dir_name)
            if dir_entry is not None and not recreate:
                raise errors.DirectoryExists(path)

            if dir_entry is None:
                new_dir = self._create_dir_node(dir_name)
                parent_dir.add_child_node(dir_name, new_dir)

            return self.opendir(path)

    def openbin(self, path, mode="r", buffering=-1, **options):
        with self._lock:
            mode = Mode(mode)
            mode.validate_bin()
            drive, _path = self._resolve_path(path)
            dir_path, file_name = self._resolve_split(_path)

            if len(file_name) == 0:
                raise errors.FileExpected(path)

            raise NotImplementedError
            dir_node = self._get_node(dir_path)
            if mode.create:
                if not dir_node.has_child(file_name):
                    new_entry = self._make_dir_entry

    def remove(self, path):
        raise NotImplementedError

    def removedir(self, path):
        raise NotImplementedError

    def setinfo(self, path, info):
        raise NotImplementedError

    def verify(self, path: Optional[str] = None) -> bool:
        raise NotImplementedError


EssenceFS = _EssenceFS

__all__ = [
    "_EssenceFS",
]


def _basename(name: str):
    return basename(name.replace("\\", "/"))


class _LazyEssenceNode(_EssenceNode):
    def __init__(self, backing: Union[SgaFsDrive, SgaFsFolder, SgaFsFile]):
        self._backing = backing
        self._loaded = False
        if isinstance(backing, SgaFsDrive):
            resource_type = ResourceType.directory
            name = backing.path
            self._load = self.__load_drive
        elif isinstance(backing, SgaFsFolder):
            resource_type = ResourceType.directory
            name = _basename(backing.name)
            self._load = self.__load_folder
        elif isinstance(backing, SgaFsFile):
            resource_type = ResourceType.file
            name = backing.name
            self._load = self.__load_file
        else:
            raise ValueError(backing)
        super().__init__(resource_type, name)
        self._decomp_handle = None
        self._comp_handle = None

    def __load_check(self):
        if self._loaded:
            return True
        self._loaded = True
        return False

    def __load_child_node(self, child_node: _EssenceNode):
        return super().add_child_node(child_node.name, child_node)

    def __load_drive(self):
        if self.__load_check():
            return
        d: SgaFsDrive = self._backing
        for folder in d.root.iter_folders:
            self.__load_child_node(_LazyEssenceNode(folder))
        for file in d.root.iter_files:
            self.__load_child_node(_LazyEssenceNode(file))

    def __load_folder(self):
        if self.__load_check():
            return
        d: SgaFsFolder = self._backing
        for folder in d.iter_folders:
            self.__load_child_node(_LazyEssenceNode(folder))
        for file in d.iter_files:
            self.__load_child_node(_LazyEssenceNode(file))

    def __load_file(self):
        if self.__load_check():
            return
        d: SgaFsFile = self._backing
        self._decomp_handle = d.data(decompress=True)
        self._comp_handle = d.data(decompress=False)

    def _load(self):
        raise NotImplementedError

    def add_child_node(self, child_path: str, child_node: _EssenceNode):
        self._load()
        return super().add_child_node(child_path, child_node)

    def get_child_node(self, child_path: str) -> Optional[_EssenceNode]:
        self._load()
        return super().get_child_node(child_path)

    def has_child(self, child_path):
        self._load()
        return super().has_child(child_path)

    def list_dir(self):
        self._load()
        return super().list_dir()

    def openbin(self, mode: Mode, lock: RLock, decompress:bool=True):
        self._load()
        handle = self._decomp_handle if decompress else self._comp_handle
        return _EssenceFileHandle(handle, mode, lock, name=self.name)

    def __repr__(self):
        _TYPENAMES = {ResourceType.file: "FILE", ResourceType.directory: "FOLD"}
        return f"<{self.__class__.__qualname__} [{_TYPENAMES.get(self._resource_type, 'UNK?')}] '{self.name}'>"

    def verify(self):
        return self._backing.verify()


class LazyEssenceFS(_EssenceFS):
    def __init__(self, stream: BinaryIO, fs_info: SgaFs):
        super().__init__()
        self._stream = stream
        self._streamlock = RLock()
        self._info = fs_info
        for drive in fs_info._drives:
            drive_node = _LazyEssenceNode(drive)
            self._roots[drive.path] = drive_node

    def __enter__(self):
        return self

    def close(self):  # type: () -> None
        self._stream.close()
        super().close()

    def __del__(self):
        self.close()
        return super().__del__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return super().__exit__(exc_type, exc_val, exc_tb)

    def _create_node(self, resource_type: ResourceType, name: str, *args, **kwargs):
        return _EssenceNode(resource_type, name)

    def openbin(self, path, mode="r", buffering=-1, **options):
        _mode = mode
        mode = Mode(mode)
        if not mode.reading and not mode.binary:
            raise NotImplementedError
        node: _LazyEssenceNode = self._get_node(path)
        return node.openbin(mode, self._streamlock, decompress=options.get("decompress",True))

    def verify(self, path: Optional[str] = None) -> bool:
        if path is None:
            return self._info.verify()
        else:
            node = self._get_node(path)
            if node.is_dir:
                raise errors.FileExpected(path)
            return node.verify()
