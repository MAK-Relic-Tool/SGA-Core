import os
from contextlib import contextmanager
from pathlib import PurePath, Path
import pathlib
from typing import Optional, List, Union, Type

from relic.sga.core.abstract import File, Archive, Folder, Drive
from relic.sga.core.protocols import IOContainer


class _SGAFlavour(pathlib._Flavour):
    # I'm under the assumption that SGA's "Flavour" is just simplified Windows, with 'drive' being any word

    sep = pathlib._WindowsFlavour.sep
    altsep = pathlib._WindowsFlavour.altsep
    is_supported = True

    def splitroot(self, part: str, sep=sep):
        # Windows has alot of clever do-dads here,
        #   I don't want to dig through it and do it perfectly; so for now,
        #       A "Good enough" solution
        has_drive = ":" in part
        drive, root = "", ""
        if has_drive:
            drive_split = part.split(":")
            if len(drive_split) > 2:
                raise NotImplementedError
            drive = drive_split[0] + ":"
            part = drive_split[1]

        if part[0] == sep:
            root = part[0]
            part = part.lstrip(sep)

        return drive, root, part


class _SGAStatResult:
    def __init__(self, size: int):
        self.st_size = property(lambda _: size)


class _SGAAccessor(pathlib._Accessor):
    def __init__(self, archive: Archive):
        self.archive = archive

    def _resolve_root(self, path: PurePath) -> Union[Drive, Folder, File]:
        archive = self.archive

        if path.drive is not None:
            for drive in archive.drives:
                if drive.name == path.drive:
                    if len(path.parts) == 1:
                        return drive
                    else:
                        rel_path = path.relative_to(drive.name)
                        return self._resolve_container(drive, rel_path)
            raise FileNotFoundError
        else:
            for drive in archive.drives:
                try:
                    if len(path.parts) == 1:
                        return drive
                    else:
                        rel_path = path.relative_to(drive.name)
                        return self._resolve_container(drive, rel_path)
                except FileNotFoundError:
                    continue
            raise FileNotFoundError

    def _resolve_drive(
        self, drive: Drive, path: PurePath
    ) -> Union[Drive, Folder, File]:
        rel_path = path.relative_to(container.name)
        return self._resolve_container(drive, rel_path)

    def _resolve_container(
        self, container: Folder, path: PurePath
    ) -> Union[Folder, File]:

        rel_path = path.relative_to(container.name)
        return self._resolve_container(drive, rel_path)

    def _get_node(self, path, node_cls: Type) -> Union[Folder, File, Drive]:
        node = self._resolve_root(path)
        if node is None:
            raise FileNotFoundError
        elif not isinstance(node, node_cls):
            raise TypeError
        return node

    def _get_file(self, path) -> File:
        return self._get_node(path, File)

    def _get_folder(self, path) -> Folder:
        return self._get_node(path, Folder)

    def _get_drive(self, path) -> Drive:
        return self._get_node(path, Drive)

    def stat(self, pathobj, *args, **kwargs):
        file = self._get_file(pathobj)
        try:
            data: bytes = file.data
            size = len(data)
        except TypeError:
            size = -1
        return _SGAStatResult(size)

    def lstat(self, *args, **kwargs):
        raise NotImplementedError("lstat() is not supported in SGA")

    def open(self, *args, **kwargs) -> int:
        raise NotImplementedError("open() is not supported in SGA; use mem_open.")

    @contextmanager
    def mem_open(self, path, *args, **kwargs):
        file = self._get_file(path)
        yield file.open(read_only=False)

    def listdir(self, *args, **kwargs) -> List[str]:
        raise NotImplementedError("listdir() is not supported in SGA")

    def scandir(self, *args, **kwargs) -> List[str]:
        raise NotImplementedError(
            "scandir() is not supported in SGA; an implementation is needed"
        )

    def chmod(self, *args, **kwargs):
        raise NotImplementedError(
            "chmod() is not supported; SGA Files lack permission information."
        )

    def lchmod(self, path, mode):
        raise NotImplementedError(
            "lchmod() is not supported; SGA Files lack permission information."
        )

    def mkdir(self, path: PurePath, *args, **kwargs):
        parent = self._get_dir(path.parent)
        for sub_folder in parent.sub_folders:
            if sub_folder.path.name == path.name:
                raise FileExistsError
        for file in parent.files:
            if file.path.name == path.name:
                raise FileExistsError

        new_dir = Folder(path.name, [], [], parent=parent)
        parent.sub_folders.append(new_dir)

    def unlink(self, *args, **kwargs):
        raise NotImplementedError(
            "unlink() is not supported; an implementation is needed."
        )

    def link(self, *args, **kwargs):
        raise NotImplementedError(
            "link() is not supported; an implementation is needed."
        )

    def rmdir(self, *args, **kwargs):
        raise NotImplementedError(
            "rmdir() is not supported; an implementation is needed."
        )

    def rename(self, *args, **kwargs):
        raise NotImplementedError(
            "rename() is not supported; an implementation is needed."
        )

    def replace(self, *args, **kwargs):
        raise NotImplementedError(
            "replace() is not supported; an implementation is needed."
        )

    def symlink(self, *args, **kwargs):
        raise NotImplementedError(
            "symlink() is not supported; an implementation is needed."
        )

    def readlink(self, *args, **kwargs):
        raise NotImplementedError(
            "readlink() is not supported; an implementation is needed."
        )

    def utime(self, *args, **kwargs):
        raise NotImplementedError(
            "utime() is not supported; an implementation is needed."
        )

    def owner(self, *args, **kwargs):
        raise NotImplementedError(
            "owner() is not supported; an implementation is needed."
        )

    def group(self, *args, **kwargs):
        raise NotImplementedError(
            "group() is not supported; an implementation is needed."
        )


_sga_flavour = _SGAFlavour()


# _sga_accessor = _SGAAccessor()


class PureSGAPath(PurePath):
    _flavour = _sga_flavour
    __slots__ = ()


class SGAPath(Path, PureSGAPath):
    def _init(self, template=None):
        if template is not None:
            self._accessor = template._accessor
        else:
            self._accessor = _SGAAccessor(archive=self._archive)

    def __new__(cls, archive, *args, **kwargs):
        self = cls._from_parts(args, init=False)
        if not self._flavour.is_supported:
            raise NotImplementedError(
                "cannot instantiate %r on your system" % (cls.__name__,)
            )
        self._archive = archive
        self._init()
        return self

    @property
    def parent(self):
        """The logical parent of the path."""
        drv = self.drive
        root = self.root
        parts = self.parts
        if len(parts) == 1 and (drv or root):
            return self
        parent_path = self._from_parsed_parts(drv, root, parts[:-1], init=False)
        parent_path._archive = self._archive
        parent_path._init()
        return parent_path

    def open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        return self._accessor.mem_open(self)

    def _opener(self, *args, **kwargs):
        raise NotImplementedError("SGA does not support _opener().")

    def _raw_open(self, *args, **kwargs):
        raise NotImplementedError("SGA does not support _raw_open().")


if __name__ == "__main__":
    psgap = PureSGAPath("data:root/folder/file.txt")
    archive = Archive("Test", None, [])
    sgap = SGAPath(archive, str(psgap))
    print(sgap.name)
    print(sgap.root)
    print(sgap.drive)
    try:
        print(sgap.stat())
    except FileNotFoundError:
        ...

    try:
        print(sgap.mkdir())
    except FileNotFoundError:
        ...

    with sgap.open() as handle:
        print(handle.read())
    pass
