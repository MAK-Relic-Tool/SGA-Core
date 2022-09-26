from typing import Optional

from relic.sga.core.filesystem import EssenceFS

try:
    import pathlib
    from pathlib import Path, PurePath

    PATHLIB_SUPPORT = True
except ImportError:
    PATHLIB_SUPPORT = False
    Path = None
    PurePath = None

if PATHLIB_SUPPORT:
    _Flavour = pathlib._Flavour
    _WindowsFlavour = pathlib._WindowsFlavour
    _Accessor = pathlib._Accessor

    class _EssenceFlavour(_Flavour):
        # I'm under the assumption that SGA's "Flavour" is just simplified Windows, with 'drive' being any word

        sep = _WindowsFlavour.sep
        altsep = _WindowsFlavour.altsep
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

    # A quick glance didn't tell me how to construct a StatInfo
    #   So this relies on ducktyping
    class _EssenceStatResult:
        def __init__(self, **kwargs):
            for kw, arg in kwargs.items():
                setattr(self, "st_{kw}", property(lambda _: arg))

    class _EssenceAccessor(_Accessor):
        def __init__(self, fs: EssenceFS):
            self.fs = fs

        def _get_statresult(self, path: PurePath, namespace: str):
            info_dict = self.fs.getinfo(str(path), [namespace]).raw[namespace]
            return _EssenceStatResult(**info_dict)

        def stat(self, path: PurePath, *args, **kwargs):
            return self._get_statresult(path, "stat")

        def lstat(self, path: PurePath, *args, **kwargs):
            return self._get_statresult(path, "lstat")

        def open(self):
            raise NotImplementedError(
                "Cannot support os.open (files do not exist on OS); use _Accessor.fopen()"
            )

        def fopen(
            self,
            path: PurePath,
            mode: str = "r",
            buffering=-1,
            encoding: Optional[str] = None,
            errors: Optional[str] = None,
            newline: str = "",
            **kwargs
        ):
            return self.fs.open(
                str(path),
                mode=mode,
                buffering=buffering,
                encoding=encoding,
                errors=errors,
                newline=newline,
                **kwargs
            )

        def listdir(self, path: PurePath):
            return self.fs.listdir(str(path))

        def scandir(self, path: PurePath):
            return self.fs.scandir(str(path), namespaces=None)

        def chmod(self, path: PurePath, *args, **kwargs):
            raise NotImplementedError("chmod() is not supported")

        def lchmod(self, path: PurePath, *args, **kwargs):
            raise NotImplementedError("lchmod() is not supported")

        def mkdir(self, path: PurePath, *args, **kwargs):
            return self.fs.makedir(str(path))

        def unlink(self, path: PurePath, *args, **kwargs):
            raise NotImplementedError("unlink() is not supported")

        def link_to(self, path: PurePath, *args, **kwargs):
            raise NotImplementedError("link_to() is not supported")

        def rmdir(self, path: PurePath, *args, **kwargs):
            return self.fs.removedir(str(path))

        def rename(self, src: PurePath, dst: str, *args, **kwargs):
            return self.fs.move(str(src), dst)

        def replace(self, src: PurePath, dst: str, *args, **kwargs):
            return self.fs.move(str(src), dst, overwrite=True)

        def symlink(self, *args, **kwargs):
            raise NotImplementedError("symlink() is not supported")

        def utime(self, *args, **kwargs):
            raise NotImplementedError("os.utime is not supported")

        # Helper for resolve()
        def readlink(self, path):
            raise NotImplementedError("os.readlink() is not supported")

        def owner(self, path):
            raise NotImplementedError("Path.owner() is not supported")

        def group(self, path):
            raise NotImplementedError("Path.group() is not supported")

    _essence_flavour = _EssenceFlavour()

    class EssencePurePath(PurePath):
        _flavour = _essence_flavour
        __slots__ = ()

    class EssencePath(Path, EssencePurePath):
        def _init(self, template=None):
            if template is not None:
                self._accessor = template._accessor
            else:
                self._accessor = _EssenceAccessor(fs=self._fs)

        def __new__(cls, fs: EssenceFS, *args, **kwargs):
            self = cls._from_parts(args, init=False)
            if not self._flavour.is_supported:
                raise NotImplementedError(
                    "cannot instantiate %r on your system" % (cls.__name__,)
                )
            self._fs = fs
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
            parent_path._fs = self._fs
            parent_path._init()
            return parent_path

        def open(
            self, mode="r", buffering=-1, encoding=None, errors=None, newline=None
        ):
            return self._accessor.mem_open(self)

        def _opener(self, *args, **kwargs):
            raise NotImplementedError("EssencePath does not support _opener().")

        def _raw_open(self, *args, **kwargs):
            raise NotImplementedError("EssencePath does not support _raw_open().")

else:
    _Flavour = None
    _WindowsFlavour = None
    _Accessor = None
    _EssenceFlavour = None
    _EssenceStatResult = None
    _EssenceAccessor = None
    _essence_flavour = None

    class EssencePurePath:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "pathlib is not installed; please install pathlib (`pip install pathlib`) to use this feature."
            )

    class EssencePath:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "pathlib is not installed; please install pathlib (`pip install pathlib`) to use this feature."
            )
