from __future__ import annotations

from typing import BinaryIO, ClassVar, Tuple, Generic, Type, Optional, List, Protocol, Union, Iterable

from relic.sga.core import Version
from relic.sga.core.lazyio import LazyBinary, BinaryWindow, tell_end, T

def _safe_get_parent_name(parent:BinaryIO, default:Optional[str]=None):
    return default if not hasattr(parent,"name") else parent.name

class ArchivePtrs(Protocol):
    @property
    def toc_pos(self) -> int:
        raise NotImplementedError

    @property
    def toc_size(self) -> int:
        raise NotImplementedError

    @property
    def data_pos(self) -> int:
        raise NotImplementedError

    @property
    def data_size(self) -> Optional[int]:
        raise NotImplementedError


class SgaHeader(LazyBinary, ArchivePtrs):
    def __init__(self, parent: BinaryIO, *args, **kwargs):
        super().__init__(parent, *args, **kwargs, name="SGA Meta Block")

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def toc_pos(self) -> int:
        raise NotImplementedError

    @property
    def toc_size(self) -> int:
        raise NotImplementedError

    @property
    def data_pos(self) -> int:
        raise NotImplementedError

    @property
    def data_size(self) -> int:
        raise NotImplementedError



class SgaTocHeader(LazyBinary):
    _DRIVE_POS: ClassVar[Tuple[int, int]] = None
    _DRIVE_COUNT: ClassVar[Tuple[int, int]] = None
    _FOLDER_POS: ClassVar[Tuple[int, int]] = None
    _FOLDER_COUNT: ClassVar[Tuple[int, int]] = None
    _FILE_POS: ClassVar[Tuple[int, int]] = None
    _FILE_COUNT: ClassVar[Tuple[int, int]] = None
    _NAME_POS: ClassVar[Tuple[int, int]] = None
    _NAME_COUNT: ClassVar[Tuple[int, int]] = None

    class TablePointer:
        def __init__(self, container:LazyBinary, pos:Tuple[int,int], count:Tuple[int,int]):
            self._POS = pos
            self._COUNT = count
            self._container = container

        @property
        def offset(self):
            buffer = self._container._read_bytes(*self._POS)
            return self._container._unpack_int(buffer)

        @offset.setter
        def offset(self, value: int):
            buffer = self._container._pack_int(value, self._POS[1])
            self._container._write_bytes(buffer, *self._POS)

        @property
        def count(self):
            buffer = self._container._read_bytes(*self._COUNT)
            return self._container._unpack_int(buffer)

        @count.setter
        def count(self, value: int):
            buffer = self._container._pack_int(value, self._COUNT[1])
            self._container._write_bytes(buffer, *self._COUNT)

        @property
        def info(self) -> Tuple[int, int]:
            return self.offset, self.count

        @info.setter
        def info(self, value: Tuple[int, int]):
            pos, count = value
            self.offset = pos
            self.count = count

    def __init__(self, parent: BinaryIO, *args, **kwargs):
        super().__init__(
            parent, *args, **kwargs, close_parent=False, name="SGA ToC Header"
        )
        self._drive = self.TablePointer(self,self._DRIVE_POS,self._DRIVE_COUNT)
        self._folder = self.TablePointer(self,self._FOLDER_POS,self._FOLDER_COUNT)
        self._file = self.TablePointer(self,self._FILE_POS,self._FILE_COUNT)
        self._name = self.TablePointer(self,self._NAME_POS,self._NAME_COUNT)

    # DRIVE
    @property
    def drive(self) -> TablePointer:
        return self._drive

    @property
    def folder(self) -> TablePointer:
        return self._folder

    @property
    def file(self) -> TablePointer:
        return self._file

    @property
    def name(self) -> TablePointer:
        return self._name



class SgaTocDrive(LazyBinary):
    _ALIAS = None
    _NAME = None
    _FIRST_FOLDER = None
    _LAST_FOLDER = None
    _FIRST_FILE = None
    _LAST_FILE = None
    _ROOT_FOLDER = None
    _SIZE = None
    _STR_ENC = "ascii"
    _STR_PAD = "\0"

    def __init__(self, parent: BinaryIO, *args, **kwargs):
        super().__init__(
            parent,
            *args,
            **kwargs,
            close_parent=False,
            name="SGA Toc Drive ['Alias Not Loaded (Initing)']",
        )
        self._name = f"SGA Toc Drive ['{self.alias}']"

    @property
    def alias(self):
        buffer = self._read_bytes(*self._ALIAS)
        terminated_str = self._unpack_str(buffer, self._STR_ENC, strip=self._STR_PAD)
        result = terminated_str.rstrip("\0")
        return result

    @alias.setter
    def alias(self, value:str):
        buffer = self._pack_str(value, self._STR_ENC, size=self._ALIAS[1], padding=self._STR_PAD)
        self._write_bytes(buffer,*self._ALIAS)

    @property
    def name(self):
        buffer = self._read_bytes(*self._NAME)
        terminated_str = self._unpack_str(buffer, self._STR_ENC, strip=self._STR_PAD)
        result = terminated_str.rstrip("\0")
        return result

    @name.setter
    def name(self, value:str):
        buffer = self._pack_str(value, self._STR_ENC, size=self._NAME[1], padding=self._STR_PAD)
        self._write_bytes(buffer,*self._NAME)

    @property
    def first_folder(self):
        buffer = self._read_bytes(*self._FIRST_FOLDER)
        return self._unpack_int(buffer)

    @first_folder.setter
    def first_folder(self, value:int):
        buffer = self._pack_int(value,self._FIRST_FOLDER[1])
        self._write_bytes(buffer,*self._FIRST_FOLDER)

    @property
    def last_folder(self):
        buffer = self._read_bytes(*self._LAST_FOLDER)
        return self._unpack_int(buffer)

    @last_folder.setter
    def last_folder(self, value:int):
        buffer = self._pack_int(value,self._LAST_FOLDER[1])
        self._write_bytes(buffer,*self._LAST_FOLDER)

    @property
    def first_file(self):
        buffer = self._read_bytes(*self._FIRST_FILE)
        return self._unpack_int(buffer)

    @first_file.setter
    def first_file(self, value:int):
        buffer = self._pack_int(value,self._FIRST_FILE[1])
        self._write_bytes(buffer,*self._FIRST_FILE)

    @property
    def last_file(self):
        buffer = self._read_bytes(*self._LAST_FILE)
        return self._unpack_int(buffer)

    @last_file.setter
    def last_file(self, value:int):
        buffer = self._pack_int(value,self._LAST_FILE[1])
        self._write_bytes(buffer,*self._LAST_FILE)

    @property
    def root_folder(self):
        buffer = self._read_bytes(*self._ROOT_FOLDER)
        return self._unpack_int(buffer)

    @root_folder.setter
    def root_folder(self, value:int):
        buffer = self._pack_int(value,self._ROOT_FOLDER[1])
        self._write_bytes(buffer,*self._ROOT_FOLDER)


class SgaTocFolder(LazyBinary):
    _NAME_OFFSET = None
    _SUB_FOLDER_START = None
    _SUB_FOLDER_STOP = None
    _FIRST_FILE = None
    _LAST_FILE = None
    _SIZE = None

    def __init__(self, parent: BinaryIO, *args, **kwargs):
        super().__init__(
            parent, *args, **kwargs, close_parent=False, name="SGA Toc Folder []"
        )
        self._name = f"SGA Toc Folder ['{self.name}']"

    @property
    def name_offset(self):
        buffer = self._read_bytes(*self._NAME_OFFSET)
        result = self._unpack_int(buffer)
        return result

    @name_offset.setter
    def name_offset(self, value):
        buffer = self._pack_int(value,self._NAME_OFFSET[1])
        self._write_bytes(buffer,*self._NAME_OFFSET)

    @property
    def first_folder(self):
        buffer = self._read_bytes(*self._SUB_FOLDER_START)
        return self._unpack_int(buffer)

    @first_folder.setter
    def first_folder(self, value):
        buffer = self._pack_int(value, self._SUB_FOLDER_START[1])
        self._write_bytes(buffer, * self._SUB_FOLDER_START)

    @property
    def last_folder(self):
        buffer = self._read_bytes(*self._SUB_FOLDER_STOP)
        return self._unpack_int(buffer)

    @last_folder.setter
    def last_folder(self, value):
        buffer = self._pack_int(value, self._SUB_FOLDER_STOP[1])
        self._write_bytes(buffer, * self._SUB_FOLDER_STOP)

    @property
    def first_file(self):
        buffer = self._read_bytes(*self._FIRST_FILE)
        return self._unpack_int(buffer)

    @first_file.setter
    def first_file(self, value):
        buffer = self._pack_int(value, self._FIRST_FILE[1])
        self._write_bytes(buffer, * self._FIRST_FILE)

    @property
    def last_file(self):
        buffer = self._read_bytes(*self._LAST_FILE)
        return self._unpack_int(buffer)

    @last_file.setter
    def last_file(self, value):
        buffer = self._pack_int(value, self._LAST_FILE[1])
        self._write_bytes(buffer, *self._LAST_FILE)


class SgaNameWindow(BinaryWindow):
    def __init__(
        self,
        parent: BinaryIO,
        offset: int,
        count: int,
        length_mode: bool = False,
        encoding: str = "utf-8",
    ):
        self._encoding = encoding
        self._cacheable = not parent.writable()
        self.length_mode = length_mode
        size = count if length_mode else tell_end(parent)
        super().__init__(parent, offset, size, name="SGA ToC Name Buffer")
        self._cache = None
        self._init_cache()

    def _init_cache(self):
        if not self._cacheable:
            return
        if self._cache is None:
            self._cache = {}

        # Length mode can preload the cache
        if self.length_mode:
            self.seek(0)
            buffer = self.read()
            names:List[bytes] = buffer.split(b"\0")
            counter = 0
            for name in names:
                self._cache[counter] = name.decode(self._encoding)
                counter += len(name) + 1  # +1 for "\0"

    @staticmethod
    def _read_until_terminal(
        stream: BinaryIO, start: int, buffer_size: int = 64, terminal: bytes = b"\x00"
    ):
        parts = []
        stream.seek(start)
        while True:
            buffer = stream.read(buffer_size)
            split = buffer.split(terminal, maxsplit=1)
            parts.append(split[0])
            if len(split) > 1:
                break
        return b"".join(parts)

    def get_name(self, name_offset: int) -> str:
        if self._cache is not None and name_offset in self._cache:
            return self._cache[name_offset]

        name_buffer = self._read_until_terminal(self, name_offset)
        name = name_buffer.decode(self._encoding)

        if self._cache is not None:
            self._cache[name_offset] = name

        return name


class SgaTocInfoArea(Generic[T]):
    def __init__(self, parent: BinaryIO, offset: int, count: int, cls: Type[T]):
        self._parent = parent
        self._cls = cls
        self._windows = {}
        self._info_offset = offset
        self._info_count = count

    def __get_window(self, index: int) -> T:
        offset, count = self._info_offset, self._info_count
        if not (0 <= index < count):
            raise IndexError(index, f"Valid indexes are ['{0}', '{count}')")

        if index not in self._windows:
            self._windows[index] = self._cls(
                BinaryWindow(
                    self._parent,
                    offset + self._cls._SIZE * index,
                    self._cls._SIZE,
                    name=f"SGA ToC Info Area ['{index}']",
                )
            )

        return self._windows[index]

    def __getitem__(self, item: Union[int, slice]) -> Union[T, List[T]]:
        if isinstance(item, slice):
            return list(
                self.__get_window(index) for index in item.indices(self._info_count)
            )
        else:
            return self.__get_window(item)

    def __len__(self):
        return self._info_count

    def __iter__(self) -> Iterable[T]:
        for _ in range(self._info_count):
            yield self[_]


class SgaTocFile:
    @property
    def name_offset(self):
        raise NotImplementedError

    @property
    def data_offset(self):
        raise NotImplementedError

    @property
    def compressed_size(self):  # length_in_archive
        raise NotImplementedError

    @property
    def decompressed_size(self):  # length_on_disk
        raise NotImplementedError

    @property
    def storage_type(self):
        raise NotImplementedError


class SgaToc(LazyBinary):
    def __init__(self, parent: BinaryIO):
        super().__init__(parent, name="SGA ToC")

    @property
    def header(self) -> SgaTocHeader:
        raise NotImplementedError

    @property
    def drives(self) -> SgaTocInfoArea[SgaTocDrive]:
        raise NotImplementedError

    @property
    def folders(self) -> SgaTocInfoArea[SgaTocFolder]:
        raise NotImplementedError

    @property
    def files(self) -> SgaTocInfoArea[SgaTocFile]:
        raise NotImplementedError

    @property
    def names(self) -> SgaNameWindow:
        raise NotImplementedError


class SgaFile(LazyBinary):
    _MAGIC_WORD = (0, 8)
    _VERSION = (8, 4)
    _MAGIC_VERSION_SIZE = 12

    def __init__(self, parent: BinaryIO):
        super().__init__(parent, close_parent=False, name=f"SGA File ['{_safe_get_parent_name(parent)}']")

    @property
    def magic_word(self) -> bytes:
        return self._read_bytes(*self._MAGIC_WORD)

    @property
    def version(self) -> Version:
        buffer = self._read_bytes(*self._VERSION)
        major = self._unpack_uint16(buffer[:2])
        minor = self._unpack_uint16(buffer[2:])
        return Version(major, minor)

    @version.setter
    def version(self, value: Version):
        major = self._pack_uint16(value.major)
        minor = self._pack_uint16(value.minor)
        buffer = b"".join([major, minor])
        self._write_bytes(buffer, *self._VERSION)

    @property
    def meta(self) -> SgaHeader:
        raise NotImplementedError

    @property
    def table_of_contents(self) -> SgaToc:
        raise NotImplementedError

    @property
    def data_block(self) -> BinaryWindow:
        raise NotImplementedError
