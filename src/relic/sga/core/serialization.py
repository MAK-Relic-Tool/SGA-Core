from __future__ import annotations

from io import BytesIO
from typing import (
    BinaryIO,
    ClassVar,
    Tuple,
    Generic,
    Type,
    Optional,
    List,
    Protocol,
    Union,
    TypeVar,
    Dict,
    Literal,
    Iterator,
)

from relic.core.errors import RelicToolError
from relic.core.lazyio import (
    BinaryWindow,
    tell_end,
    BinaryProxySerializer,
    BinaryProxy,
    BinaryWrapper,
    BinarySerializer,
)

from relic.sga.core.definitions import Version, StorageType

_T = TypeVar("_T")

_NULL_PTR = (None, None)


def _safe_get_parent_name(
    parent: BinaryIO, default: Optional[str] = None
) -> Optional[str]:
    return default if not hasattr(parent, "name") else parent.name


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


class SgaHeader(BinaryProxySerializer, ArchivePtrs):
    def __init__(self, parent: BinaryIO):
        super().__init__(parent)

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


class SgaTocHeader(BinaryProxySerializer):
    _DRIVE_POS: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _DRIVE_COUNT: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _FOLDER_POS: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _FOLDER_COUNT: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _FILE_POS: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _FILE_COUNT: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _NAME_POS: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _NAME_COUNT: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore

    class TablePointer:
        def __init__(
            self, parent: SgaTocHeader, pos: Tuple[int, int], count: Tuple[int, int]
        ):
            self._offset_ptr = pos
            self._count_ptr = count
            self._serializer = parent._serializer

        @property
        def offset(self) -> int:
            return self._serializer.int.read(*self._offset_ptr)

        @offset.setter
        def offset(self, value: int) -> None:
            self._serializer.int.write(value, *self._offset_ptr)

        @property
        def count(self) -> int:
            return self._serializer.int.read(*self._count_ptr)

        @count.setter
        def count(self, value: int) -> None:
            self._serializer.int.write(value, *self._count_ptr)

        @property
        def info(self) -> Tuple[int, int]:
            return self.offset, self.count

        @info.setter
        def info(self, value: Tuple[int, int]) -> None:
            pos, count = value
            self.offset = pos
            self.count = count

    # Non-breaking change to init; old-style class-var init still works
    def __init__(
        self,
        parent: BinaryIO,
        drive_pos_ptr: Optional[Tuple[int, int]] = None,
        drive_count_ptr: Optional[Tuple[int, int]] = None,
        folder_pos_ptr: Optional[Tuple[int, int]] = None,
        folder_count_ptr: Optional[Tuple[int, int]] = None,
        file_pos_ptr: Optional[Tuple[int, int]] = None,
        file_count_ptr: Optional[Tuple[int, int]] = None,
        name_pos_ptr: Optional[Tuple[int, int]] = None,
        name_count_ptr: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(
            parent,
        )

        self._drive = self.TablePointer(
            self, drive_pos_ptr or self._DRIVE_POS, drive_count_ptr or self._DRIVE_COUNT
        )
        self._folder = self.TablePointer(
            self,
            folder_pos_ptr or self._FOLDER_POS,
            folder_count_ptr or self._FOLDER_COUNT,
        )
        self._file = self.TablePointer(
            self, file_pos_ptr or self._FILE_POS, file_count_ptr or self._FILE_COUNT
        )
        self._name = self.TablePointer(
            self, name_pos_ptr or self._NAME_POS, name_count_ptr or self._NAME_COUNT
        )

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


class SgaTocDrive(BinaryProxySerializer):
    _ALIAS: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _NAME: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _FIRST_FOLDER: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _LAST_FOLDER: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _FIRST_FILE: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _LAST_FILE: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _ROOT_FOLDER: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _SIZE: ClassVar[int] = _NULL_PTR  # type: ignore
    _INT_BYTEORDER: ClassVar[Literal["little"]] = "little"
    _INT_SIGNED: ClassVar[bool] = False
    _STR_ENC = "ascii"
    _STR_PAD = "\0"

    def __init__(self, parent: BinaryIO):
        super().__init__(
            parent,
        )

    @property
    def alias(self) -> str:
        return self._serializer.c_string.read(
            *self._ALIAS, encoding=self._STR_ENC, padding=self._STR_PAD
        )

    @alias.setter
    def alias(self, value: str) -> None:
        self._serializer.c_string.write(
            value, *self._ALIAS, encoding=self._STR_ENC, padding=self._STR_PAD
        )

    @property
    def name(self) -> str:
        return self._serializer.c_string.read(
            *self._NAME, encoding=self._STR_ENC, padding=self._STR_PAD
        )

    @name.setter
    def name(self, value: str) -> None:
        self._serializer.c_string.write(
            value, *self._NAME, encoding=self._STR_ENC, padding=self._STR_PAD
        )

    @property
    def first_folder(self) -> int:
        return self._serializer.int.read(
            *self._FIRST_FOLDER, byteorder=self._INT_BYTEORDER, signed=self._INT_SIGNED
        )

    @first_folder.setter
    def first_folder(self, value: int) -> None:
        self._serializer.int.write(
            value,
            *self._FIRST_FOLDER,
            byteorder=self._INT_BYTEORDER,
            signed=self._INT_SIGNED,
        )

    @property
    def last_folder(self) -> int:
        return self._serializer.int.read(
            *self._LAST_FOLDER, byteorder=self._INT_BYTEORDER, signed=self._INT_SIGNED
        )

    @last_folder.setter
    def last_folder(self, value: int) -> None:
        self._serializer.int.write(
            value,
            *self._LAST_FOLDER,
            byteorder=self._INT_BYTEORDER,
            signed=self._INT_SIGNED,
        )

    @property
    def first_file(self) -> int:
        return self._serializer.int.read(
            *self._FIRST_FILE, byteorder=self._INT_BYTEORDER, signed=self._INT_SIGNED
        )

    @first_file.setter
    def first_file(self, value: int) -> None:
        self._serializer.int.write(
            value,
            *self._FIRST_FILE,
            byteorder=self._INT_BYTEORDER,
            signed=self._INT_SIGNED,
        )

    @property
    def last_file(self) -> int:
        return self._serializer.int.read(
            *self._LAST_FILE, byteorder=self._INT_BYTEORDER, signed=self._INT_SIGNED
        )

    @last_file.setter
    def last_file(self, value: int) -> None:
        self._serializer.int.write(
            value,
            *self._LAST_FILE,
            byteorder=self._INT_BYTEORDER,
            signed=self._INT_SIGNED,
        )

    @property
    def root_folder(self) -> int:
        return self._serializer.int.read(
            *self._ROOT_FOLDER, byteorder=self._INT_BYTEORDER, signed=self._INT_SIGNED
        )

    @root_folder.setter
    def root_folder(self, value: int) -> None:
        self._serializer.int.write(
            value,
            *self._ROOT_FOLDER,
            byteorder=self._INT_BYTEORDER,
            signed=self._INT_SIGNED,
        )


class SgaTocFolder(BinaryProxySerializer):
    _NAME_OFFSET: ClassVar[Tuple[int, int]] = (None, None)  # type: ignore
    _SUB_FOLDER_START: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _SUB_FOLDER_STOP: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _FIRST_FILE: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _LAST_FILE: ClassVar[Tuple[int, int]] = _NULL_PTR  # type: ignore
    _SIZE: ClassVar[int] = _NULL_PTR  # type: ignore
    _INT_BYTEORDER: ClassVar[Literal["little"]] = "little"
    _INT_SIGNED: ClassVar[bool] = False

    def __init__(self, parent: BinaryIO):
        super().__init__(parent)

    @property
    def name_offset(self) -> int:
        return self._serializer.int.read(
            *self._NAME_OFFSET, byteorder=self._INT_BYTEORDER, signed=self._INT_SIGNED
        )

    @name_offset.setter
    def name_offset(self, value: int) -> None:
        self._serializer.int.write(
            value,
            *self._NAME_OFFSET,
            byteorder=self._INT_BYTEORDER,
            signed=self._INT_SIGNED,
        )

    @property
    def first_folder(self) -> int:
        return self._serializer.int.read(
            *self._SUB_FOLDER_START,
            byteorder=self._INT_BYTEORDER,
            signed=self._INT_SIGNED,
        )

    @first_folder.setter
    def first_folder(self, value: int) -> None:
        self._serializer.int.write(
            value,
            *self._SUB_FOLDER_START,
            byteorder=self._INT_BYTEORDER,
            signed=self._INT_SIGNED,
        )

    @property
    def last_folder(self) -> int:
        return self._serializer.int.read(
            *self._SUB_FOLDER_STOP,
            byteorder=self._INT_BYTEORDER,
            signed=self._INT_SIGNED,
        )

    @last_folder.setter
    def last_folder(self, value: int) -> None:
        self._serializer.int.write(
            value,
            *self._SUB_FOLDER_STOP,
            byteorder=self._INT_BYTEORDER,
            signed=self._INT_SIGNED,
        )

    @property
    def first_file(self) -> int:
        return self._serializer.int.read(
            *self._FIRST_FILE, byteorder=self._INT_BYTEORDER, signed=self._INT_SIGNED
        )

    @first_file.setter
    def first_file(self, value: int) -> None:
        self._serializer.int.write(
            value,
            *self._FIRST_FILE,
            byteorder=self._INT_BYTEORDER,
            signed=self._INT_SIGNED,
        )

    @property
    def last_file(self) -> int:
        return self._serializer.int.read(
            *self._LAST_FILE, byteorder=self._INT_BYTEORDER, signed=self._INT_SIGNED
        )

    @last_file.setter
    def last_file(self, value: int) -> None:
        self._serializer.int.write(
            value,
            *self._LAST_FILE,
            byteorder=self._INT_BYTEORDER,
            signed=self._INT_SIGNED,
        )


class SgaNameWindow(BinaryProxySerializer):
    def __init__(
        self,
        parent: BinaryIO,
        offset: int,
        count: int,
        length_mode: bool = False,
        encoding: str = "utf-8",
    ) -> None:
        size = count if length_mode else tell_end(parent)
        self._window = BinaryWindow(parent, offset, size, name="SGA ToC Name Buffer")
        super().__init__(self._window)
        self._count = count if not length_mode else None

        self._encoding = encoding
        self._cacheable = parent.readable() and not parent.writable()
        self.length_mode = length_mode

        self._cache: Optional[Dict[int, str]] = None
        self._init_cache()

    def _init_cache(self) -> None:
        if not self._cacheable:
            return
        if self._cache is None:
            self._cache = {}

        # Length mode can preload the cache
        if self.length_mode:
            self._serializer.stream.seek(0)
            buffer = self._serializer.stream.read()
            names: List[bytes] = buffer.split(b"\0")
            counter = 0
            for name in names:
                self._cache[counter] = name.decode(self._encoding)
                counter += len(name) + 1  # +1 for "\0"
            self._count = len(self._cache)

    @staticmethod
    def _read_until_terminal(
        stream: BinaryIO, start: int, buffer_size: int = 64, terminal: bytes = b"\x00"
    ) -> bytes:
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

        name_buffer = self._read_until_terminal(self._serializer.stream, name_offset)
        name = name_buffer.decode(self._encoding)

        if self._cache is not None:
            self._cache[name_offset] = name

        return name


_TocWindowCls = TypeVar("_TocWindowCls", BinaryProxySerializer, BinaryWrapper)


class SgaTocInfoArea(Generic[_TocWindowCls]):
    def __init__(
        self,
        parent: Union[BinaryIO, BinaryProxy],
        offset: int,
        count: int,
        cls: Type[_TocWindowCls],
        cls_size: Optional[int] = None,
    ) -> None:
        self._parent = parent
        self._cls: Type[_TocWindowCls] = cls
        if hasattr(self._cls, "_SIZE"):
            self._cls_size = self._cls._SIZE
        elif cls_size is not None:
            self._cls_size = cls_size
        else:
            raise RelicToolError("TOC Window size could not be determined!")

        self._windows: Dict[int, _TocWindowCls] = {}
        self._info_offset = offset
        self._info_count = count

    def __get_window(self, index: int) -> _TocWindowCls:
        offset, count = self._info_offset, self._info_count
        if not 0 <= index < count:
            raise IndexError(index, f"Valid indexes are ['{0}', '{count}')")

        if index not in self._windows:
            self._windows[index] = self._cls(
                BinaryWindow(
                    self._parent,
                    offset + self._cls_size * index,
                    self._cls_size,
                    name=f"SGA ToC Info Area ['{index}']",
                )
            )

        return self._windows[index]

    def __getitem__(
        self, item: Union[int, slice]
    ) -> Union[_TocWindowCls, List[_TocWindowCls]]:
        if isinstance(item, slice):
            return list(
                self.__get_window(index)
                for index in range(*item.indices(self._info_count))
            )
        return self.__get_window(item)

    def __len__(self) -> int:
        return self._info_count

    def __iter__(self) -> Iterator[_TocWindowCls]:
        for index in range(self._info_count):
            yield self[index]  # type: ignore


class SgaTocFile:
    @property
    def name_offset(self) -> int:
        raise NotImplementedError

    @property
    def data_offset(self) -> int:
        raise NotImplementedError

    @property
    def compressed_size(self) -> int:  # length_in_archive
        raise NotImplementedError

    @property
    def decompressed_size(self) -> int:  # length_on_disk
        raise NotImplementedError

    @property
    def storage_type(self) -> StorageType:
        raise NotImplementedError


class SgaToc(BinaryProxySerializer):
    def __init__(self, parent: BinaryIO):
        super().__init__(parent)

    @property
    def header(self) -> SgaTocHeader:
        raise NotImplementedError

    @property
    def drives(self) -> SgaTocInfoArea[SgaTocDrive]:  # type: ignore
        raise NotImplementedError

    @property
    def folders(self) -> SgaTocInfoArea[SgaTocFolder]:  # type: ignore
        raise NotImplementedError

    @property
    def files(self) -> SgaTocInfoArea[SgaTocFile]:  # type: ignore
        raise NotImplementedError

    @property
    def names(self) -> SgaNameWindow:
        raise NotImplementedError


class SgaFile(BinaryProxySerializer):
    _MAGIC_WORD = (0, 8)
    _VERSION = (8, 4)
    _MAGIC_VERSION_SIZE = 12
    _VERSION_INT_FMT = {"byteorder": "little", "signed": False}

    @property
    def magic_word(self) -> bytes:
        return self._serializer.read_bytes(*self._MAGIC_WORD)

    @property
    def version(self) -> Version:
        buffer = self._serializer.read_bytes(*self._VERSION)
        major = self._serializer.uint16.unpack(buffer[:2], **self._VERSION_INT_FMT)  # type: ignore
        minor = self._serializer.uint16.unpack(buffer[2:], **self._VERSION_INT_FMT)  # type: ignore
        return Version(major, minor)

    @version.setter
    def version(self, value: Version) -> None:
        major = self._serializer.uint16.pack(value.major, **self._VERSION_INT_FMT)  # type: ignore
        minor = self._serializer.uint16.pack(value.minor, **self._VERSION_INT_FMT)  # type: ignore
        buffer = b"".join([major, minor])
        self._serializer.write_bytes(buffer, *self._VERSION)

    @property
    def meta(self) -> SgaHeader:
        raise NotImplementedError

    @property
    def table_of_contents(self) -> SgaToc:
        raise NotImplementedError

    @property
    def data_block(self) -> BinaryWindow:
        raise NotImplementedError


class VersionSerializer:
    _INT_SIZE = 2
    _MAJOR = (0, _INT_SIZE)
    _MINOR = (2, _INT_SIZE)
    _SIZE = _INT_SIZE * 2
    _INT_BYTEORDER: ClassVar[Literal["little"]] = "little"
    _INT_SIGNED = False

    @classmethod
    def unpack(cls, buffer: bytes) -> Version:
        with BytesIO(buffer) as reader:
            serializer = BinarySerializer(reader)
            major = serializer.uint16.read(
                *cls._MAJOR, byteorder=cls._INT_BYTEORDER, signed=cls._INT_SIGNED
            )
            minor = serializer.uint16.read(
                *cls._MINOR, byteorder=cls._INT_BYTEORDER, signed=cls._INT_SIGNED
            )
            return Version(major, minor)

    @classmethod
    def read(cls, stream: BinaryIO) -> Version:
        buffer = stream.read(cls._SIZE)
        return cls.unpack(buffer)

    @classmethod
    def pack(cls, version: Version) -> bytes:
        with BytesIO(b"\0" * cls._SIZE) as writer:
            serializer = BinarySerializer(writer)
            serializer.uint16.write(
                version.major,
                *cls._MAJOR,
                byteorder=cls._INT_BYTEORDER,
                signed=cls._INT_SIGNED,
            )
            serializer.uint16.write(
                version.minor,
                *cls._MINOR,
                byteorder=cls._INT_BYTEORDER,
                signed=cls._INT_SIGNED,
            )
            return writer.getvalue()

    @classmethod
    def write(cls, stream: BinaryIO, version: Version) -> int:
        buffer = cls.pack(version)
        return stream.write(buffer)
