from __future__ import annotations

import hashlib
import typing
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePath
from typing import (
    BinaryIO,
    List,
    Dict,
    Optional,
    Callable,
    Tuple,
    Iterable,
    TypeVar,
    Generic,
)

from fs.base import FS

from serialization_tools.size import KiB, MiB
from serialization_tools.structx import Struct

from relic.sga.core.definitions import (
    StorageType,
    Version,
    MagicWord,
    _validate_magic_word,
)
from relic.sga.core.errors import (
    MD5MismatchError,
    VersionMismatchError,
    DecompressedSizeMismatch,
)
from relic.sga.core.filesystem import EssenceFS, _EssenceDriveFS, EssenceFSHandler
from relic.sga.core.protocols import StreamSerializer, T


@dataclass
class TocBlock:
    drive_info: Tuple[int, int]
    folder_info: Tuple[int, int]
    file_info: Tuple[int, int]
    name_info: Tuple[int, int]

    @classmethod
    def default(cls) -> TocBlock:
        null_pair = (0, 0)
        return cls(null_pair, null_pair, null_pair, null_pair)


class TocHeaderSerializer(StreamSerializer[TocBlock]):
    def __init__(self, layout: Struct):
        self.layout = layout

    def unpack(self, stream: BinaryIO) -> TocBlock:
        (
            drive_pos,
            drive_count,
            folder_pos,
            folder_count,
            file_pos,
            file_count,
            name_pos,
            name_count,
        ) = self.layout.unpack_stream(stream)

        return TocBlock(
            (drive_pos, drive_count),
            (folder_pos, folder_count),
            (file_pos, file_count),
            (name_pos, name_count),
        )

    def pack(self, stream: BinaryIO, value: TocBlock) -> int:
        args = (
            value.drive_info[0],
            value.drive_info[1],
            value.folder_info[0],
            value.folder_info[1],
            value.file_info[0],
            value.file_info[1],
            value.name_info[0],
            value.name_info[1],
        )
        packed: int = self.layout.pack_stream(stream, *args)
        return packed


@dataclass
class DriveDef:
    alias: str
    name: str
    root_folder: int
    folder_range: Tuple[int, int]
    file_range: Tuple[int, int]


class DriveDefSerializer(StreamSerializer[DriveDef]):
    def __init__(self, layout: Struct):
        self.layout = layout

    def unpack(self, stream: BinaryIO) -> DriveDef:
        encoded_alias: bytes
        encoded_name: bytes
        (
            encoded_alias,
            encoded_name,
            folder_start,
            folder_end,
            file_start,
            file_end,
            root_folder,
        ) = self.layout.unpack_stream(stream)
        alias: str = encoded_alias.rstrip(b"\0").decode("ascii")
        name: str = encoded_name.rstrip(b"\0").decode("ascii")
        folder_range = (folder_start, folder_end)
        file_range = (file_start, file_end)
        return DriveDef(
            alias=alias,
            name=name,
            root_folder=root_folder,
            folder_range=folder_range,
            file_range=file_range,
        )

    def pack(self, stream: BinaryIO, value: DriveDef) -> int:
        alias: bytes = value.alias.encode("ascii")
        name: bytes = value.name.encode("ascii")
        args = (
            alias,
            name,
            value.folder_range[0],
            value.folder_range[1],
            value.file_range[0],
            value.file_range[1],
            value.root_folder,
        )
        packed: int = self.layout.pack_stream(stream, *args)
        return packed


@dataclass
class FolderDef:
    name_pos: int
    folder_range: Tuple[int, int]
    file_range: Tuple[int, int]


class FolderDefSerializer(StreamSerializer[FolderDef]):
    def __init__(self, layout: Struct):
        self.layout = layout

    def unpack(self, stream: BinaryIO) -> FolderDef:
        (
            name_pos,
            folder_start,
            folder_end,
            file_start,
            file_end,
        ) = self.layout.unpack_stream(stream)
        folder_range = (folder_start, folder_end)
        file_range = (file_start, file_end)
        return FolderDef(
            name_pos=name_pos, folder_range=folder_range, file_range=file_range
        )

    def pack(self, stream: BinaryIO, value: FolderDef) -> int:
        args = (
            value.name_pos,
            value.folder_range[0],
            value.folder_range[1],
            value.file_range[0],
            value.file_range[1],
        )
        packed: int = self.layout.pack_stream(stream, *args)
        return packed


@dataclass
class MetaBlock:
    name: str
    ptrs: ArchivePtrs


# TMetadata = TypeVar("TMetadata")
TMetaBlock = TypeVar("TMetaBlock", bound=MetaBlock)
TTocMetaBlock = TypeVar("TTocMetaBlock")


@dataclass
class FileDef:
    name_pos: int
    data_pos: int
    length_on_disk: int
    length_in_archive: int
    storage_type: StorageType


TFileDef = TypeVar("TFileDef", bound=FileDef)
AssembleFileMetaFunc = Callable[[TFileDef], Dict[str, object]]
DisassembleFileMetaFunc = Callable[[Dict[str, object]], TFileDef]
AssembleMetaFunc = Callable[
    [BinaryIO, TMetaBlock, Optional[TTocMetaBlock]], Dict[str, object]
]
DisassembleMetaFunc = Callable[
    [BinaryIO, Dict[str, object]], Tuple[TMetaBlock, TTocMetaBlock]
]


def _write_data(data: bytes, stream: BinaryIO) -> int:
    """
    Returns the index the data was written to.
    """
    pos = stream.tell()
    stream.write(data)
    return pos


def _get_or_write_name(name: str, stream: BinaryIO, lookup: Dict[str, int]) -> int:
    if name in lookup:
        return lookup[name]

    pos = lookup[name] = stream.tell()
    enc_name = name.encode("ascii") + b"\0"
    stream.write(enc_name)
    return pos


@dataclass
class TOCSerializationInfo(Generic[TFileDef]):
    drive: StreamSerializer[DriveDef]
    folder: StreamSerializer[FolderDef]
    file: StreamSerializer[TFileDef]
    name_toc_is_count: bool


ESSENCE_NAMESPACE = "essence"


class FSAssembler(Generic[TFileDef]):
    """
    A Helper class used to assemble the SGA hierarchy
    """

    def __init__(
        self,
        stream: BinaryIO,
        ptrs: ArchivePtrs,
        toc: TocBlock,
        toc_serialization_info: TOCSerializationInfo[TFileDef],
        build_file_meta: AssembleFileMetaFunc[TFileDef],
    ):
        self.stream: BinaryIO = stream
        self.ptrs: ArchivePtrs = ptrs
        self.toc: TocBlock = toc
        self.toc_serialization_info: TOCSerializationInfo[
            TFileDef
        ] = toc_serialization_info
        self.build_file_meta: AssembleFileMetaFunc[TFileDef] = build_file_meta
        self.names: Dict[int, str] = {}

    # decompress_files: bool = False
    # lazy: bool = False

    def read_toc_part(
        self,
        toc_info: Tuple[int, int],
        serializer: StreamSerializer[T],
    ) -> List[T]:
        self.stream.seek(self.ptrs.header_pos + toc_info[0])
        return [serializer.unpack(self.stream) for _ in range(toc_info[1])]

    def read_toc(
        self,
    ) -> Tuple[List[DriveDef], List[FolderDef], List[TFileDef], Dict[int, str]]:
        drives = self.read_toc_part(
            self.toc.drive_info, self.toc_serialization_info.drive
        )
        folders = self.read_toc_part(
            self.toc.folder_info, self.toc_serialization_info.folder
        )
        files = self.read_toc_part(self.toc.file_info, self.toc_serialization_info.file)
        names = (
            _read_toc_names_as_count(
                self.stream, self.toc.name_info, self.ptrs.header_pos
            )
            if self.toc_serialization_info.name_toc_is_count
            else _read_toc_names_as_size(
                self.stream, self.toc.name_info, self.ptrs.header_pos
            )
        )
        return drives, folders, files, names

    def assemble_file(self, parent_dir: FS, file_def: TFileDef) -> None:
        name = self.names[file_def.name_pos]

        metadata = self.build_file_meta(file_def)
        file_compressed = file_def.storage_type != StorageType.STORE
        lazy_info = FileLazyInfo(
            jump_to=self.ptrs.data_pos + file_def.data_pos,
            packed_size=file_def.length_in_archive,
            unpacked_size=file_def.length_on_disk,
            stream=self.stream,
            decompress=file_compressed,  # self.decompress_files,
        )

        data = lazy_info.read(file_compressed)  # self.decompress_files)

        essence_info: Dict[str, object] = {"storage_type": int(file_def.storage_type)}
        if metadata is not None:
            essence_info.update(metadata)

        with parent_dir.open(name, "wb") as file:
            file.write(data)

        info = {ESSENCE_NAMESPACE: essence_info}
        parent_dir.setinfo(name, info)

    def _assemble_container(
        self,
        container: FS,
        file_range: Tuple[int, int],
        folder_range: Tuple[int, int],
        files: List[TFileDef],
        folders: List[FolderDef],
        file_offset: int,
        folder_offset: int,
    ) -> None:
        offsetted_file_range = [
            file_range[0] - file_offset,
            file_range[1] - file_offset,
        ]
        offsetted_folder_range = [
            folder_range[0] - folder_offset,
            folder_range[1] - folder_offset,
        ]

        container_files = files[offsetted_file_range[0] : offsetted_file_range[1]]
        container_folders = folders[
            offsetted_folder_range[0] : offsetted_folder_range[1]
        ]

        for file_def in container_files:
            self.assemble_file(container, file_def)

        for folder_def in container_folders:
            self.assemble_folder(
                container, folder_def, files, folders, file_offset, folder_offset
            )

    def assemble_folder(
        self,
        parent_dir: FS,
        folder_def: FolderDef,
        files: List[TFileDef],
        folders: List[FolderDef],
        file_offset: int,
        folder_offset: int,
    ) -> FS:
        raw_folder_name = self.names[folder_def.name_pos]
        folder_name_as_path = PurePath(raw_folder_name)
        folder_name = (
            folder_name_as_path.parts[-1]
            if len(folder_name_as_path.parts) > 0
            else raw_folder_name
        )

        folder = parent_dir.makedir(folder_name)
        self._assemble_container(
            folder,
            folder_def.file_range,
            folder_def.folder_range,
            files,
            folders,
            file_offset,
            folder_offset,
        )
        return folder

    def assemble_drive(
        self,
        essence_fs: EssenceFS,
        drive_def: DriveDef,
        folder_defs: List[FolderDef],
        file_defs: List[TFileDef],
    ) -> FS:
        local_file_defs = file_defs[drive_def.file_range[0] : drive_def.file_range[1]]
        local_folder_defs = folder_defs[
            drive_def.folder_range[0] : drive_def.folder_range[1]
        ]

        file_offset = drive_def.file_range[0]
        folder_offset = drive_def.folder_range[0]

        # make root folder relative to our folder slice
        drive_folder_index = drive_def.root_folder - folder_offset
        drive_folder_def = local_folder_defs[drive_folder_index]

        drive = essence_fs.create_drive(drive_def.alias)
        self._assemble_container(
            drive,
            drive_folder_def.file_range,
            drive_folder_def.folder_range,
            local_file_defs,
            local_folder_defs,
            file_offset,
            folder_offset,
        )
        return drive

    def assemble(self, fs: EssenceFS) -> None:
        drive_defs, folder_defs, file_defs, names = self.read_toc()
        self.names.update(names)
        for drive_def in drive_defs:
            self.assemble_drive(fs, drive_def, folder_defs, file_defs)


class FSDisassembler(Generic[TFileDef]):
    def __init__(
        self,
        fs: EssenceFS,
        toc_stream: BinaryIO,
        data_stream: BinaryIO,
        name_stream: BinaryIO,
        toc_serialization_info: TOCSerializationInfo[TFileDef],
        meta2def: DisassembleFileMetaFunc[TFileDef],
    ):
        self.fs = fs
        """A stream containing the TOC Block"""
        self.toc_stream = toc_stream
        """A stream containing the DATA Block"""
        self.data_stream = data_stream
        """A stream containing the NAME Block"""
        self.name_stream = name_stream
        """A collection containing serializers for DriveDef, FolderDef, FileDef, and a flag to determine whether the NAME Block uses 'size in bytes ~ SIZE' or 'number of elements ~ COUNT'"""
        self.toc_serialization_info = toc_serialization_info
        """A function which converts FileMetadata to a FileDef"""
        self.meta2def = meta2def
        """A collection of file definitions laid out sequentially (by folder). This is populated and used inside the assembler."""
        self.flat_files: List[TFileDef] = []
        """A collection of folder definitions laid out sequentially (by drive/parent folder). This is populated and used inside the assembler."""
        self.flat_folders: List[FolderDef] = []
        """A collection of drive definitions), ordered arbitrarily. This is populated and used inside the assembler."""
        self.flat_drives: List[DriveDef] = []
        """A lookup table to find names already written to the NAME block; contains the position of the desired name in the NAME block."""
        self.flat_names: Dict[str, int] = {}

    def disassemble_file(self, container_fs: FS, file_name: str) -> TFileDef:
        with container_fs.open(file_name, "rb") as handle:
            data = handle.read()

        metadata = dict(container_fs.getinfo(file_name, ["essence"]).raw["essence"])

        file_def: TFileDef = self.meta2def(metadata)
        _storage_type_value: int = metadata["storage_type"]  # type: ignore
        storage_type = StorageType(_storage_type_value)
        if storage_type == StorageType.STORE:
            store_data = data
        elif storage_type in [
            StorageType.BUFFER_COMPRESS,
            StorageType.STREAM_COMPRESS,
        ]:
            store_data = zlib.compress(data)  # TODO process in chunks for large files
        else:
            raise NotImplementedError

        file_def.storage_type = storage_type
        file_def.length_on_disk = len(data)
        file_def.length_in_archive = len(store_data)

        file_def.name_pos = _get_or_write_name(
            file_name, self.name_stream, self.flat_names
        )
        file_def.data_pos = _write_data(store_data, self.data_stream)

        return file_def

    def flatten_file_collection(self, container_fs: FS) -> Tuple[int, int]:
        subfile_start = len(self.flat_files)
        subfile_defs = [
            self.disassemble_file(container_fs, file_info.name)
            for file_info in container_fs.scandir("/")
            if not file_info.is_dir
        ]
        self.flat_files.extend(subfile_defs)
        subfile_end = len(self.flat_files)
        return subfile_start, subfile_end

    def flatten_folder_collection(self, container_fs: FS, path: str) -> Tuple[int, int]:
        # Create temporary None folders to ensure a continuous range of child folders; BEFORE entering any child folders
        subfolder_start = len(self.flat_folders)
        folders = [
            file_info.name
            for file_info in container_fs.scandir("/")
            if file_info.is_dir
        ]
        self.flat_folders.extend([None] * len(folders))  # type:ignore
        subfolder_end = len(self.flat_folders)

        # Enter subfolders, and add them to the flat array
        subfolder_defs = [
            self.disassemble_folder(container_fs.opendir(folder), f"{path}/{folder}")
            for folder in folders
        ]
        self.flat_folders[subfolder_start:subfolder_end] = subfolder_defs
        return subfolder_start, subfolder_end

    def disassemble_folder(self, folder_fs: FS, path: str) -> FolderDef:
        folder_def = FolderDef(None, None, None)  # type: ignore

        # Subfiles
        subfile_range = self.flatten_file_collection(folder_fs)
        # Subfolders
        # # Since Relic typically uses the first folder as the root folder; I will try to preserve that parent folders come before their child folders
        subfolder_range = self.flatten_folder_collection(folder_fs, path)

        folder_name = str(path).split(":", 1)[-1]  # Strip 'alias:' from path

        if folder_name[0] == "/":
            folder_name = folder_name[1:]  # strip leading '/'

        folder_def.name_pos = _get_or_write_name(
            folder_name, self.name_stream, self.flat_names
        )
        folder_def.file_range = subfile_range
        folder_def.folder_range = subfolder_range

        return folder_def

    def disassemble_drive(self, drive: _EssenceDriveFS, alias: str) -> DriveDef:
        name = ""
        drive_folder_def = FolderDef(None, None, None)  # type: ignore
        root_folder = len(self.flat_folders)
        folder_start = len(self.flat_folders)
        file_start = len(self.flat_files)
        self.flat_folders.append(drive_folder_def)

        drive_folder_def.name_pos = _get_or_write_name(
            name, self.name_stream, self.flat_names
        )
        drive_folder_def.file_range = self.flatten_file_collection(drive)
        drive_folder_def.folder_range = self.flatten_folder_collection(drive, name)

        folder_end = len(self.flat_folders)
        file_end = len(self.flat_files)

        drive_def = DriveDef(
            alias,
            name,
            root_folder,
            folder_range=(folder_start, folder_end),
            file_range=(file_start, file_end),
        )
        return drive_def

    def write_toc(self) -> TocBlock:
        """
        Writes TOC data to the stream.

        The TocHeader returned is relative to the toc stream's start, does not include the TocHeader itself.
        """
        # Normally, this is drive -> folder -> file -> names
        #   But the TOC can handle an arbitrary order (due to ptrs); so we only do this to match their style
        drive_offset = self.toc_stream.tell()
        for drive_def in self.flat_drives:
            self.toc_serialization_info.drive.pack(self.toc_stream, drive_def)

        folder_offset = self.toc_stream.tell()
        for folder_def in self.flat_folders:
            self.toc_serialization_info.folder.pack(self.toc_stream, folder_def)

        file_offset = self.toc_stream.tell()
        for file_def in self.flat_files:
            self.toc_serialization_info.file.pack(self.toc_stream, file_def)

        name_offset = self.toc_stream.tell()
        name_size = self.name_stream.tell()
        self.name_stream.seek(0)
        _chunked_copy(self.name_stream, self.toc_stream, chunk_size=64 * KiB)
        return TocBlock(
            drive_info=(drive_offset, len(self.flat_drives)),
            folder_info=(folder_offset, len(self.flat_folders)),
            file_info=(file_offset, len(self.flat_files)),
            name_info=(
                name_offset,
                len(self.flat_names)
                if self.toc_serialization_info.name_toc_is_count
                else name_size,
            ),
        )

    def disassemble(self) -> TocBlock:
        for name, drive_fs in self.fs.iterate_fs():
            drive_fs = typing.cast(_EssenceDriveFS, drive_fs)
            drive_def = self.disassemble_drive(drive_fs, name)
            self.flat_drives.append(drive_def)

        return self.write_toc()


def _read_toc_names_as_count(
    stream: BinaryIO, toc_info: Tuple[int, int], header_pos: int, buffer_size: int = 256
) -> Dict[int, str]:
    NULL = 0
    NULL_CHAR = b"\0"
    stream.seek(header_pos + toc_info[0])

    names: Dict[int, str] = {}
    running_buffer = bytearray()
    offset = 0
    while len(names) < toc_info[1]:
        buffer = stream.read(buffer_size)
        if len(buffer) == 0:
            raise Exception("Ran out of data!")  # TODO, proper exception
        terminal_null = buffer[-1] == NULL
        parts = buffer.split(NULL_CHAR)
        if len(parts) > 1:
            parts[0] = running_buffer + parts[0]
            running_buffer.clear()
            if not terminal_null:
                running_buffer.extend(parts[-1])
            parts = parts[:-1]  # drop empty or partial

        else:
            if not terminal_null:
                running_buffer.extend(parts[0])
                offset += len(buffer)
                continue

        remaining = toc_info[1] - len(names)
        available = min(len(parts), remaining)
        for _ in range(available):
            name = parts[_]
            names[offset] = name.decode("ascii")
            offset += len(name) + 1
    return names


def _read_toc_names_as_size(
    stream: BinaryIO, toc_info: Tuple[int, int], header_pos: int
) -> Dict[int, str]:
    stream.seek(header_pos + toc_info[0])
    name_buffer = stream.read(toc_info[1])
    parts = name_buffer.split(b"\0")
    names: Dict[int, str] = {}
    offset = 0
    for part in parts:
        names[offset] = part.decode("ascii")
        offset += len(part) + 1
    return names


def _chunked_read(
    stream: BinaryIO, size: Optional[int] = None, chunk_size: Optional[int] = None
) -> Iterable[bytes]:
    if size is None and chunk_size is None:
        yield stream.read()
    elif size is None and chunk_size is not None:
        while True:
            buffer = stream.read(chunk_size)
            yield buffer
            if len(buffer) != chunk_size:
                break
    elif size is not None and chunk_size is None:
        yield stream.read(size)
    elif size is not None and chunk_size is not None:
        chunks = size // chunk_size
        for _ in range(chunks):
            yield stream.read(chunk_size)
        total_read = chunk_size * chunks
        if total_read < size:
            yield stream.read(size - total_read)
    else:
        raise Exception("Something impossible happened!")


def _chunked_copy(
    in_stream: BinaryIO,
    out_stream: BinaryIO,
    size: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> None:
    for chunk in _chunked_read(in_stream, size, chunk_size):
        out_stream.write(chunk)


@dataclass
class Md5ChecksumHelper:
    expected: Optional[bytes]
    stream: Optional[BinaryIO]
    start: int
    size: Optional[int] = None
    eigen: Optional[bytes] = None

    def read(self, stream: Optional[BinaryIO] = None) -> bytes:
        stream = self.stream if stream is None else stream
        if stream is None:
            raise IOError("No Stream Provided!")
        stream.seek(self.start)
        md5 = hashlib.md5(self.eigen) if self.eigen is not None else hashlib.md5()
        # Safer for large files to read chunked
        for chunk in _chunked_read(stream, self.size, 256 * KiB):
            md5.update(chunk)
        md5_str = md5.hexdigest()
        return bytes.fromhex(md5_str)

    def validate(self, stream: Optional[BinaryIO] = None) -> None:
        result = self.read(stream)
        if self.expected != result:
            raise MD5MismatchError(result, self.expected)


def _fix_toc(toc: TocBlock, cur_toc_start: int, desired_toc_start: int) -> None:
    def _fix(info: Tuple[int, int]) -> Tuple[int, int]:
        return info[0] + (cur_toc_start - desired_toc_start), info[1]

    toc.folder_info = _fix(toc.folder_info)
    toc.file_info = _fix(toc.file_info)
    toc.drive_info = _fix(toc.drive_info)
    toc.name_info = _fix(toc.name_info)


class EssenceFSSerializer(
    EssenceFSHandler, Generic[TFileDef, TMetaBlock, TTocMetaBlock]
):
    # Would use a dataclass; but I also want to be able to override defaults in parent dataclasses
    def __init__(
        self,
        version: Version,
        meta_serializer: StreamSerializer[TMetaBlock],
        toc_serializer: StreamSerializer[TocBlock],
        toc_meta_serializer: Optional[StreamSerializer[TTocMetaBlock]],
        toc_serialization_info: TOCSerializationInfo[TFileDef],
        assemble_meta: AssembleMetaFunc[TMetaBlock, TTocMetaBlock],
        disassemble_meta: DisassembleMetaFunc[TMetaBlock, TTocMetaBlock],
        build_file_meta: AssembleFileMetaFunc[TFileDef],
        gen_empty_meta: Callable[[], TMetaBlock],
        finalize_meta: Callable[[BinaryIO, TMetaBlock], None],
        meta2def: Callable[[Dict[str, object]], TFileDef],
    ):
        self.version = version
        self.meta_serializer = meta_serializer
        self.toc_serializer = toc_serializer
        self.toc_meta_serializer = toc_meta_serializer
        self.toc_serialization_info = toc_serialization_info
        self.assemble_meta = assemble_meta
        self.disassemble_meta = disassemble_meta
        self.build_file_meta = build_file_meta
        self.gen_empty_meta = gen_empty_meta
        self.finalize_meta = finalize_meta
        self.meta2def = meta2def

    def read(self, stream: BinaryIO) -> EssenceFS:
        # Magic & Version; skippable so that we can check for a valid file and read the version elsewhere
        _validate_magic_word(MagicWord, stream, advance=True)
        stream_version = Version.unpack(stream)
        if stream_version != self.version:
            raise VersionMismatchError(stream_version, self.version)

        meta_block = self.meta_serializer.unpack(stream)
        stream.seek(meta_block.ptrs.header_pos)
        toc_block = self.toc_serializer.unpack(stream)
        # Additional TOC information is not present in earlier versions
        toc_meta_block = (
            self.toc_meta_serializer.unpack(stream)
            if self.toc_meta_serializer is not None
            else None
        )

        name, metadata = meta_block.name, self.assemble_meta(
            stream, meta_block, toc_meta_block
        )
        assembler: FSAssembler[TFileDef] = FSAssembler(
            stream=stream,
            ptrs=meta_block.ptrs,
            toc=toc_block,
            toc_serialization_info=self.toc_serialization_info,
            # decompress_files=decompress,
            build_file_meta=self.build_file_meta,
            # lazy=lazy,
        )
        essence_fs = EssenceFS()
        assembler.assemble(essence_fs)
        essence_info: Dict[str, object] = {
            "name": name,
            "version": {"major": stream_version.major, "minor": stream_version.minor},
        }
        if metadata is not None:
            essence_info.update(metadata)

        essence_fs.setmeta(essence_info, ESSENCE_NAMESPACE)
        return essence_fs

    def write(self, stream: BinaryIO, essence_fs: EssenceFS) -> int:
        archive_metadata: Dict[str, object] = typing.cast(
            Dict[str, object], essence_fs.getmeta("essence")
        )
        archive_name: str = typing.cast(str, archive_metadata["name"])
        # IDK why I write to a temp stream; maybe to preserve dest stream in case of errors?
        with BytesIO() as temp_stream:
            MagicWord.write_magic_word(temp_stream)
            self.version.pack(temp_stream)
            with BytesIO() as data_stream:
                with BytesIO() as toc_stream:
                    with BytesIO() as name_stream:
                        disassembler = FSDisassembler(
                            fs=essence_fs,
                            toc_stream=toc_stream,
                            data_stream=data_stream,
                            name_stream=name_stream,
                            toc_serialization_info=self.toc_serialization_info,
                            meta2def=self.meta2def,
                        )

                        partial_toc = disassembler.disassemble()

                        partial_meta, toc_meta = self.disassemble_meta(
                            temp_stream, archive_metadata
                        )
                        # we need to come back with the correct data
                        meta_writeback = temp_stream.tell()
                        empty_meta = self.gen_empty_meta()
                        self.meta_serializer.pack(temp_stream, empty_meta)

                        # the start of the toc stream in the current stream
                        toc_start = temp_stream.tell()
                        toc_writeback = toc_start
                        self.toc_serializer.pack(temp_stream, TocBlock.default())

                        if self.toc_meta_serializer:
                            self.toc_meta_serializer.pack(temp_stream, toc_meta)

                        toc_rel_start = temp_stream.tell()
                        toc_stream.seek(0)
                        _chunked_copy(toc_stream, temp_stream, chunk_size=64 * KiB)
                        toc_end = temp_stream.tell()  # The end of the TOC block;
                        toc_size = toc_end - toc_start

                        data_start = temp_stream.tell()
                        data_stream.seek(0)
                        _chunked_copy(data_stream, temp_stream, chunk_size=1 * MiB)
                        data_size = data_stream.tell()

                        partial_meta.name = archive_name
                        partial_meta.ptrs = ArchivePtrs(
                            toc_start, toc_size, data_start, data_size
                        )
                        _fix_toc(partial_toc, toc_rel_start, toc_start)

                        temp_stream.seek(toc_writeback)
                        self.toc_serializer.pack(temp_stream, partial_toc)

                        if self.finalize_meta is not None:
                            self.finalize_meta(temp_stream, partial_meta)

                        temp_stream.seek(meta_writeback)
                        self.meta_serializer.pack(temp_stream, partial_meta)

            temp_stream.seek(0)
            _chunked_copy(temp_stream, stream, chunk_size=16 * MiB)
            return temp_stream.tell()


#   Archives have 7 blocks:
#       MagicBlock
#           Contains "_ARCHIVE" (8 byte long ASCII string)
#           Contains Version (UINT16, UINT16 tuple)
#       MetaBlock
#           Several Metadata sections
#           PTR Block
#           TOC Block
#       FileBlock
#       FolderBlock
#       DriveBlock
#       NameBlock
#       DataBlock


@dataclass
class FileLazyInfo:
    jump_to: int
    packed_size: int
    unpacked_size: int
    stream: BinaryIO
    decompress: bool

    def read(self, decompress: Optional[bool] = None) -> bytes:
        decompress = self.decompress if decompress is None else decompress
        jump_back = self.stream.tell()
        self.stream.seek(self.jump_to)
        in_buffer = self.stream.read(self.packed_size)
        if decompress and self.packed_size != self.unpacked_size:
            out_buffer = zlib.decompress(in_buffer)
            if len(out_buffer) != self.unpacked_size:
                raise DecompressedSizeMismatch(len(out_buffer), self.unpacked_size)
        else:
            out_buffer = in_buffer
        self.stream.seek(jump_back)
        return out_buffer


@dataclass
class ArchivePtrs:
    """
    Contains 'pointers' to the TOC Block (header_pos, header_size) and the DATA Block (data_pos, data_size)
    """

    header_pos: int
    header_size: int
    data_pos: int
    data_size: Optional[int] = None

    @classmethod
    def default(cls) -> ArchivePtrs:
        """
        Creates a 'Default' Archive Ptrs Object; used to create a valid placeholder until proper data is supplied.
        """
        return cls(0, 0, 0, 0)


__all__ = [
    "TocBlock",
    "TocHeaderSerializer",
    "DriveDef",
    "DriveDefSerializer",
    "FolderDef",
    "FolderDefSerializer",
    "MetaBlock",
    "TMetaBlock",
    "TTocMetaBlock",
    "FileDef",
    "TFileDef",
    "AssembleFileMetaFunc",
    "DisassembleFileMetaFunc",
    "AssembleMetaFunc",
    "DisassembleMetaFunc",
    "TOCSerializationInfo",
    "FSAssembler",
    "FSDisassembler",
    "Md5ChecksumHelper",
    "EssenceFSSerializer",
    "FileLazyInfo",
    "ArchivePtrs",
]
