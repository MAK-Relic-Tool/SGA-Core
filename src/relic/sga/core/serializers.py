from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePath
from typing import (
    BinaryIO,
    List,
    Dict,
    Optional,
    Callable,
    Tuple,
    Iterable,
    TypeVar,
    Union,
    Any,
    Generic,
)

from serialization_tools.size import KiB, MiB
from serialization_tools.structx import Struct

from relic.sga.core import abstract, protocols
from relic.sga.core.abstract import (
    DriveDef,
    FolderDef,
    FileLazyInfo,
    TFileMeta,
    TocBlock,
    ArchivePtrs,
    File,
    Folder,
    Drive,
    FileDef,
    Archive,
)
from relic.sga.core.definitions import StorageType, Version, MagicWord
from relic.sga.core.errors import MD5MismatchError, VersionMismatchError
from relic.sga.core.protocols import StreamSerializer, T


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


TMetadata = TypeVar("TMetadata")
TMetaBlock = TypeVar("TMetaBlock", bound=MetaBlock)
TTocMetaBlock = TypeVar("TTocMetaBlock")
TFileDef = TypeVar("TFileDef", bound=FileDef)
AssembleFileMetaFunc = Callable[[TFileDef], TFileMeta]
DisassembleFileMetaFunc = Callable[[TFileMeta], TFileDef]
AssembleMetaFunc = Callable[[BinaryIO, TMetaBlock, Optional[TTocMetaBlock]], TMetadata]
DisassembleMetaFunc = Callable[[BinaryIO, TMetadata], Tuple[TMetaBlock, TTocMetaBlock]]


def write_data(data: bytes, stream: BinaryIO) -> int:
    """
    Returns the index the data was written to.
    """
    pos = stream.tell()
    stream.write(data)
    return pos


def get_or_write_name(name: str, stream: BinaryIO, lookup: Dict[str, int]) -> int:
    if name in lookup:
        return lookup[name]
    else:
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


@dataclass
class IOAssembler(Generic[TFileDef, TFileMeta]):
    """
    A Helper class used to assemble the SGA hierarchy
    """

    stream: BinaryIO
    ptrs: ArchivePtrs
    toc: TocBlock
    toc_serialization_info: TOCSerializationInfo[TFileDef]

    build_file_meta: AssembleFileMetaFunc[TFileDef, TFileMeta]
    decompress_files: bool = False
    lazy: bool = False

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
            read_toc_names_as_count(
                self.stream, self.toc.name_info, self.ptrs.header_pos
            )
            if self.toc_serialization_info.name_toc_is_count
            else _read_toc_names_as_size(
                self.stream, self.toc.name_info, self.ptrs.header_pos
            )
        )
        return drives, folders, files, names

    def assemble_file(
        self, file_def: TFileDef, names: Dict[int, str]
    ) -> File[TFileMeta]:
        name = names[file_def.name_pos]
        metadata: TFileMeta = self.build_file_meta(file_def)
        lazy_info = FileLazyInfo(
            jump_to=self.ptrs.data_pos + file_def.data_pos,
            packed_size=file_def.length_in_archive,
            unpacked_size=file_def.length_on_disk,
            stream=self.stream,
            decompress=self.decompress_files,
        )
        file_compressed = file_def.storage_type != StorageType.STORE
        file = File(
            name=name,
            _data=None,
            storage_type=file_def.storage_type,
            _is_compressed=file_compressed,
            metadata=metadata,
            _lazy_info=lazy_info,
        )
        if not self.lazy:
            load_lazy_data(file)
        return file

    def assemble_folder(
        self,
        folder_def: FolderDef,
        files: List[File[TFileMeta]],
        file_offset: int,
        names: Dict[int, str],
    ) -> Folder[TFileMeta]:
        raw_folder_name = names[folder_def.name_pos]
        folder_name_as_path = PurePath(raw_folder_name)
        folder_name = (
            folder_name_as_path.parts[-1]
            if len(folder_name_as_path.parts) > 0
            else raw_folder_name
        )
        subfile_start = folder_def.file_range[0] - file_offset
        subfile_end = folder_def.file_range[1] - file_offset
        sub_files = files[subfile_start:subfile_end]
        folder = Folder(folder_name, [], sub_files, None)
        return folder
        # folders.append(folder)

    def assemble_subfolder(
        self,
        folder_defs: List[FolderDef],
        folders: List[Folder[TFileMeta]],
        folder_offset: int,
    ) -> None:
        for folder_def, folder in zip(folder_defs, folders):
            subfolder_start = folder_def.folder_range[0] - folder_offset
            subfolder_end = folder_def.folder_range[1] - folder_offset
            folder.sub_folders = folders[subfolder_start:subfolder_end]

        for folder in folders:
            _apply_self_as_parent(folder)

    def assemble_drive(
        self,
        drive_def: DriveDef,
        folder_defs: List[FolderDef],
        file_defs: List[TFileDef],
        names: Dict[int, str],
    ) -> Drive[TFileMeta]:
        local_file_defs = file_defs[drive_def.file_range[0] : drive_def.file_range[1]]
        local_files = [self.assemble_file(f_def, names) for f_def in local_file_defs]

        local_folder_defs = folder_defs[
            drive_def.folder_range[0] : drive_def.folder_range[1]
        ]
        local_folders = [
            self.assemble_folder(
                folder_def, local_files, drive_def.file_range[0], names
            )
            for folder_def in local_folder_defs
        ]
        # make root folder relative to our folder slice
        root_folder = drive_def.root_folder - drive_def.folder_range[0]
        drive_folder = local_folders[root_folder]
        drive = Drive(
            drive_def.alias,
            drive_def.name,
            drive_folder.sub_folders,
            drive_folder.files,
        )
        _apply_self_as_parent(drive)
        return drive

    def assemble(self) -> List[Drive[TFileMeta]]:
        drive_defs, folder_defs, file_defs, names = self.read_toc()
        drives: List[Drive[TFileMeta]] = [
            self.assemble_drive(drive_def, folder_defs, file_defs, names)
            for drive_def in drive_defs
        ]
        return drives


@dataclass
class IODisassembler(Generic[TFileMeta, TFileDef]):
    def __init__(
        self,
        drives: List[Drive[TFileMeta]],
        toc_stream: BinaryIO,
        data_stream: BinaryIO,
        name_stream: BinaryIO,
        toc_serialization_info: TOCSerializationInfo[TFileDef],
        meta2def: DisassembleFileMetaFunc[TFileMeta, TFileDef],
    ):
        self.drives = drives
        self.toc_stream = toc_stream
        self.data_stream = data_stream
        self.name_stream = name_stream
        self.toc_serialization_info = toc_serialization_info
        self.meta2def = meta2def
        self.flat_files: List[TFileDef] = []
        self.flat_folders: List[FolderDef] = []
        self.flat_drives: List[DriveDef] = []
        self.flat_names: Dict[str, int] = {}

    def disassemble_file(self, file: File[TFileMeta]) -> TFileDef:
        file_def: TFileDef = self.meta2def(file.metadata)
        data = file.data
        if file.storage_type == StorageType.STORE:
            store_data = data
        elif file.storage_type in [
            StorageType.BUFFER_COMPRESS,
            StorageType.STREAM_COMPRESS,
        ]:
            store_data = zlib.compress(data)  # TODO process in chunks for large files
        else:
            raise NotImplementedError

        file_def.storage_type = file.storage_type
        file_def.length_on_disk = len(data)
        file_def.length_in_archive = len(store_data)

        file_def.name_pos = get_or_write_name(
            file.name, self.name_stream, self.flat_names
        )
        file_def.data_pos = write_data(store_data, self.data_stream)

        return file_def

    def flatten_file_collection(self, files: List[File[TFileMeta]]) -> Tuple[int, int]:
        subfile_start = len(self.flat_files)
        subfile_defs = [self.disassemble_file(file) for file in files]
        self.flat_files.extend(subfile_defs)
        subfile_end = len(self.flat_files)
        return subfile_start, subfile_end

    def flatten_folder_collection(
        self, folders: List[Folder[TFileMeta]]
    ) -> Tuple[int, int]:
        # Create temporary None folders to ensure a continuous range of child folders; BEFORE entering any child folders
        subfolder_start = len(self.flat_folders)
        self.flat_folders.extend([None] * len(folders))  # type:ignore
        subfolder_end = len(self.flat_folders)

        # Enter subfolders, and add them to the flat array
        subfolder_defs = [self.disassemble_folder(folder) for folder in folders]
        self.flat_folders[subfolder_start:subfolder_end] = subfolder_defs
        return subfolder_start, subfolder_end

    def disassemble_folder(self, folder: Folder[TFileMeta]) -> FolderDef:
        # Subfiles
        subfile_range = self.flatten_file_collection(folder.files)

        # Subfolders
        # # Since Relic typically uses the first folder as the root folder; I will try to preserve that parent folders come before their child folders
        folder_def = FolderDef(None, None, None)  # type: ignore

        subfolder_range = self.flatten_folder_collection(folder.sub_folders)

        folder_name = str(folder.path).split(":")[-1]  # Strip 'alias:' from path

        folder_def.name_pos = get_or_write_name(
            folder_name, self.name_stream, self.flat_names
        )
        folder_def.file_range = subfile_range
        folder_def.folder_range = subfolder_range

        return folder_def

    def disassemble_drive(self, drive: Drive[TFileMeta]) -> DriveDef:
        drive_folder_def = FolderDef(None, None, None)  # type: ignore
        root_folder = len(self.flat_folders)
        folder_start = len(self.flat_folders)
        file_start = len(self.flat_files)
        self.flat_folders.append(drive_folder_def)

        drive_folder_def.name_pos = get_or_write_name(
            "", self.name_stream, self.flat_names
        )
        drive_folder_def.file_range = self.flatten_file_collection(drive.files)
        drive_folder_def.folder_range = self.flatten_folder_collection(
            drive.sub_folders
        )

        folder_end = len(self.flat_folders)
        file_end = len(self.flat_files)

        drive_def = DriveDef(
            drive.alias,
            drive.name,
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
        for drive in self.drives:
            self.disassemble_drive(drive)

        return self.write_toc()


def _apply_self_as_parent(
    collection: Union[Folder[TFileMeta], Drive[TFileMeta]]
) -> None:
    for folder in collection.sub_folders:
        folder.parent = collection
    for file in collection.files:
        file.parent = collection


def read_toc_names_as_count(
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
    elif size is not None and chunk_size is not None:  # MyPy
        chunks = size // chunk_size
        for _ in range(chunks):
            yield stream.read(chunk_size)
        total_read = chunk_size * chunks
        if total_read < size:
            yield stream.read(size - total_read)
    else:
        raise Exception("Something impossible happened!")


def _chunked_copy(
    input: BinaryIO,
    output: BinaryIO,
    size: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> None:
    for chunk in _chunked_read(input, size, chunk_size):
        output.write(chunk)


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


def load_lazy_data(file: File[TFileMeta]) -> None:
    lazy_info: Optional[FileLazyInfo] = file._lazy_info
    if lazy_info is None:
        raise Exception("API read files, but failed to create lazy info!")
    file.data = lazy_info.read()  # decompress should use cached value
    file._lazy_info = None


def _fix_toc(toc: TocBlock, cur_toc_start: int, desired_toc_start: int) -> None:
    def _fix(info: Tuple[int, int]) -> Tuple[int, int]:
        return info[0] + (cur_toc_start - desired_toc_start), info[1]

    toc.folder_info = _fix(toc.folder_info)
    toc.file_info = _fix(toc.file_info)
    toc.drive_info = _fix(toc.drive_info)
    toc.name_info = _fix(toc.name_info)


class ArchiveSerializer(
    protocols.ArchiveSerializer[Archive[TMetadata, TFileMeta]],
    Generic[TMetadata, TFileMeta, TFileDef, TMetaBlock, TTocMetaBlock],
):
    # Would use a dataclass; but I also want to be able to override defaults in parent dataclasses
    def __init__(
        self,
        version: Version,
        meta_serializer: StreamSerializer[TMetaBlock],
        toc_serializer: StreamSerializer[TocBlock],
        toc_meta_serializer: Optional[StreamSerializer[TTocMetaBlock]],
        toc_serialization_info: TOCSerializationInfo[TFileDef],
        assemble_meta: AssembleMetaFunc[TMetaBlock, TTocMetaBlock, TMetadata],
        disassemble_meta: DisassembleMetaFunc[TMetadata, TMetaBlock, TTocMetaBlock],
        build_file_meta: AssembleFileMetaFunc[TFileDef, TFileMeta],
        gen_empty_meta: Callable[[], TMetaBlock],
        finalize_meta: Callable[[BinaryIO, TMetaBlock], None],
        meta2def: Callable[[TFileMeta], TFileDef],
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

    def read(
        self,
        stream: BinaryIO,
        lazy: bool = False,
        decompress: bool = True,
        skip_magic_and_version: bool = False,
    ) -> Archive[TMetadata, TFileMeta]:
        # Magic & Version; skippable so that we can check for a valid file and read the version elsewhere
        if not skip_magic_and_version:
            if not MagicWord.check_magic_word(stream, advance=True):
                raise NotImplementedError
            stream_version = Version.unpack(stream)
            if stream_version != self.version:
                raise VersionMismatchError(stream_version, self.version)

        meta_block = self.meta_serializer.unpack(stream)
        stream.seek(meta_block.ptrs.header_pos)
        toc_block = self.toc_serializer.unpack(stream)
        toc_meta_block = (
            self.toc_meta_serializer.unpack(stream)
            if self.toc_meta_serializer is not None
            else None
        )

        name, metadata = meta_block.name, self.assemble_meta(
            stream, meta_block, toc_meta_block
        )
        assembler = IOAssembler(
            stream=stream,
            ptrs=meta_block.ptrs,
            toc=toc_block,
            toc_serialization_info=self.toc_serialization_info,
            decompress_files=decompress,
            build_file_meta=self.build_file_meta,
            lazy=lazy,
        )
        drives = assembler.assemble()

        return Archive(name, metadata, drives)

    def write(self, stream: BinaryIO, archive: Archive[TMetadata, TFileMeta]) -> int:
        with BytesIO() as temp_stream:
            MagicWord.write_magic_word(temp_stream)
            self.version.pack(temp_stream)
            with BytesIO() as data_stream:
                with BytesIO() as toc_stream:
                    with BytesIO() as name_stream:
                        disassembler = IODisassembler(
                            drives=archive.drives,
                            toc_stream=toc_stream,
                            data_stream=data_stream,
                            name_stream=name_stream,
                            toc_serialization_info=self.toc_serialization_info,
                            meta2def=self.meta2def,
                        )

                        partial_toc = disassembler.disassemble()

                        partial_meta, toc_meta = self.disassemble_meta(
                            temp_stream, archive.metadata
                        )

                        meta_writeback = (
                            temp_stream.tell()
                        )  # we need to come back with the correct data
                        empty_meta = self.gen_empty_meta()
                        self.meta_serializer.pack(temp_stream, empty_meta)

                        toc_start = (
                            temp_stream.tell()
                        )  # the start of the toc stream in the current stream
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

                        partial_meta.name = archive.name
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


#   Archives have 6 blocks:
#       MetaBlock
#           Several Metadata sections
#           PTR Block
#           TOC Block
#       FileBlock
#       FolderBlock
#       DriveBlock
#       NameBlock
#       DataBlock
