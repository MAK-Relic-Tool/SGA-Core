from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import (
    Optional,
    Tuple,
    BinaryIO,
    TypeVar,
)

from relic.sga.core.definitions import StorageType
from relic.sga.core.errors import DecompressedSizeMismatch


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
class DriveDef:
    alias: str
    name: str
    root_folder: int
    folder_range: Tuple[int, int]
    file_range: Tuple[int, int]


@dataclass
class FolderDef:
    name_pos: int
    folder_range: Tuple[int, int]
    file_range: Tuple[int, int]


@dataclass
class FileDef:
    name_pos: int
    data_pos: int
    length_on_disk: int
    length_in_archive: int
    storage_type: StorageType


TMeta = TypeVar("TMeta")
TFileMeta = TypeVar("TFileMeta")


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


@dataclass
class ArchivePtrs:
    header_pos: int
    header_size: int
    data_pos: int
    data_size: Optional[int] = None

    @classmethod
    def default(cls) -> ArchivePtrs:
        return cls(0, 0, 0, 0)
