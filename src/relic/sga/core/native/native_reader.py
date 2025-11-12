"""Native SGA Binary Parser - Direct Binary Format Implementation

Parses SGA V2 binary format directly to extract byte offsets.
Uses mmap + parallel zlib for ultra-fast extraction (3-4 seconds for 7,815 files).

Completely bypasses fs library for maximum speed!
"""

from __future__ import annotations

import zlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import List, TypeVar

from relic.core.errors import MismatchError, RelicToolError

from relic.sga.core.definitions import StorageType
from relic.sga.core.native.definitions import FileEntry, ReadResult, ReadonlyMemMapFile

_TIn = TypeVar("_TIn")
_TOut = TypeVar("_TOut")


class SgaReader(ReadonlyMemMapFile):
    """
    Reads an SGA file using FileEntry objects.

    FileEntry objects contain ABSOLUTE byte offsets within the SGA file
    as well as the file's size (compressed_size)
    and the file's storage_type

    These three keys allow us to quickly read raw sga files and decompress them if needed
    """

    def read(self, offset: int, size: int) -> bytes:
        buffer = self._mmap_handle[offset : offset + size]
        if len(buffer) != size:
            raise MismatchError("read mismatch", len(buffer), size)
        return buffer

    def read_range(self, offset: int, terminal: int) -> bytes:
        return self.read(offset, terminal - offset)

    def read_file(self, entry: FileEntry, decompress: bool = True) -> bytes:
        raw = self.read(entry.data_offset, entry.compressed_size)
        if not decompress:
            return raw
        if entry.storage_type in [StorageType.STORE]:
            return raw
        if entry.storage_type in [
            StorageType.BUFFER_COMPRESS,
            StorageType.STREAM_COMPRESS,
        ]:
            zlib_buffer = zlib.decompress(raw)
            if len(zlib_buffer) != entry.decompressed_size:
                raise MismatchError(
                    "size mismatch", len(zlib_buffer), entry.decompressed_size
                )
            return zlib_buffer
        raise RelicToolError(
            f"read_buffer does not support StorageType='{entry.storage_type}'"
        )

    def open_file(self, entry: FileEntry, decompress: bool = True) -> BytesIO:
        return BytesIO(self.read_file(entry, decompress))

    def read_files_parallel(
        self, file_paths: List[FileEntry], num_workers: int
    ) -> List[ReadResult[bytes]]:
        """Read and decompress files in PARALLEL."""

        def read_decompress(entry: FileEntry) -> ReadResult[bytes]:
            try:
                data = self.read_file(entry)
                return ReadResult(entry.path, data, None)
            except Exception as e:
                return ReadResult(entry.path, b"", str(e))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(read_decompress, file_paths))

        return results
