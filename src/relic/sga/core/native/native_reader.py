"""Native SGA Binary Parser - Direct Binary Format Implementation

Parses SGA V2 binary format directly to extract byte offsets.
Uses mmap + parallel zlib for ultra-fast extraction (3-4 seconds for 7,815 files).

Completely bypasses fs library for maximum speed!
"""

from __future__ import annotations

import mmap
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, List, Tuple

from relic.core.errors import MismatchError

from relic.sga.core.definitions import OSFlags, StorageType
from relic.sga.core.native.definitions import FileEntry, ReadResult


class SgaReader:
    """
    Reads an SGA file using FileEntry objects.

    FileEntry objects contain ABSOLUTE byte offsets within the SGA file
    as well as the file's size (compressed_size)
    and the file's storage_type

    These three keys allow us to quickly read raw sga files and decompress them if needed
    """

    def __init__(self, sga_path: str):
        self._sga_path = sga_path
        self._mmap_handle: mmap.mmap = None  # type: ignore
        self._file_handle: int = None  # type: ignore

    def read_buffer(self, entry: FileEntry, decompress: bool = True) -> bytes:
        raw = self._mmap_handle[
            entry.data_offset : entry.data_offset + entry.compressed_size
        ]
        if not decompress:
            return raw
        if entry.storage_type in [StorageType.STORE]:
            return raw
        elif entry.storage_type in [
            StorageType.BUFFER_COMPRESS,
            StorageType.STREAM_COMPRESS,
        ]:
            zlib_buffer = zlib.decompress(raw)
            if len(zlib_buffer) != entry.decompressed_size:
                raise MismatchError(
                    "size mismatch", len(zlib_buffer), entry.decompressed_size
                )  # TODO
            return zlib_buffer
        raise NotImplementedError  # TODO

    def read_file(self, entry: FileEntry, decompress: bool = True) -> BytesIO:
        return BytesIO(self.read_buffer(entry, decompress))

    def read_files_parallel(
        self, file_paths: List[FileEntry], num_workers: int = 16
    ) -> List[ReadResult]:
        """Read and decompress files in PARALLEL."""

        def read_decompress(entry: FileEntry) -> ReadResult:
            try:
                data = self.read_buffer(entry)
                return ReadResult(entry.path, data, None)
            except Exception as e:
                return ReadResult(entry.path, b"", str(e))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(read_decompress, file_paths))

        return results

    def open(self) -> None:
        """Open memory-mapped access."""
        if self._mmap_handle is None:
            self._file_handle = os.open(
                self._sga_path, OSFlags.O_RDONLY | OSFlags.O_BINARY
            )
            self._mmap_handle = mmap.mmap(self._file_handle, 0, access=mmap.ACCESS_READ)
        if self._mmap_handle is None:
            raise Exception("failed to init mmap")

    def close(self) -> None:
        """Close memory-mapped access."""
        if self._mmap_handle:
            self._mmap_handle.close()
            self._mmap_handle = None  # type: ignore
        if self._file_handle is not None:
            os.close(self._file_handle)
            self._file_handle = None  # type: ignore

    def __enter__(self) -> SgaReader:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
