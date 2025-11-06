"""Native SGA Binary Parser - Direct Binary Format Implementation

Parses SGA V2 binary format directly to extract byte offsets.
Uses mmap + parallel zlib for ultra-fast extraction (3-4 seconds for 7,815 files).

Completely bypasses fs library for maximum speed!
"""

from __future__ import annotations

import mmap
import os
import struct
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Literal

if sys.platform == "win32":  # mypy *ONLY* checks for win32; as specified in pep0484
    _RB_FLAG = os.O_RDONLY | os.O_BINARY
else:
    _RB_FLAG = os.O_RDONLY


@dataclass(slots=True)
class FileEntry:
    """File entry with absolute byte offset in SGA file."""

    path: str
    data_offset: int  # Absolute byte offset in .sga file
    compressed_size: int
    decompressed_size: int
    storage_type: int  # 0=uncompressed, 1/2=zlib


class NativeSGAReader:
    """REAL native SGA reader - parses binary format directly.

    This completely bypasses fs library by:
    1. Manually parsing SGA V2 binary header
    2. Manually parsing TOC to get file entries
    3. Extracting TRUE byte offsets for each file
    4. Using mmap for zero-copy access
    5. Parallel zlib decompression

    Target: 3-4 seconds for 7,815 files!
    """

    def __init__(self, sga_path: str, verbose: bool = True):
        """Parse SGA file.

        Args:
            sga_path: Path to SGA archive
            verbose: Print parsing progress
        """
        self.sga_path = sga_path
        self.verbose = verbose
        self._files: Dict[str, FileEntry] = {}
        self._mmap_handle: Optional[mmap.mmap] = None
        self._file_handle: Optional[int] = None
        self._data_block_start = 0

        # Parse the binary format
        self._parse_sga_binary()

    def _log(self, msg: str) -> None:
        """Log if verbose."""
        if self.verbose:
            print(f"[Parser] {msg}")  # TODO use logger

    def _parse_sga_binary(self) -> None:  # TODO; move to v2
        """Parse SGA V2 binary format manually."""
        self._log(f"Opening {self.sga_path}...")

        with open(self.sga_path, "rb") as f:
            # Parse header
            magic = f.read(8)
            if magic != b"_ARCHIVE":
                raise ValueError(
                    f"Not a valid SGA file! Magic: {magic!r}"
                )  # TODO: Use MisMatchError

            # Read version
            major, minor = struct.unpack("<HH", f.read(4))
            self._log(f"SGA Version: {major}.{minor}")

            if major != 2:
                raise ValueError(
                    f"Only SGA V2 supported, got V{major}.{minor}"
                )  # TODO: Use VersionNotSupportedError

            # Read header (SGA V2 header is 180 bytes total, TOC starts at 180)
            # The actual offsets are 12 bytes later than documented:
            # toc_size at offset 172, data_pos at offset 176
            f.seek(172)
            toc_size = struct.unpack("<I", f.read(4))[0]
            data_pos = struct.unpack("<I", f.read(4))[0]

            self._log(f"TOC size: {toc_size} bytes")
            self._log(f"Data starts at offset: {data_pos}")

            self._data_block_start = data_pos

            # TOC starts at offset 180 for V2
            f.seek(180)

            # Parse TOC Header
            # Format: drive_pos(4), drive_count(2), folder_pos(4), folder_count(2),
            #         file_pos(4), file_count(2), name_pos(4), name_count(2)
            drive_offset = struct.unpack("<I", f.read(4))[0]
            drive_count = struct.unpack("<H", f.read(2))[0]
            folder_offset = struct.unpack("<I", f.read(4))[0]
            folder_count = struct.unpack("<H", f.read(2))[0]
            file_offset = struct.unpack("<I", f.read(4))[0]
            file_count = struct.unpack("<H", f.read(2))[0]
            name_offset = struct.unpack("<I", f.read(4))[0]
            name_count = struct.unpack("<I", f.read(4))[0]

            self._log(
                f"TOC: {drive_count} drives, {folder_count} folders, {file_count} files, {name_count} strings"
            )

            # TOC base is at 180
            toc_base = 180

            # Parse string table FIRST (names are stored here!)
            self._log("Parsing string table...")
            f.seek(toc_base + name_offset)
            string_table_data = f.read(name_count)

            def read_string_at_offset(offset: int) -> str:
                """Read null-terminated string from string table."""
                end = string_table_data.find(b"\x00", offset)
                if end == -1:
                    end = len(string_table_data)
                return string_table_data[offset:end].decode("utf-8", errors="ignore")

            # Parse drives (138 bytes each)
            self._log("Parsing drives...")
            drives = []
            f.seek(toc_base + drive_offset)
            for _ in range(drive_count):
                # Drive structure: alias(64), name(64), first_folder(2), last_folder(2),
                #                  first_file(2), last_file(2), root_folder(2)
                alias = f.read(64).rstrip(b"\x00").decode("utf-8", errors="ignore")
                name = f.read(64).rstrip(b"\x00").decode("utf-8", errors="ignore")
                first_folder, last_folder, first_file, last_file, root_folder = (
                    struct.unpack("<HHHHH", f.read(10))
                )
                drives.append(
                    {
                        "alias": alias,
                        "name": name,
                        "root_folder": root_folder,
                        "first_folder": first_folder,
                        "last_folder": last_folder,
                        "first_file": first_file,
                        "last_file": last_file,
                    }
                )
                self._log(f"  Drive: {name} (root folder: {root_folder})")

            # Parse folders (12 bytes each)
            self._log("Parsing folders...")
            folders = []
            f.seek(toc_base + folder_offset)
            for i in range(folder_count):
                # Folder: name_offset(4), subfolder_start(2), subfolder_stop(2), first_file(2), last_file(2)
                name_off, subfolder_start, subfolder_stop, first_file, last_file = (
                    struct.unpack("<IHHHH", f.read(12))
                )
                folder_name = read_string_at_offset(name_off)
                folders.append(
                    {
                        "name": folder_name,
                        "subfolder_start": subfolder_start,
                        "subfolder_stop": subfolder_stop,
                        "first_file": first_file,
                        "last_file": last_file,
                    }
                )

            # Parse files (20 bytes each)
            self._log(f"Parsing {file_count} files...")
            files = []
            f.seek(toc_base + file_offset)
            for i in range(file_count):
                # File: name_offset(4), flags(4), data_offset(4), compressed_size(4), decompressed_size(4)
                name_off, flags, data_offset, compressed_size, decompressed_size = (
                    struct.unpack("<IIIII", f.read(20))
                )
                file_name = read_string_at_offset(name_off)

                # Storage type is in upper nibble of flags
                storage_type = (flags & 0xF0) >> 4

                files.append(
                    {
                        "name": file_name,
                        "data_offset": data_offset,
                        "compressed_size": compressed_size,
                        "decompressed_size": decompressed_size,
                        "storage_type": storage_type,
                    }
                )

                if i < 5:  # Debug first 5
                    self._log(
                        f"  File[{i}]: {file_name}, offset={data_offset}, comp={compressed_size}, decomp={decompressed_size}, type={storage_type}"
                    )

            # Build file map
            self._log("Building file map...")
            self._log(f"Data block starts at offset: {self._data_block_start}")
            for drive in drives:
                drive_name = drive["name"]
                self._build_file_paths(
                    folders, files, drive["root_folder"], drive_name, ""
                )

            self._log(f"Successfully parsed {len(self._files)} files!")

    def _build_file_paths(
        self,
        folders: list[dict[str, Any]],
        files: list[dict[str, Any]],
        folder_idx: int,
        drive_name: str,
        current_path: str,
    ) -> None:
        """Recursively build full file paths."""
        if folder_idx >= len(folders):
            return

        folder = folders[folder_idx]
        folder_name = folder["name"]

        # Normalize folder name (remove backslashes)
        folder_name = folder_name.replace("\\", "/")

        # Build folder path - folder names are often full paths from root, not relative
        # So we just use the folder_name directly
        full_folder_path = folder_name if folder_name else current_path

        # Add files in this folder
        for file_idx in range(folder["first_file"], folder["last_file"]):
            if file_idx < len(files):
                file = files[file_idx]

                # Build full path
                if full_folder_path:
                    full_path = f"{drive_name}/{full_folder_path}/{file['name']}"
                else:
                    full_path = f"{drive_name}/{file['name']}"

                # Create entry - data_offset is RELATIVE to data block!
                # Absolute offset = data_block_start + data_offset
                entry = FileEntry(
                    path=full_path,
                    data_offset=self._data_block_start
                    + file["data_offset"],  # Make it absolute!
                    compressed_size=file["compressed_size"],
                    decompressed_size=file["decompressed_size"],
                    storage_type=file["storage_type"],
                )
                self._files[full_path] = entry

        # Recurse into subfolders
        for subfolder_idx in range(folder["subfolder_start"], folder["subfolder_stop"]):
            self._build_file_paths(
                folders, files, subfolder_idx, drive_name, full_folder_path
            )

    def open_mmap(self) -> None:
        """Open memory-mapped access."""
        if self._mmap_handle is None:
            self._file_handle = os.open(self.sga_path, _RB_FLAG)
            self._mmap_handle = mmap.mmap(self._file_handle, 0, access=mmap.ACCESS_READ)

    def close_mmap(self) -> None:
        """Close memory-mapped access."""
        if self._mmap_handle:
            self._mmap_handle.close()
            self._mmap_handle = None
        if self._file_handle is not None:
            os.close(self._file_handle)
            self._file_handle = None

    def read_file(self, file_path: str) -> bytes:
        """Read file using mmap + decompress.

        Args:
            file_path: Full path to file

        Returns:
            Decompressed file data
        """
        # Normalize path (convert backslashes to forward slashes)
        file_path = file_path.replace("\\", "/").strip("/")

        if file_path not in self._files:
            raise KeyError(f"File not found: {file_path}")

        entry = self._files[file_path]

        # Open mmap if needed
        if self._mmap_handle is None:
            self.open_mmap()

        # Read compressed data from TRUE byte offset!
        compressed_data: bytes = self._mmap_handle[
            entry.data_offset : entry.data_offset + entry.compressed_size
        ]  # type:ignore

        # Decompress if needed (storage_type: 0=STORE, 1=STREAM_COMPRESS, 2=BUFFER_COMPRESS)
        if entry.storage_type != 0:
            # Use decompressobj() like relic does
            decompressor = zlib.decompressobj()
            data = decompressor.decompress(bytes(compressed_data))
            data += decompressor.flush()
        else:
            data = bytes(compressed_data)

        return data

    def read_files_parallel(
        self, file_paths: List[str], num_workers: int = 16
    ) -> List[Tuple[str, bytes, str | None]]:
        """Read and decompress files in PARALLEL."""
        if self._mmap_handle is None:
            self.open_mmap()

        def read_decompress(file_path: str) -> Tuple[str, bytes, str | None]:
            try:
                data = self.read_file(file_path)
                return (file_path, data, None)
            except Exception as e:
                return (file_path, b"", str(e))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(read_decompress, file_paths))

        return results

    def list_files(self) -> List[str]:
        """Get list of all files."""
        return list(self._files.keys())

    def __enter__(self) -> NativeSGAReader:
        self.open_mmap()
        return self

    def close(self) -> None:
        self.close_mmap()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        self.close_mmap()
        return False
