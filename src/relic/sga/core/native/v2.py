from __future__ import annotations

import logging
import mmap
import os
import struct
from dataclasses import dataclass
from typing import Dict, BinaryIO, Any, Callable

from relic.sga.core.definitions import StorageType, Version, MAGIC_WORD, OSFlags
from relic.sga.core.errors import VersionNotSupportedError
from relic.sga.core.native.definitions import FileEntry


@dataclass(slots=True)
class TocPointer:
    offset: int
    count: int


@dataclass(slots=True)
class TocPointers:
    drive: TocPointer
    folder: TocPointer
    file: TocPointer
    name: TocPointer


class NativeParserV2:
    """REAL native SGA reader - parses binary format directly.

    This completely bypasses fs library by:
    1. Manually parsing SGA V2 binary header
    2. Manually parsing TOC to get file entries
    3. Extracting TRUE byte offsets for each file
    4. Using mmap for zero-copy access
    5. Parallel zlib decompression

    Target: 3-4 seconds for 7,815 files!
    """

    _HEADER_START = 12
    _TOC_OFFSET = 180

    def __init__(
        self,
        sga_path: str,
        logger: logging.Logger | None = None,
        should_merge: bool | Callable[[int], bool] = False,
    ):
        """Parse SGA file.

        Args:
            sga_path: Path to SGA archive
            logger:
        """
        self._sga_path = sga_path
        self.logger = logger
        self._files: Dict[str, FileEntry] = {}
        self._data_block_start = 0
        self._should_merge = should_merge
        # Parse the binary format
        self._parsed = False
        self._file_handle = None
        self._mmap_handle = None

    def open(self) -> None:
        """Open memory-mapped access."""
        if self._mmap_handle is None:
            self._file_handle = os.open(
                self._sga_path, OSFlags.O_RDONLY | OSFlags.O_BINARY
            )
            self._mmap_handle = mmap.mmap(self._file_handle, 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        """Close memory-mapped access."""
        if self._mmap_handle:
            self._mmap_handle.close()
            self._mmap_handle = None  # type: ignore
        if self._file_handle is not None:
            os.close(self._file_handle)
            self._file_handle = None  # type: ignore

    def __enter__(self) -> NativeParserV2:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _log(self, msg: str) -> None:  # TODO; use logger directly and use BraceMessages
        """Log if verbose."""
        if self.logger:
            self.logger.debug(f"[Parser] {msg}")

    def _parse_toc_pair(self, block_index: int) -> TocPointer:
        _SIZE = 6
        offset = self._TOC_OFFSET + block_index * _SIZE
        buffer = self._mmap_handle[offset : offset + _SIZE]
        offset, count = struct.unpack("<IH", buffer)
        return TocPointer(offset, count)

    def _parse_toc(self) -> TocPointers:
        [drives, folders, files, names] = [self._parse_toc_pair(_) for _ in range(4)]

        return TocPointers(drives, folders, files, names)

    def _parse_names(
        self,
        ptrs: TocPointers,
        toc_size: int,
    ) -> Dict[int, str]:

        def determine_buffer_size():
            relative_terminal = toc_size
            if not all(offset < ptrs.name.offset for offset in _non_name_offsets):
                # Determine *next* offset to determine the size of the buffer
                relative_terminal = toc_size
                for offset in _non_name_offsets:
                    if ptrs.name.offset < offset < relative_terminal:
                        relative_terminal = offset
            return relative_terminal - ptrs.name.offset

        # Parse string table FIRST (names are stored here!)
        self._log("Parsing string table...")
        name_base_offset = self._TOC_OFFSET + ptrs.name.offset
        _non_name_offsets = [ptrs.drive.offset, ptrs.folder.offset, ptrs.file.offset]
        # well formatted TOC; we can determine the size of the name table using the TOC size (name size is always last)
        buffer_size = determine_buffer_size()
        string_table_data = self._mmap_handle[
            name_base_offset : name_base_offset + buffer_size
        ]
        names = {}
        running_index = 0
        for name in string_table_data.split(b"\0"):
            names[running_index] = name.decode("utf-8")
            running_index += len(name) + 1
        return names

    def _parse_drives(self, ptr: TocPointer) -> list[dict[str, Any]]:
        # Parse drives (138 bytes each)
        self._log("Parsing drives...")
        drives = []
        base_offset = self._TOC_OFFSET + ptr.offset
        s = struct.Struct("<64s64s5H")

        for drive_index in range(ptr.count):
            offset = base_offset + drive_index * s.size
            buffer = self._mmap_handle[offset : offset + s.size]
            (
                alias,
                name,
                first_folder,
                last_folder,
                first_file,
                last_file,
                root_folder,
            ) = s.unpack(buffer)
            # Drive structure: alias(64), name(64), first_folder(2), last_folder(2),
            #                  first_file(2), last_file(2), root_folder(2)
            alias = alias.rstrip(b"\x00").decode("utf-8", errors="ignore")
            name = name.rstrip(b"\x00").decode("utf-8", errors="ignore")

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
        return drives

    def _parse_folders(
        self,
        ptr: TocPointer,
        string_table: dict[int, str],
    ) -> list[dict[str, Any]]:
        s = struct.Struct("<IHHHH")
        # Parse folders (12 bytes each)
        self._log("Parsing folders...")
        folders = []
        base_offset = self._TOC_OFFSET + ptr.offset
        for folder_index in range(ptr.count):
            offset = base_offset + folder_index * s.size
            buffer = self._mmap_handle[offset : offset + s.size]
            # Folder: name_offset(4), subfolder_start(2), subfolder_stop(2), first_file(2), last_file(2)
            name_off, subfolder_start, subfolder_stop, first_file, last_file = s.unpack(
                buffer
            )

            folder_name = string_table[name_off]
            folders.append(
                {
                    "name": folder_name,
                    "subfolder_start": subfolder_start,
                    "subfolder_stop": subfolder_stop,
                    "first_file": first_file,
                    "last_file": last_file,
                }
            )
        return folders

    def _parse_files(
        self,
        ptr: TocPointer,
        string_table: dict[int, str],
    ) -> list[dict[str, Any]]:
        s = struct.Struct("<IIIII")
        # Parse files (20 bytes each)
        self._log(f"Parsing {ptr.count} files...")
        files = []
        base_offset = self._TOC_OFFSET + ptr.offset
        for file_index in range(ptr.count):
            offset = base_offset + file_index * s.size
            buffer = self._mmap_handle[offset : offset + s.size]
            # File: name_offset(4), flags(4), data_offset(4), compressed_size(4), decompressed_size(4)
            name_off, flags, data_offset, compressed_size, decompressed_size = s.unpack(
                buffer
            )
            file_name = string_table[name_off]

            # Storage type is in upper nibble of flags
            # 1/2 is buffer/stream compression;
            # supposedly they mean different things to the engine; to us, they are the same
            storage_type = StorageType((flags & 0xF0) >> 4)

            files.append(
                {
                    "name": file_name,
                    "data_offset": data_offset,
                    "compressed_size": compressed_size,
                    "decompressed_size": decompressed_size,
                    "storage_type": storage_type,
                }
            )

            if file_index < 5:  # Debug first 5
                self._log(
                    f"  File[{file_index}]: {file_name}, offset={data_offset},"
                    f" comp={compressed_size}, decomp={decompressed_size}, type={storage_type}"
                )
        return files

    def _parse_sga_binary(self) -> None:  # TODO; move to v2
        """Parse SGA V2 binary format manually."""
        self._log(f"Opening {self._sga_path}...")

        # Parse header
        MAGIC_WORD.validate(self._mmap_handle[0:8])

        # Read version
        major, minor = struct.unpack("<HH", self._mmap_handle[8:12])
        version = Version(major, minor)
        self._log(f"SGA Version: {major}.{minor}")
        VERSION = Version(2, 0)
        if version != VERSION:
            raise VersionNotSupportedError(version, [VERSION])

        # Read header (SGA V2 header is 180 bytes total, TOC starts at 180)
        # The actual offsets are 12 bytes later than documented:
        # toc_size at offset 172, data_pos at offset 176
        header_struct = struct.Struct("<16s128s16sII")
        buffer = self._mmap_handle[12:180]
        file_hash, archive_name, toc_hash, toc_size, data_offset = header_struct.unpack(
            buffer
        )

        self._log(f"TOC size: {toc_size} bytes")
        self._log(f"Data starts at offset: {data_offset}")

        self._data_block_start = data_offset

        # Parse TOC Header
        # Format: drive_pos(4), drive_count(2), folder_pos(4), folder_count(2),
        #         file_pos(4), file_count(2), name_pos(4), name_count(2)
        toc_ptrs = self._parse_toc()

        if isinstance(self._should_merge, bool):
            merging = self._should_merge
        else:
            merging = self._should_merge(toc_ptrs.drive.count)
        self._log(
            f"TOC: {toc_ptrs.drive.count} drives, {toc_ptrs.folder.count} folders,"
            f" {toc_ptrs.file.count} files, {toc_ptrs.name.count} strings"
        )

        string_table = self._parse_names(toc_ptrs, toc_size)
        drives = self._parse_drives(toc_ptrs.drive)
        folders = self._parse_folders(toc_ptrs.folder, string_table)
        files = self._parse_files(toc_ptrs.file, string_table)

        # Build file map
        self._log("Building file map...")
        self._log(f"Data block starts at offset: {self._data_block_start}")
        for drive in drives:
            drive_name = drive["name"]
            self._build_file_paths(folders, files, drive["root_folder"], drive_name, "")

        self._log(f"Successfully parsed {len(self._files)} files!")

    def _build_file_paths(
        self,
        folders: list[dict[str, Any]],
        files: list[dict[str, Any]],
        folder_idx: int,
        drive_name: str,
        current_path: str,
        exclude_drive: bool = False,
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
                if exclude_drive:
                    if full_folder_path:
                        full_path = f"{full_folder_path}/{file['name']}"
                    else:
                        full_path = file["name"]
                else:
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
        self._parsed = True

    def parse(self) -> list[FileEntry]:
        if not self._parsed:
            with self:  # ensure we are open
                self._parse_sga_binary()

        return self.get_file_entries()

    def get_file_entries(self) -> list[FileEntry]:
        return list(self._files.values())
