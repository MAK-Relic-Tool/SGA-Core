from __future__ import annotations

import mmap
import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, BinaryIO, List, Optional, Any, Literal

from relic.sga.core.definitions import OSFlags, StorageType
from relic.sga.core.native.definitions import FileEntry


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


    def _parse_toc_pair(self, f:BinaryIO) -> tuple[int,int]:
        offset = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<H", f.read(2))[0]
        return offset, count


    def _parse_toc(self, f:BinaryIO) -> tuple[int,int,int,int,int,int,int,int]:
        (dO, dC) = self._parse_toc_pair(f)
        (foldO, foldC)  = self._parse_toc_pair(f)
        (fileO, fileC) = self._parse_toc_pair(f)
        (nameO, nameC) = self._parse_toc_pair(f)
        return dO, dC, foldO, foldC, fileO, fileC, nameO, nameC

    def _parse_names(self, f:BinaryIO, drive_offset:int, folder_offset:int, file_offset:int, name_offset:int, toc_size:int) -> Dict[int,str]:
        toc_base = 180
        # Parse string table FIRST (names are stored here!)
        self._log("Parsing string table...")
        f.seek(toc_base + name_offset)
        _non_name_offsets = [drive_offset, folder_offset, file_offset]
        # well formatted TOC; we can determine the size of the name table using the TOC size (name size is always last)
        name_buffer_terminal = toc_size
        if not all(offset < name_offset for offset in _non_name_offsets):
            # Determine *next* offset to determine the size of the buffer
            name_buffer_terminal = toc_size
            for offset in _non_name_offsets:
                if name_offset < offset < name_buffer_terminal:
                    name_buffer_terminal = offset
        string_table_data = f.read(name_buffer_terminal - name_offset)
        names = {}
        running_index = 0
        for name in string_table_data.split(b"\0"):
            names[running_index] = name.decode("utf-8")
            running_index += len(name) + 1
        return names


    def _parse_drives(self, f:BinaryIO, drive_offset:int, drive_count:int) -> list[dict[str,Any]]:
        toc_base = 180
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
        return drives


    def _parse_folders(self, f:BinaryIO, folder_offset:int, folder_count:int, string_table:dict[int,str]) -> list[dict[str,Any]]:
        toc_base = 180

        # Parse folders (12 bytes each)
        self._log("Parsing folders...")
        folders = []
        f.seek(toc_base + folder_offset)
        for i in range(folder_count):
            # Folder: name_offset(4), subfolder_start(2), subfolder_stop(2), first_file(2), last_file(2)
            name_off, subfolder_start, subfolder_stop, first_file, last_file = (
                struct.unpack("<IHHHH", f.read(12))
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

    def _parse_files(self, f:BinaryIO, file_offset:int, file_count:int, string_table:dict[int,str]) -> list[dict[str, Any]]:
        toc_base = 180
        # Parse files (20 bytes each)
        self._log(f"Parsing {file_count} files...")
        files = []
        f.seek(toc_base + file_offset)
        for i in range(file_count):
            # File: name_offset(4), flags(4), data_offset(4), compressed_size(4), decompressed_size(4)
            name_off, flags, data_offset, compressed_size, decompressed_size = (
                struct.unpack("<IIIII", f.read(20))
            )
            file_name = string_table[name_off]

            # Storage type is in upper nibble of flags
            # 1/2 is buffer/stream compression; supposedly they mean different things to the engine; to us, they are the same
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

            if i < 5:  # Debug first 5
                self._log(
                    f"  File[{i}]: {file_name}, offset={data_offset}, comp={compressed_size}, decomp={decompressed_size}, type={storage_type}"
                )
        return files


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
            drive_offset, drive_count, folder_offset, folder_count, file_offset, file_count, name_offset, name_count = self._parse_toc(f)

            self._log(
                f"TOC: {drive_count} drives, {folder_count} folders, {file_count} files, {name_count} strings"
            )

            # TOC base is at 180
            toc_base = 180

            string_table = self._parse_names(f, drive_offset, folder_offset, file_offset, name_offset, toc_size)
            drives = self._parse_drives(f,drive_offset, drive_count)
            folders = self._parse_folders(f,folder_offset, folder_count, string_table)
            files = self._parse_files(f,file_offset,file_count,string_table)

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
            self._file_handle = os.open(
                self.sga_path, OSFlags.O_RDONLY | OSFlags.O_BINARY
            )
            self._mmap_handle = mmap.mmap(self._file_handle, 0, access=mmap.ACCESS_READ)

    def close_mmap(self) -> None:
        """Close memory-mapped access."""
        if self._mmap_handle:
            self._mmap_handle.close()
            self._mmap_handle = None
        if self._file_handle is not None:
            os.close(self._file_handle)
            self._file_handle = None
    #
    # def read_file(self, file_path: str) -> bytes:
    #     """Read file using mmap + decompress.
    #
    #     Args:
    #         file_path: Full path to file
    #
    #     Returns:
    #         Decompressed file data
    #     """
    #     # Normalize path (convert backslashes to forward slashes)
    #     file_path = file_path.replace("\\", "/").strip("/")
    #
    #     if file_path not in self._files:
    #         raise KeyError(f"File not found: {file_path}")
    #
    #     entry = self._files[file_path]
    #
    #     # Open mmap if needed
    #     if self._mmap_handle is None:
    #         self.open_mmap()
    #
    #     # Read compressed data from TRUE byte offset!
    #     compressed_data: bytes = self._mmap_handle[
    #         entry.data_offset : entry.data_offset + entry.compressed_size
    #     ]  # type:ignore
    #
    #     # Decompress if needed (storage_type: 0=STORE, 1=STREAM_COMPRESS, 2=BUFFER_COMPRESS)
    #     if entry.storage_type != 0:
    #         # Use decompressobj() like relic does
    #         decompressor = zlib.decompressobj()
    #         data = decompressor.decompress(bytes(compressed_data))
    #         data += decompressor.flush()
    #     else:
    #         data = bytes(compressed_data)
    #
    #     return data
    #
    # def read_files_parallel(
    #     self, file_paths: List[str], num_workers: int = 16
    # ) -> List[Tuple[str, bytes, str | None]]:
    #     """Read and decompress files in PARALLEL."""
    #     if self._mmap_handle is None:
    #         self.open_mmap()
    #
    #     def read_decompress(file_path: str) -> Tuple[str, bytes, str | None]:
    #         try:
    #             data = self.read_file(file_path)
    #             return (file_path, data, None)
    #         except Exception as e:
    #             return (file_path, b"", str(e))
    #
    #     with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #         results = list(executor.map(read_decompress, file_paths))
    #
    #     return results

    def list_files(self) -> List[str]:
        """Get list of all files."""
        return list(self._files.keys())

    def __enter__(self) -> NativeParserV2:
        self.open_mmap()
        return self

    def close(self) -> None:
        self.close_mmap()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        self.close_mmap()
        return False
