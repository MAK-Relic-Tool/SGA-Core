from __future__ import annotations

import datetime
import mmap
import os
from dataclasses import dataclass, field
from typing import Self, Any, TypeVar, Generic

from relic.sga.core.definitions import StorageType, OSFlags


@dataclass(slots=True)
class FileEntry:
    """File entry with absolute byte offset in SGA file."""

    path: str
    data_offset: int  # Absolute byte offset in .sga file
    compressed_size: int
    decompressed_size: int
    storage_type: StorageType
    modified: datetime.datetime | None = None


@dataclass(slots=True)
class ExtractionTimings:
    parsing_sga: float = 0
    filtering_files: float = 0
    creating_dirs: float = 0
    creating_batches: float = 0
    executing_batches: float = 0
    parsing_results: float = 0

    @property
    def total_time(self) -> float:
        return sum(
            [
                self.parsing_sga,
                self.filtering_files,
                self.creating_dirs,
                self.creating_batches,
                self.executing_batches,
                self.parsing_results,
            ]
        )


@dataclass(slots=True)  # Use __slots__ for 50% memory reduction!
class ExtractionStats:
    """Statistics for extraction operation."""

    total_files: int = 0
    extracted_files: int = 0
    failed_files: int = 0
    total_bytes: int = 0
    extracted_bytes: int = 0
    skipped_files: int = 0
    timings: ExtractionTimings = field(default_factory=lambda: ExtractionTimings())


@dataclass(slots=True)
class ExtractionPlan:
    total_files: int
    total_bytes: int
    categories: dict[str, ExtractionPlanCategory]
    estimated_time_seconds: float
    recommended_workers: int


@dataclass(slots=True)
class ExtractionPlanCategory:
    file_count: int
    total_bytes: int
    workers: int


_T = TypeVar("_T")


@dataclass(slots=True)
class ReadResult(Generic[_T]):

    path: str
    data: _T | None
    error: str | None = None


@dataclass(slots=True)
class BatchResult:

    success: bool
    path: str
    error: str | None = None


@dataclass(slots=True)
class WriteResult:

    path: str
    success: bool
    error: str | None = None


@dataclass(slots=True)
class ChecksumResult:
    path: str
    checksum: str | None
    error: str | None = None


class ReadonlyMemMapFile:
    def __init__(self, path: str):
        self._file_path = path
        self._file_handle: int = None  # type: ignore
        self._mmap_handle: mmap.mmap = None  # type: ignore

    def open(self) -> None:
        """Open memory-mapped access."""
        if self._mmap_handle is None:
            self._file_handle = os.open(
                self._file_path, OSFlags.O_RDONLY | OSFlags.O_BINARY
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

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
