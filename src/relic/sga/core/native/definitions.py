from __future__ import annotations

import datetime
import mmap
import os
from dataclasses import dataclass, field
from typing import Self, Any, TypeVar, Generic, List, cast

from relic.core.errors import MismatchError

from relic.sga.core.definitions import StorageType, OSFlags


@dataclass(slots=True)
class FileEntry:
    """File entry with absolute byte offset in SGA file."""

    drive: str
    folder_path: str
    name: str

    def full_path(self, include_drive: bool = True) -> str:
        if include_drive:
            return os.path.join(self.drive, self.folder_path, self.name)
        else:
            return os.path.join(self.folder_path, self.name)

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
    timings: ExtractionTimings = field(default_factory=ExtractionTimings)


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
_TIn = TypeVar("_TIn")
_TOut = TypeVar("_TOut")


@dataclass(slots=True)
class Result(Generic[_TIn, _TOut]):

    input: _TIn
    output: _TOut | None = None
    errors: List[str | Exception] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @classmethod
    def create_error(cls, input: _TIn, *errors: str | Exception) -> Result[_TIn, _TOut]:
        return cls(input=input, output=None, errors=list(errors))


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

    def _read(self, offset: int, size: int) -> bytes:
        buffer = self._mmap_handle[offset : offset + size]
        if len(buffer) != size:
            raise MismatchError("Read", size, len(buffer))
        return buffer

    def _read_range(self, offset: int, terminal: int) -> bytes:
        return self._read(offset, terminal - offset)
