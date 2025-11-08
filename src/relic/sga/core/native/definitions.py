from __future__ import annotations

import datetime
from dataclasses import dataclass

from relic.sga.core.definitions import StorageType


@dataclass(slots=True)
class FileEntry:
    """File entry with absolute byte offset in SGA file."""

    path: str
    data_offset: int  # Absolute byte offset in .sga file
    compressed_size: int
    decompressed_size: int
    storage_type: StorageType
    modified: datetime.datetime | None = None


@dataclass(slots=True)  # Use __slots__ for 50% memory reduction!
class ExtractionStats:
    """Statistics for extraction operation."""

    total_files: int = 0
    extracted_files: int = 0
    failed_files: int = 0
    total_bytes: int = 0
    extracted_bytes: int = 0
    skipped_files: int = 0


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


@dataclass(slots=True)
class ReadResult:

    path: str
    data: bytes | None
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
