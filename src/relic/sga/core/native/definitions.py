from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)  # Use __slots__ for 50% memory reduction!
class FileEntry:
    """Metadata for a single file in the archive."""

    path: str
    size: int
    compressed_size: int
    storage_type: str  # 'store', 'zlib', etc.
    checksum: Optional[str] = None

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.compressed_size == 0:
            return 1.0
        return self.size / self.compressed_size

    @property
    def is_compressed(self) -> bool:
        """Check if file is compressed."""
        return self.storage_type.lower() != "store"


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
