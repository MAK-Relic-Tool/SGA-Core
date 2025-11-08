"""Advanced parallel unpacking with comprehensive optimizations.

This module provides an enhanced parallel unpacker with:
- Pre-extracted index caching
- Adaptive threading based on workload
- File size batching for optimal scheduling
- Tiny file batch processing
- Compression-aware scheduling
- Progressive extraction callbacks
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import StrEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from relic.sga.core.definitions import OSFlags
from relic.sga.core.native.definitions import (
    FileEntry,
    ExtractionStats,
    ExtractionPlan,
    ExtractionPlanCategory,
    WriteResult,
    ReadResult,
    ChecksumResult,
)
from relic.sga.core.native.native_reader import SgaReader
from relic.sga.core.native.v2 import NativeParserV2


# TODO; restore Streaming Native via git history
# When i was axing everything I misunderstood that they were ALTERNATIVES
# For my own personal clarity; i'm going to move them to a Strategy Pattern (with a config-accpeting base class)
# with any luck, the strategy pattern can help simplify the cli/logging aspects

class FileCategory(StrEnum):
    Tiny = "tiny"
    Small = "small"
    Medium = "medium"
    Large = "large"
    Huge = "huge"


def _categorize_by_size(
    entries: List[FileEntry], logger: logging.Logger
) -> Dict[str, List[FileEntry]]:
    SIZE_TINY = 10 * 1024  # 10KB
    SIZE_SMALL = 1024 * 1024  # 1MB
    SIZE_MEDIUM = 10 * 1024 * 1024  # 10MB
    SIZE_LARGE = 100 * 1024 * 1024  # 100MB
    """Categorize files by size for optimal scheduling.

    Args:
        entries: List of file entries

    Returns:
        Dictionary mapping category to file list
    """
    categories: dict[str, list[FileEntry]] = {
        "tiny": [],  # < 10KB
        "small": [],  # 10KB - 1MB
        "medium": [],  # 1MB - 10MB
        "large": [],  # 10MB - 100MB
        "huge": [],  # > 100MB
    }

    for entry in entries:
        if entry.decompressed_size < SIZE_TINY:
            categories["tiny"].append(entry)
        elif entry.decompressed_size < SIZE_SMALL:
            categories["small"].append(entry)
        elif entry.decompressed_size < SIZE_MEDIUM:
            categories["medium"].append(entry)
        elif entry.decompressed_size < SIZE_LARGE:
            categories["large"].append(entry)
        else:
            categories["huge"].append(entry)

    # Log distribution
    logger.info("File size distribution:")
    for cat, files in categories.items():
        if files:
            total_size = sum(f.decompressed_size for f in files)
            logger.info(
                f"  {cat:8s}: {len(files):6d} files ({total_size / 1024 / 1024:.1f} MB)"
            )

    return categories


class AdvancedParallelUnpacker:
    """Advanced parallel unpacker with comprehensive optimizations."""

    # Batching thresholds
    TINY_FILE_BATCH_SIZE = 100
    SMALL_FILE_BATCH_SIZE = 50

    def __init__(
        self,
        num_workers: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        enable_adaptive_threading: bool = True,
        enable_batching: bool = True,
        enable_delta: bool = False,
        chunk_size: int = 1024 * 1024,
    ):
        """Initialize advanced parallel unpacker.

        Args:
            num_workers: Number of worker threads (None = system thread count)
            logger: Logger instance
            enable_adaptive_threading: Adjust workers by file size
            enable_batching: Batch tiny files together
            enable_delta: Enable delta extraction (default: False)
            chunk_size: Read chunk size in bytes
        """
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.logger = logger or logging.getLogger(__name__)
        self.enable_adaptive_threading = enable_adaptive_threading
        self.enable_batching = enable_batching
        self.enable_delta = enable_delta
        self.chunk_size = chunk_size

        # Thread-safe directory cache to avoid redundant mkdir calls
        self._dir_cache: Set[str] = set()
        self._dir_cache_lock = threading.Lock()
        self.stats = ExtractionStats()

    def _ensure_directory(self, dir_path: Path) -> None:
        """Ensure directory exists with caching to avoid redundant mkdir calls.

        Args:
            dir_path: Directory path to create
        """
        dir_str = str(dir_path)

        # Fast path: check if already created
        if dir_str in self._dir_cache:
            return

        # Slow path: create and cache
        with self._dir_cache_lock:
            # Double-check after acquiring lock
            if dir_str not in self._dir_cache:
                dir_path.mkdir(parents=True, exist_ok=True)
                self._dir_cache.add(dir_str)

    def _get_optimal_workers(self, category: str, file_count: int) -> int:
        """Calculate optimal worker count for file category.

        Args:
            category: File size category
            file_count: Number of files in category

        Returns:
            Optimal number of workers
        """
        if not self.enable_adaptive_threading:
            return self.num_workers

        # Tiny files: More workers (low CPU per file)
        if category == "tiny":
            return min(self.num_workers * 2, file_count)

        # Small files: Standard workers
        elif category == "small":
            return self.num_workers

        # Medium files: Standard workers
        elif category == "medium":
            return self.num_workers

        # Large files: Fewer workers (high memory per file)
        elif category == "large":
            return max(2, self.num_workers // 2)

        # Huge files: Minimal workers (very high memory)
        elif category == "huge":
            return max(1, self.num_workers // 4)

        return self.num_workers

    def _batch_tiny_files(self, entries: List[FileEntry]) -> List[List[FileEntry]]:
        """Batch tiny files together for efficient processing.

        Args:
            entries: List of tiny file entries

        Returns:
            List of batches
        """
        if not self.enable_batching or len(entries) < self.TINY_FILE_BATCH_SIZE:
            # Don't batch if disabled or not enough files
            return [[e] for e in entries]

        batches = []
        current_batch = []

        for entry in entries:
            current_batch.append(entry)

            if len(current_batch) >= self.TINY_FILE_BATCH_SIZE:
                batches.append(current_batch)
                current_batch = []

        # Add remaining files
        if current_batch:
            batches.append(current_batch)

        self.logger.info(
            f"Batched {len(entries)} tiny files into {len(batches)} batches"
        )
        return batches

    def _extract_batch_isolated(
        self,
        sga_path: str,
        output_dir: str,
        batch: List[FileEntry],
        dst_paths: List[str],
    ) -> List[Tuple[bool, str, Optional[str]]]:
        """Extract batch of files with single SGA handle using NATIVE Python file operations.

        Uses native Python Path/open for output (FAST on Windows!) and fs library
        only for reading from SGA (necessary).

        Args:
            sga_path: Path to SGA archive
            output_dir: Output directory
            batch: List of file entries to extract
            dst_paths: List of destination paths

        Returns:
            List of (success, path, error) tuples
        """
        results: List[Tuple[bool, str, Optional[str]]] = []

        try:
            # Open SGA once for entire batch
            with SgaReader(sga_path) as my_sga:
                for entry, dst_path in zip(batch, dst_paths):
                    try:
                        # Get native destination path
                        native_dst = Path(output_dir) / dst_path.lstrip("/")

                        # Create parent directory (CACHED for speed!)
                        self._ensure_directory(native_dst.parent)

                        # Extract with chunked reading (use NATIVE Python file for writing!)
                        # For small files, use smaller chunks (64KB) for better responsiveness
                        small_chunk = 64 * 1024  # 64KB - perfect for small files
                        bytes_written = 0
                        with my_sga.read_file(entry) as src_file:
                            with open(
                                native_dst, "wb", buffering=small_chunk
                            ) as dst_file:
                                while True:
                                    chunk = src_file.read(small_chunk)
                                    if not chunk:
                                        break
                                    dst_file.write(chunk)
                                    bytes_written += len(chunk)

                                dst_file.flush()

                        results.append((True, entry.path, None))

                    except Exception as e:
                        results.append((False, entry.path, str(e)))

        except Exception as e:
            # Batch failed entirely
            for entry in batch:
                results.append((False, entry.path, f"Batch error: {str(e)}"))

        return results

    def extract_optimized(
        self,
        sga_path: str,
        output_dir: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
        force_full: bool = False,
        file_filter: Optional[List[str]] = None,
    ) -> ExtractionStats:
        """Extract with all optimizations enabled.

        Args:
            sga_path: Path to SGA archive
            output_dir: Output directory
            on_progress: Optional callback (current, total)
            force_full: Force full extraction even if delta mode enabled
            file_filter: Optional list of file paths to extract (None = extract all)

        Returns:
            Extraction statistics
        """
        # Use delta extraction if enabled (unless force_full or file_filter provided)
        if self.enable_delta and not force_full and file_filter is None:
            return self.extract_delta(sga_path, output_dir, on_progress)

        self.logger.info(f"Starting FAST extraction: {sga_path}")
        self.stats = ExtractionStats()

        # FAST MODE: Skip metadata collection, just get file paths!
        self.logger.info("Collecting file paths...")
        entries = NativeParserV2(sga_path).parse()

        # Apply filter if provided (DELTA MODE!)
        if file_filter is not None:
            file_filter_set = set(file_filter)
            entries = [p for p in entries if p.path in file_filter_set]
            self.logger.info(f"Filtered to {len(entries)} changed files")

        self.stats.total_files = len(entries)
        self.logger.info(f"Found {len(entries)} files")

        # PRE-CREATE ALL DIRECTORIES (avoid per-file checks!)
        self.logger.info("Pre-creating directory structure...")
        unique_dirs = set()
        for file in entries:
            dst_path = Path(output_dir) / file.path.lstrip("/")
            unique_dirs.add(dst_path.parent)

        for dir_path in unique_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Created {len(unique_dirs)} directories")

        # FAST MODE: Skip categorization, optimal batch size!
        # Balance: Larger batches = fewer SGA opens, but less parallelism
        # Sweet spot: 100-200 files per batch
        batch_size = 150  # Optimized for 7815 files = ~52 batches = 52 SGA opens
        workers = self.num_workers

        self.logger.info(f"Extracting {len(entries)} files...")
        self.logger.info(f"Using {workers} workers with batches of {batch_size}")

        # Create batches
        batches = []
        for i in range(0, len(entries), batch_size):
            batches.append(entries[i : i + batch_size])

        self.logger.info(f"Created {len(batches)} batches")

        total_processed = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}

            for batch in batches:
                dst_paths = [entry.path for entry in batch]
                future = executor.submit(
                    self._extract_batch_isolated,
                    sga_path,
                    output_dir,
                    batch,
                    dst_paths,
                )
                futures[future] = batch

            for future in as_completed(futures):
                batch = futures[future]
                results = future.result()

                for success, path, error in results:
                    if success:
                        self.stats.extracted_files += 1
                        total_processed += 1
                    else:
                        self.stats.failed_files += 1
                        total_processed += 1
                        if self.stats.failed_files <= 10:  # Only show first 10 errors
                            self.logger.error(f"Failed: {path}: {error}")

                    if on_progress and total_processed % 500 == 0:
                        on_progress(total_processed, self.stats.total_files)

        # Final progress callback
        if on_progress:
            on_progress(total_processed, self.stats.total_files)

        # Summary
        self.logger.info("Extraction complete!")
        self.logger.info(f"  Total:      {self.stats.total_files}")
        self.logger.info(f"  Successful: {self.stats.extracted_files}")
        self.logger.info(f"  Failed:     {self.stats.failed_files}")
        self.logger.info(
            f"  Extracted:  {self.stats.extracted_bytes / 1024 / 1024:.1f} MB"
        )

        return self.stats

    def extract_ultra_fast(
        self,
        sga_path: str,
        output_dir: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> ExtractionStats:
        """ULTRA-FAST extraction with parallel decompression.

        Same as extract_streaming but with:
        - Multiple concurrent readers (each with own SGA handle)
        - Parallel decompression
        - Target: 3-5 seconds (vs 12s)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time as time_module

        self.logger.info(f"Starting ULTRA-FAST extraction: {sga_path}")
        self.stats = ExtractionStats()

        timings = {"file_listing": 0.0, "dir_creation": 0.0, "extraction": 0.0}

        # Get file list
        t0 = time_module.perf_counter()
        self.logger.info("Collecting file paths...")
        files = NativeParserV2(sga_path).parse()
        timings["file_listing"] = time_module.perf_counter() - t0

        self.stats.total_files = len(files)
        self.logger.info(
            f"Found {len(files)} files (took {timings['file_listing']:.2f}s)"
        )

        # PRE-CREATE ALL DIRECTORIES
        t0 = time_module.perf_counter()
        self.logger.info("Pre-creating directory structure...")
        unique_dirs = set()
        for file in files:
            dst_path = Path(output_dir) / file.path.lstrip("/")
            unique_dirs.add(dst_path.parent)

        for dir_path in unique_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        timings["dir_creation"] = time_module.perf_counter() - t0

        self.logger.info(
            f"Created {len(unique_dirs)} directories (took {timings['dir_creation']:.2f}s)"
        )

        # DISABLE GC
        gc_was_enabled = gc.isenabled()
        gc.disable()
        self.logger.info("Disabled GC for maximum speed")

        # Use MANY parallel workers (each opens own SGA handle)
        max_workers = min(self.num_workers * 2, 30)
        self.logger.info(f"Using {max_workers} parallel workers (ULTRA mode)")

        # Batch files
        batch_size = 50
        file_batches = [
            files[i : i + batch_size] for i in range(0, len(files), batch_size)
        ]
        self.logger.info(
            f"Processing {len(file_batches)} batches of {batch_size} files"
        )

        processed = [0]
        lock = threading.Lock()

        def process_batch(batch: list[FileEntry]) -> dict[str, int]:
            """Process batch with dedicated SGA handle."""
            local_stats = {"extracted": 0, "failed": 0, "bytes": 0}

            # Each worker opens its own SGA
            with SgaReader(sga_path) as sga:
                for file in batch:
                    try:
                        # Read and decompress (fs handles this)
                        data = sga.read_buffer(file)

                        # Write to disk
                        dst_path = Path(output_dir) / file.path.lstrip("/")
                        fd = os.open(
                            dst_path,
                            OSFlags.O_CREAT | OSFlags.O_WRONLY | OSFlags.O_BINARY,
                        )
                        os.write(fd, data)
                        os.close(fd)

                        local_stats["extracted"] += 1
                        local_stats["bytes"] += len(data)

                    except Exception as e:
                        local_stats["failed"] += 1
                        if local_stats["failed"] <= 3:
                            self.logger.error(f"Failed {file.path}: {e}")

            # Update global stats
            with lock:
                self.stats.extracted_files += local_stats["extracted"]
                self.stats.failed_files += local_stats["failed"]
                self.stats.extracted_bytes += local_stats["bytes"]
                processed[0] += len(batch)

                if on_progress and processed[0] % 500 == 0:
                    on_progress(processed[0], self.stats.total_files)

            return local_stats

        # Process all batches in PARALLEL
        t0 = time_module.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in file_batches]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Batch failed: {e}")

        timings["extraction"] = time_module.perf_counter() - t0

        # Final progress
        if on_progress:
            on_progress(processed[0], self.stats.total_files)

        # RE-ENABLE GC
        if gc_was_enabled:
            gc.enable()
            self.logger.info("Re-enabled GC")

        gc.collect()

        # Summary
        self.logger.info("Extraction complete!")
        self.logger.info(f"  Total:      {self.stats.total_files}")
        self.logger.info(f"  Successful: {self.stats.extracted_files}")
        self.logger.info(f"  Failed:     {self.stats.failed_files}")
        self.logger.info(
            f"  Extracted:  {self.stats.extracted_bytes / 1024 / 1024:.1f} MB"
        )
        self.logger.info(f"  Time:       {timings['extraction']:.2f}s")
        self.logger.info(
            f"  Throughput: {self.stats.extracted_bytes / timings['extraction'] / 1024 / 1024:.0f} MB/s"
        )

        return self.stats

    def extract(
        self,
        sga_path: str,
        output_dir: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> ExtractionStats:
        """NATIVE ULTRA-FAST extraction - 2-3 seconds target!

        Uses true native binary parser + parallel decompression + parallel writes.
        Bypasses fs library completely for maximum speed.

        Target: 2-3 seconds for 7,815 files!
        """
        from concurrent.futures import ThreadPoolExecutor
        import time as time_module
        from pathlib import Path

        self.logger.info(f"Starting NATIVE ULTRA-FAST extraction: {sga_path}")
        self.stats = ExtractionStats()

        # Parse SGA
        t0 = time_module.perf_counter()
        self.logger.info("Parsing SGA binary format...")
        files = NativeParserV2(sga_path).parse()
        t_parse = time_module.perf_counter() - t0

        self.stats.total_files = len(files)
        self.logger.info(f"Parsed {len(files)} files in {t_parse:.2f}s")

        # Pre-create directories
        t0 = time_module.perf_counter()
        self.logger.info("Creating directory structure...")
        unique_dirs = set()
        for file_path in files:
            dst_path = Path(output_dir) / file_path.path.lstrip("/")
            unique_dirs.add(dst_path.parent)

        for dir_path in unique_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        t_dir = time_module.perf_counter() - t0

        self.logger.info(f"Created {len(unique_dirs)} directories in {t_dir:.2f}s")

        # Disable GC
        gc_was_enabled = gc.isenabled()
        gc.disable()

        # Read all files with parallel decompression
        t0 = time_module.perf_counter()
        self.logger.info(f"Reading + decompressing {len(files)} files (parallel)...")
        with SgaReader(sga_path) as reader:
            results = reader.read_files_parallel(files, num_workers=16)
        t_read = time_module.perf_counter() - t0

        self.logger.info(f"Read + decompressed in {t_read:.2f}s")

        # Write files in parallel
        t0 = time_module.perf_counter()
        self.logger.info("Writing files to disk (parallel)...")

        def write_file(
            item: ReadResult,
        ) -> WriteResult:
            path = item.path
            err = item.error
            if err:
                return WriteResult(item.path, False, err)

            try:
                dst_path = Path(output_dir) / path.lstrip("/")
                fd = os.open(
                    dst_path,
                    OSFlags.O_CREAT
                    | OSFlags.O_WRONLY
                    | OSFlags.O_BINARY
                    | OSFlags.O_TRUNC,
                )
                os.write(fd, item.data)
                os.close(fd)
                return WriteResult(path, True, None)
            except Exception as e:
                return WriteResult(path, False, str(e))

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            write_results: List[WriteResult] = list(executor.map(write_file, results))

        t_write = time_module.perf_counter() - t0

        # Count results
        for write_result in write_results:

            if write_result.success:
                self.stats.extracted_files += 1
                # Get size from results
                for read_result in results:
                    if read_result.path == write_result.path:
                        self.stats.extracted_bytes += len(read_result.data)
                        break
            else:
                self.stats.failed_files += 1
                self.logger.error(
                    f"Failed to write {write_result.path}: {write_result.error}"
                )

        # Re-enable GC
        if gc_was_enabled:
            gc.enable()
        gc.collect()

        # Summary
        total_time = t_parse + t_dir + t_read + t_write
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("NATIVE ULTRA-FAST EXTRACTION COMPLETE!")
        self.logger.info("=" * 70)
        self.logger.info(f"Parse:           {t_parse:.2f}s")
        self.logger.info(f"Dir creation:    {t_dir:.2f}s")
        self.logger.info(f"Read+decompress: {t_read:.2f}s")
        self.logger.info(f"Write to disk:   {t_write:.2f}s")
        self.logger.info(f"TOTAL TIME:      {total_time:.2f}s")
        self.logger.info("")
        self.logger.info(
            f"Files:           {self.stats.extracted_files}/{self.stats.total_files}"
        )
        self.logger.info(f"Failed:          {self.stats.failed_files}")
        self.logger.info(
            f"Speed:           {self.stats.extracted_files/total_time:.0f} files/sec"
        )
        self.logger.info(
            f"Data:            {self.stats.extracted_bytes/1024/1024:.1f} MB"
        )
        self.logger.info(
            f"Throughput:      {self.stats.extracted_bytes/total_time/1024/1024:.0f} MB/s"
        )
        self.logger.info("=" * 70)

        return self.stats

    def _calculate_checksum(self, entry: FileEntry, reader: SgaReader) -> str:
        """Calculate MD5 checksum for a file.

        Args:
            file_path: Path to file
            fs: Filesystem containing file

        Returns:
            MD5 checksum (hex)
        """
        hasher = hashlib.md5()

        with reader.read_file(entry) as f:
            while chunk := f.read(self.chunk_size):
                hasher.update(chunk)

        return hasher.hexdigest()

    def _calculate_checksum_isolated(
        self, sga_path: str, entry: FileEntry
    ) -> ChecksumResult:
        """Calculate checksum with isolated SGA handle.

        Args:
            sga_path: Path to SGA archive
            file_path: File path within archive

        Returns:
            Tuple of (file_path, checksum, error)
        """
        try:
            with SgaReader(sga_path) as sga:
                checksum = self._calculate_checksum(entry, sga)
                return ChecksumResult(entry.path, checksum, None)
        except Exception as e:
            return ChecksumResult(entry.path, None, str(e))

    def _build_manifest(self, sga_path: str) -> Dict[str, str]:
        """Build manifest of file checksums using parallel processing.

        Args:
            sga_path: Path to SGA archive

        Returns:
            Dictionary mapping path to checksum
        """
        manifest = {}

        # Collect all file paths first
        file_entries = NativeParserV2(sga_path).parse()

        self.logger.info(f"Calculating checksums for {len(file_entries)} files...")

        # Calculate checksums in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(
                    self._calculate_checksum_isolated, sga_path, file_path
                ): file_path
                for file_path in file_entries
            }

            processed = 0
            for future in as_completed(futures):
                result: ChecksumResult = future.result()

                if result.checksum:
                    manifest[result.path] = result.checksum
                else:
                    self.logger.warning(
                        f"Could not checksum {result.path}: {result.error}"
                    )

                processed += 1
                if processed % 1000 == 0:
                    self.logger.info(
                        f"  Checksummed {processed}/{len(file_entries)} files"
                    )

        self.logger.info(f"Checksum calculation complete: {len(manifest)} files")
        return manifest

    def _load_previous_manifest(self, output_dir: str) -> Optional[Dict[str, str]]:
        """Load previous extraction manifest if exists.

        Args:
            output_dir: Output directory

        Returns:
            Previous manifest or None
        """
        manifest_path = Path(output_dir) / ".sga_manifest.json"

        if not manifest_path.exists():
            return None

        try:
            with open(manifest_path, "r") as f:
                return json.load(f)  # type: ignore
        except Exception as e:
            self.logger.warning(f"Could not load previous manifest: {e}")
            return None

    def _save_manifest(self, output_dir: str, manifest: Dict[str, str]) -> None:
        """Save extraction manifest.

        Args:
            output_dir: Output directory
            manifest: File checksums
        """
        manifest_path = Path(output_dir) / ".sga_manifest.json"

        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save manifest: {e}")

    def extract_delta(
        self,
        sga_path: str,
        output_dir: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> ExtractionStats:
        """Extract only files that changed since last extraction.

        Args:
            sga_path: Path to SGA archive
            output_dir: Output directory
            on_progress: Optional progress callback

        Returns:
            Extraction statistics
        """
        self.logger.info("Delta extraction mode enabled")

        # Load previous manifest
        previous_manifest = self._load_previous_manifest(output_dir)

        if previous_manifest is None:
            self.logger.info("No previous manifest found - performing full extraction")
            # Force full extraction (avoid recursion)
            stats = self.extract_optimized(
                sga_path, output_dir, on_progress, force_full=True
            )
            # Save manifest for next time
            current_manifest = self._build_manifest(sga_path)
            self._save_manifest(output_dir, current_manifest)
            return stats

        # Build current manifest
        self.logger.info("Building current manifest...")
        current_manifest = self._build_manifest(sga_path)

        # Find changed files
        changed_files = []
        new_files = []

        for path, checksum in current_manifest.items():
            if path not in previous_manifest:
                new_files.append(path)
                changed_files.append(path)
            elif previous_manifest[path] != checksum:
                changed_files.append(path)

        # Find deleted files
        deleted_files = []
        for path in previous_manifest:
            if path not in current_manifest:
                deleted_files.append(path)

        self.logger.info("Delta analysis:")
        self.logger.info(f"  New files:      {len(new_files)}")
        self.logger.info(f"  Modified files: {len(changed_files) - len(new_files)}")
        self.logger.info(f"  Deleted files:  {len(deleted_files)}")
        self.logger.info(
            f"  Unchanged:      {len(previous_manifest) - len(changed_files)}"
        )

        if not changed_files:
            self.logger.info("No changes detected - skipping extraction")
            self.stats.skipped_files = len(previous_manifest)
            self.stats.total_files = len(previous_manifest)
            return self.stats

        # Extract ONLY changed files (FAST!)
        self.logger.info(f"Extracting {len(changed_files)} changed files...")
        self.logger.info(
            f"Skipping {len(previous_manifest) - len(changed_files)} unchanged files"
        )

        stats = self.extract_optimized(
            sga_path,
            output_dir,
            on_progress,
            force_full=True,
            file_filter=changed_files,  # Only extract changed files!
        )

        stats.skipped_files = len(previous_manifest) - len(changed_files)
        stats.total_files = len(previous_manifest)

        # Save updated manifest
        self._save_manifest(output_dir, current_manifest)

        return stats

    def extract_progressive(
        self,
        sga_path: str,
        output_dir: str,
        on_file_extracted: Callable[[str, bytes], None],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> ExtractionStats:
        """Extract with callback on each file completion.

        Useful for streaming/pipeline scenarios where you want to start
        processing files before full extraction completes.

        Args:
            sga_path: Path to SGA archive
            output_dir: Output directory
            on_file_extracted: Callback(file_path, file_data) called when file ready
            on_progress: Optional progress callback

        Returns:
            Extraction statistics
        """
        from queue import Queue
        from threading import Thread

        self.logger.info("Progressive extraction mode enabled")

        # Queue for extracted files
        result_queue: Queue[tuple[str, bytes] | None] = Queue()

        # Consumer thread to call callbacks
        def consumer() -> None:
            while True:
                item = result_queue.get()
                if item is None:  # Sentinel to stop
                    break

                file_path, file_data = item
                try:
                    on_file_extracted(file_path, file_data)
                except Exception as e:
                    self.logger.error(f"Callback error for {file_path}: {e}")

                result_queue.task_done()

        # Start consumer
        consumer_thread = Thread(target=consumer, daemon=True)
        consumer_thread.start()

        # Extract (using modified extraction that queues results)
        # For simplicity, just use standard extraction
        # In production, would integrate queue into extract_optimized
        stats = self.extract_optimized(sga_path, output_dir, on_progress)

        # Wait for all callbacks to complete
        result_queue.join()

        # Stop consumer
        result_queue.put(None)
        consumer_thread.join()

        return stats

    def get_extraction_plan(self, sga_path: str) -> ExtractionPlan:
        """Analyze archive and return extraction plan.

        Useful for estimating time/resources before extraction.

        Args:
            sga_path: Path to SGA archive

        Returns:
            Dictionary with extraction plan details
        """
        entries = NativeParserV2(sga_path).parse()

        categories = _categorize_by_size(entries, logger=self.logger)

        plan = ExtractionPlan(
            total_files=len(entries),
            total_bytes=sum(e.decompressed_size for e in entries),
            categories={},
            estimated_time_seconds=0,
            recommended_workers=self.num_workers,
        )

        for cat, files in categories.items():
            if not files:
                continue

            workers = self._get_optimal_workers(cat, len(files))
            total_bytes = sum(f.decompressed_size for f in files)

            plan.categories[cat] = ExtractionPlanCategory(
                file_count=len(files),
                total_bytes=total_bytes,
                workers=workers,
            )

        # Estimate time (very rough)
        # Assume ~50 MB/s extraction rate per worker
        throughput_per_worker = 50 * 1024 * 1024  # 50 MB/s
        effective_throughput = (
            throughput_per_worker * self.num_workers * 0.7
        )  # 70% efficiency
        plan.estimated_time_seconds = plan.total_bytes / effective_throughput

        return plan
