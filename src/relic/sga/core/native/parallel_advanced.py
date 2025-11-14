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

import datetime
import gc
import hashlib
import json
import logging
import math
import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed, Future
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Any,
    Generator,
    Sequence,
    TypeAlias,
)

from relic.core.logmsg import BraceMessage

from relic.sga.core.definitions import OSFlags
from relic.sga.core.native.definitions import (
    FileEntry,
    ExtractionStats,
    ReadResult,
    BatchResult,
    Result,
)
from relic.sga.core.native.handler import SgaReader, parser_registry

ProgressCallback: TypeAlias = Callable[[int, int], None]


# @no_type_check
class FakeLogger:
    def __getattr__(self, name: Any) -> Any:
        def faker(*args: Any, **kwargs: Any) -> Any:
            return self

        return faker


class _ExtractStrategy:
    def __init__(self, config: UnpackerConfig, cache: DirectoryCacher):
        self.stats: ExtractionStats = None  # type: ignore
        self.directories = cache

        self.logger = config.logger or logging.getLogger(__name__)
        self.verbose_logger = self.logger if config.verbose else FakeLogger()
        self._precreate_dirs = config.precache_dirs
        self.num_workers = config.num_workers
        if self.num_workers <= 0:
            self.logger.error(
                f"# of workers ({config.num_workers}) invalid, defaulting to 1"
            )
            self.num_workers = 1

        self.merging = config.should_merge
        self.chunk_size = config.chunk_size
        self._should_disable_gc = config.disable_gc
        self.batch_size = config.batch_size
        self.native_handles = config.native_files

    def _write(
        self,
        output_dir: str,
        path: str,
        file_data: bytes,
        modified: datetime.datetime | None,
    ) -> None:
        dst_path = Path(output_dir) / path.lstrip("/")
        self.directories.ensure_directory(dst_path.parent)
        if self.native_handles:
            with dst_path.open("wb") as dst:
                dst.write(file_data)
        else:
            fd = os.open(
                dst_path,
                OSFlags.O_CREAT | OSFlags.O_WRONLY | OSFlags.O_BINARY | OSFlags.O_TRUNC,
            )
            os.write(fd, file_data)
            os.close(fd)
        if modified is not None:
            os.utime(dst_path, (modified.timestamp(), modified.timestamp()))

    def extract(
        self,
        sga_path: str,
        output_dir: str,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs: Any,
    ) -> ExtractionStats:
        raise NotImplementedError(
            f"{self.__class__.__qualname__}.extract(...) is not implemented"
        )

    def _start(self, name: str, sga_path: str) -> None:
        self.verbose_logger.info(
            BraceMessage(
                "Starting {name} extraction: {sga_path}", name=name, sga_path=sga_path
            )
        )
        self.stats = ExtractionStats()

    def _collect(
        self, sga_path: str, file_filter: Optional[Sequence[str]] = None
    ) -> Sequence[FileEntry]:
        self.verbose_logger.info("Collecting file paths...")
        with self._timer() as timer:
            with parser_registry.create_parser(sga_path) as versioned_parser:
                entries = versioned_parser.parse()
            self.stats.timings.parsing_sga = timer()

        # Apply filter if provided (DELTA MODE!)
        if file_filter is not None:
            with self._timer() as timer:
                file_filter_set = set(file_filter)
                entries = [
                    p
                    for p in entries
                    if p.full_path(not self.merging) in file_filter_set
                ]
                self.stats.timings.filtering_files = timer()
            self.verbose_logger.info(f"Filtered to {len(entries)} changed files")

        self.stats.total_files = len(entries)
        self.verbose_logger.info(f"Found {len(entries)} files")
        return entries

    def _create_dirs(
        self,
        entries: Sequence[FileEntry],
        output_dir: str,
        force: bool = False,
    ) -> None:
        if not force and not self._precreate_dirs:
            return
        self.verbose_logger.info("Pre-creating directory structure...")
        with self._timer() as timer:
            created_dirs = 0
            for file in entries:
                dst_path = Path(output_dir) / file.full_path(
                    include_drive=not self.merging
                ).lstrip("/")
                if self.directories.ensure_directory(dst_path.parent):
                    created_dirs += 1
            self.stats.timings.creating_dirs = timer()
        self.verbose_logger.info(f"Created {created_dirs} directories")

    def _create_batches_parallel(
        self, entries: Sequence[FileEntry], batch_size: int
    ) -> List[tuple[int, int]]:
        entry_count = len(entries)

        def _batch(batch_id: int) -> Sequence[FileEntry]:
            batch_start = batch_id * batch_size
            batch_end = min(batch_start + batch_size, entry_count)
            return batch_start, batch_end

        with self._timer() as timer:
            batch_count, remaining = divmod(entry_count, batch_size)
            batch_ids = list(range(batch_size + (1 if remaining > 0 else 0)))
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                batches: List[Result[FileEntry, bool]] = list(
                    executor.map(_batch, batch_ids)
                )
            self.stats.timings.creating_batches = timer()
        return batches

    def _create_batches_serial(
        self, entries: Sequence[FileEntry], batch_size: int
    ) -> List[Sequence[FileEntry]]:
        batches = []
        with self._timer() as timer:
            batch_count, remaining = divmod(len(entries), batch_size)

            for i in range(batch_count):
                ptr = i * batch_size, (i + 1) * batch_size
                batches.append(ptr)
            if remaining > 0:
                offset = batch_count * batch_size
                ptr = offset, offset + remaining
                batches.append(ptr)

                # batches.append(entries[offset : offset + remaining])
            self.stats.timings.creating_batches = timer()
        return batches

    def _create_batches(
        self, entries: Sequence[FileEntry], batch_size: int
    ) -> List[tuple[int, int]]:
        self.verbose_logger.info(f"Extracting {len(entries)} files...")
        # Dont bother with stats if batching is skipped
        if batch_size == 0:
            self.verbose_logger.info("Batching not enabled - skipping batching...")
            return [entries]
        if batch_size < 0:
            self.logger.error(
                f"Invalid batch size '{batch_size}' - skipping batching..."
            )
            return [entries]

        self.verbose_logger.info(f"Creating batches of '{batch_size}'")
        batches = self._create_batches_parallel(entries, batch_size)
        self.verbose_logger.info(f"Created {len(batches)} batches")
        return batches

        # Create batches

    # def _extract_isolated(self, sga_path:str, output_dir:str, entry:FileEntry, merging:bool=False) -> Result[FileEntry,bool]:

    def _extract_batch_isolated(
        self,
        sga_path: str,
        output_dir: str,
        entries: Sequence[FileEntry],
        batch: tuple[int, int],
    ) -> List[Result[FileEntry, bool]]:
        """Extract batch of files with single SGA handle using NATIVE Python file operations.

        Uses native Python Path/open for output (FAST on Windows!) and fs library
        only for reading from SGA (necessary).

        Args:
            sga_path: Path to SGA archive
            output_dir: Output directory
            batch: List of file entries to extract

        Returns:
            List of (success, path, error) tuples
        """
        results: List[Result[FileEntry, bool]] = []

        last_entry_completed = -1  # sentinel; -1 signifies no files completed
        try:
            # Open SGA once for entire batch
            with SgaReader(sga_path) as my_sga:
                for i in range(*batch):
                    entry = entries[i]
                    entry_path = entry.full_path(include_drive=not self.merging)
                    try:
                        # Get native destination path
                        native_dst = Path(output_dir) / entry_path.lstrip("/")

                        # # Create parent directory (CACHED for speed!)
                        self.directories.ensure_directory(native_dst.parent)

                        # # Extract with chunked reading (use NATIVE Python file for writing!)
                        # # For small files, use smaller chunks (64KB) for better responsiveness
                        # small_chunk = max(
                        #     64 * 1024,
                        # )  # 64KB - perfect for small files
                        if self.native_handles:
                            with my_sga.open_file(entry) as src_file:
                                with open(
                                    native_dst, "wb", buffering=self.chunk_size
                                ) as dst_file:
                                    while True:
                                        chunk = src_file.read(self.chunk_size)
                                        if not chunk:
                                            break
                                        dst_file.write(chunk)

                                    dst_file.flush()
                        else:
                            data = my_sga.read_file(entry)

                            fd = os.open(
                                native_dst,
                                OSFlags.O_CREAT | OSFlags.O_WRONLY | OSFlags.O_BINARY,
                            )
                            if entry.modified:
                                os.utime(
                                    native_dst,
                                    (
                                        entry.modified.timestamp(),
                                        entry.modified.timestamp(),
                                    ),
                                )
                            os.write(fd, data)
                            os.close(fd)

                        results.append(Result(entry, True))
                        last_entry_completed = i

                    except Exception as e:
                        results.append(Result(entry, False, [e]))

        except Exception as e:
            # Batch failed entirely
            for i in range(*batch):
                if i <= last_entry_completed:
                    continue
                results.append(Result(entries[i], False, [e]))

        return results

    def _execute_batches(
        self,
        entries: Sequence[FileEntry],
        batches: Sequence[tuple[int, int]],
        executor: ThreadPoolExecutor,
        sga_path: str,
        output_dir: str,
    ) -> list[Future[list[Result[FileEntry, bool]]]]:
        futures = []
        for batch in batches:
            future: Future[list[Result[FileEntry, bool]]] = executor.submit(
                self._extract_batch_isolated,
                sga_path,
                output_dir,
                entries,
                batch,
            )
            futures.append(future)
        return futures

    def _parse_batch_results(
        self,
        futures: list[Future[list[Result[FileEntry, bool]]]],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        with self._timer() as timer:
            total_processed = 0
            for future in as_completed(futures):
                results = future.result()

                for result in results:
                    if result.output:
                        self.stats.extracted_files += 1
                        total_processed += 1
                    else:
                        self.stats.failed_files += 1
                        total_processed += 1
                        if self.stats.failed_files <= 10:  # Only show first 10 errors
                            for error in result.errors:
                                self.logger.error(
                                    f"Failed: {result.input.full_path(not self.merging)}: {error}"
                                )

                    if on_progress:
                        on_progress(total_processed, self.stats.total_files)

            # Final progress callback
            if on_progress:
                on_progress(total_processed, self.stats.total_files)
            self.stats.timings.executing_batches = timer()

    def _print_summary(self) -> None:
        self.logger.info("Extraction complete!")
        self.logger.info(f"  Total:      {self.stats.total_files}")
        self.logger.info(f"  Successful: {self.stats.extracted_files}")
        self.logger.info(f"  Failed:     {self.stats.failed_files}")
        self.logger.info(
            f"  Extracted:  {self.stats.extracted_bytes / 1024 / 1024:.1f} MB"
        )
        self.logger.info("Timings")
        self.logger.info(
            f"  Parsing SGA:          {self.stats.timings.parsing_sga:.4f}"
        )
        if self.stats.timings.filtering_files > 0:
            self.logger.info(
                f"  Filtering SGA:        {self.stats.timings.filtering_files:.4f}"
            )
        if self.stats.timings.creating_dirs > 0:
            self.logger.info(
                f"  Creating directories: {self.stats.timings.creating_dirs:.4f}"
            )
        if self.stats.timings.creating_batches > 0:
            self.logger.info(
                f"  Creating Batches:     {self.stats.timings.creating_batches:.4f}"
            )
        self.logger.info(
            f"  Running Batches:      {self.stats.timings.executing_batches:.4f}"
        )
        if self.stats.timings.parsing_results > 0:
            self.logger.info(
                f"  Parsing Results:      {self.stats.timings.parsing_results:.4f}"
            )
        self.logger.info(f"  Total Time:           {self.stats.timings.total_time:.4f}")

    @contextmanager
    def _disable_gc(self) -> Generator[None, Any, None]:
        if self._should_disable_gc:
            gc_was_enabled = gc.isenabled()
            if gc_was_enabled:
                gc.disable()
                self.logger.debug("Disabled GC")
            yield
            if gc_was_enabled:
                gc.enable()
                self.logger.debug("Re-enabled GC")
                gc.collect()
        else:
            yield

    @contextmanager
    def _timer(self) -> Generator[Callable[[], float], Any, None]:
        import time as time_module

        t0 = time_module.perf_counter()

        def delta() -> float:
            return time_module.perf_counter() - t0

        yield delta


class OptimizedStrategy(_ExtractStrategy):
    def extract(
        self,
        sga_path: str,
        output_dir: str,
        on_progress: Optional[ProgressCallback] = None,
        *,
        file_filter: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ExtractionStats:
        """Extract with all optimizations enabled.

        Args:
            sga_path: Path to SGA archive
            output_dir: Output directory
            on_progress: Optional callback (current, total)
            file_filter: Optional list of file paths to extract (None = extract all)

        Returns:
            Extraction statistics
        """

        self._start("OPTIMIZED", sga_path)
        # FAST MODE: Skip metadata collection, just get file paths!
        entries = self._collect(sga_path, file_filter)
        # # PRE-CREATE ALL DIRECTORIES (avoid per-file checks!)
        self._create_dirs(entries, output_dir)
        # Create batches
        with self._disable_gc():
            self.logger.info(f"Using {self.num_workers} workers")
            batches = self._create_batches(entries, self.batch_size)

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = self._execute_batches(
                    entries, batches, executor, sga_path, output_dir
                )
                self._parse_batch_results(futures, on_progress)
        # Summary
        self._print_summary()

        return self.stats


class StreamingStrategy(_ExtractStrategy):
    def extract(
        self,
        sga_path: str,
        output_dir: str,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs: Any,
    ) -> ExtractionStats:
        """PHASE 2: Multi-reader streaming with thread-local SGA handles.

        Architecture:
        - MULTIPLE READERS: Each with thread-local SGA handle (no lock contention!)
        - Uses NativeSGAReader for thread-safe parallel reading
        - Massive memory buffer for smooth flow
        - All optimizations from Phase 1

        Expected: 2-3x faster than single-reader streaming!
        """
        self._start("STREAMING", sga_path)

        # Get file list using native reader
        entries = self._collect(sga_path)

        self._create_dirs(entries, output_dir)

        with self._disable_gc():

            def read(entry: FileEntry) -> Result[FileEntry, bytes]:
                try:
                    with SgaReader(sga_path) as reader:
                        data = reader.read_file(entry)
                        return Result(entry, data)
                except Exception as e:
                    return Result(entry, False, [e])

            def write(entry: FileEntry, data: bytes) -> Result[FileEntry, bool]:
                try:
                    self._write(
                        output_dir,
                        entry.full_path(not self.merging),
                        data,
                        entry.modified,
                    )
                    return Result(entry, True)
                except Exception as e:
                    return Result(entry, False, [e])

            with self._timer() as timer:
                self.stats.total_files = len(entries)

                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    read_futures = [executor.submit(read, entry) for entry in entries]
                    write_futures = []
                    read_fails = write_fails = 0
                    for future in as_completed(read_futures):
                        result: Result[FileEntry, bytes] = future.result()
                        self.stats.total_bytes += result.input.decompressed_size
                        if result.has_errors:
                            self.stats.failed_files += 1
                            if read_fails < 10:
                                read_fails += 1
                                for error in result.errors:
                                    self.logger.error("Read Error: %s", error)
                        else:
                            write_futures.append(
                                executor.submit(write, result.input, result.output)
                            )
                    for future in as_completed(write_futures):
                        result = future.result()
                        if result.has_errors or not result.output:
                            self.stats.failed_files += 1
                            if write_fails < 10:
                                write_fails += 1
                                for error in result.errors:
                                    self.logger.error("Write Error: %s", error)
                        else:
                            self.stats.extracted_bytes += result.input.decompressed_size
                            self.stats.extracted_files += 1

                    self.stats.timings.executing_batches = timer()

        # Summary
        self._print_summary()

        return self.stats


class NativeStrategy(_ExtractStrategy):
    def extract(
        self,
        sga_path: str,
        output_dir: str,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs: Any,
    ) -> ExtractionStats:
        """NATIVE ULTRA-FAST extraction - 2-3 seconds target!

        Uses true native binary parser + parallel decompression + parallel writes.
        Bypasses fs library completely for maximum speed.

        Target: 2-3 seconds for 7,815 files!
        """

        self._start("NATIVE", sga_path)

        files = self._collect(sga_path)
        self._create_dirs(files, output_dir)

        with self._disable_gc():
            self.verbose_logger.info(
                f"Reading + decompressing {len(files)} files (parallel)..."
            )
            with self._timer() as timer:
                with SgaReader(sga_path) as parser:
                    buffers = parser.read_files_parallel(
                        files, num_workers=self.num_workers
                    )

                self.stats.timings.creating_batches = timer()
            self.verbose_logger.info(
                f"Read + decompressed in {self.stats.timings.creating_batches:.2f}s"
            )

            self.verbose_logger.info("Writing files to disk (parallel)...")

            def write_file(
                item: Result[FileEntry, bytes],
            ) -> Result[FileEntry, bool]:
                if item.has_errors:
                    return Result(item.input, False, item.errors)
                if item.output is None:
                    return Result(item.input, False, ["Data was None"])
                path = item.input.full_path(not self.merging)
                file_data, modified = item.output, item.input.modified
                try:
                    self._write(output_dir, path, file_data, modified)
                    return Result(item.input, True)
                except Exception as e:
                    return Result(item.input, False, [e])

            with self._timer() as timer:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    write_results: List[Result[FileEntry, bool]] = list(
                        executor.map(write_file, buffers)
                    )
                self.stats.timings.executing_batches = timer()

            # Count results
            with self._timer() as timer:
                for write_result in write_results:
                    if write_result.output:
                        self.stats.extracted_files += 1
                        self.stats.extracted_bytes += (
                            write_result.input.decompressed_size
                        )
                    else:
                        self.stats.failed_files += 1
                        for error in write_result.errors:
                            self.logger.error(
                                f"Failed to write {write_result.input.full_path(not self.merging)}: {error}"
                            )
                self.stats.timings.parsing_results = timer()

        # Summary
        self._print_summary()

        return self.stats


@dataclass(slots=True)
class UnpackerConfig:

    num_workers: int
    logger: Optional[logging.Logger] = None
    # enable_adaptive_threading: bool = True # todo; bring back for grouping batches by category?
    # enable_batching: bool = True # batching always enabled; use a non positive value (0 or any negative) to disable
    # enable_delta: bool = False # moved to its own Strategy; explicitly call extract with delta
    chunk_size: int = 1024 * 1024
    disable_gc: bool = True
    batch_size: int = 128  # arbitrary value
    native_files: bool = False
    precache_dirs: bool = True
    verbose: bool = False
    should_merge: bool = False


class DirectoryCacher:
    # Moved to its own class to allow strategies to reference the same cacher
    def __init__(self) -> None:
        # Thread-safe directory cache to avoid redundant mkdir calls
        self._dir_cache: Set[str] = set()
        self._dir_cache_lock = threading.Lock()

    def ensure_directory(self, dir_path: Path) -> bool:
        """Ensure directory exists with caching to avoid redundant mkdir calls.


        Args:
            dir_path: Directory path to create

        Returns:
            bool: True if directory was created, False otherwise
        """
        CREATED = True
        dir_str = str(dir_path)

        # Fast path: check if already created
        if dir_str in self._dir_cache:
            return not CREATED

        # Slow path: create and cache
        with self._dir_cache_lock:
            # Double-check after acquiring lock
            if dir_str not in self._dir_cache:
                dir_path.mkdir(parents=True, exist_ok=True)
                self._dir_cache.add(dir_str)
                for parent_path in dir_path.parents:
                    self._dir_cache.add(str(parent_path))
                return CREATED

        return not CREATED


class ProgressiveStrategy(_ExtractStrategy):
    def __init__(
        self,
        config: UnpackerConfig,
        cache: DirectoryCacher,
        optimized: OptimizedStrategy | None = None,
    ):
        super().__init__(config, cache)
        # We create it using the same config/cache, so it should be the same even if optimized is not given
        self._optimized = optimized or OptimizedStrategy(config, cache)

    def extract(
        self,
        sga_path: str,
        output_dir: str,
        on_progress: Optional[ProgressCallback] = None,
        on_file_extracted: Callable[[str, bytes], None] | None = None,
        **kwargs: Any,
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
        if on_file_extracted is None:
            raise ValueError("on_file_extracted was not provided")  # TODO better error
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
        stats = self._optimized.extract(sga_path, output_dir, on_progress)

        # Wait for all callbacks to complete
        result_queue.join()

        # Stop consumer
        result_queue.put(None)
        consumer_thread.join()

        return stats


class DeltaStrategy(_ExtractStrategy):
    def __init__(
        self,
        config: UnpackerConfig,
        cache: DirectoryCacher,
        optimized: OptimizedStrategy | None = None,
    ):
        super().__init__(config, cache)
        # We create it using the same config/cache, so it should be the same even if optimized is not given
        self._optimized = optimized or OptimizedStrategy(config, cache)

    def _calculate_checksum(self, entry: FileEntry, reader: SgaReader) -> str:
        """Calculate MD5 checksum for a file.

        Args:
            file_path: Path to file
            fs: Filesystem containing file

        Returns:
            MD5 checksum (hex)
        """
        hasher = hashlib.md5()

        with reader.open_file(entry) as f:
            while chunk := f.read(self.chunk_size):
                hasher.update(chunk)

        return hasher.hexdigest()

    def _calculate_checksum_isolated(
        self, sga_path: str, entry: FileEntry
    ) -> Result[FileEntry, str]:
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
                return Result(entry, checksum)
        except Exception as e:
            return Result.create_error(entry, e)

    def _build_manifest(self, sga_path: str) -> Dict[str, str]:
        """Build manifest of file checksums using parallel processing.

        Args:
            sga_path: Path to SGA archive

        Returns:
            Dictionary mapping path to checksum
        """
        manifest = {}

        # Collect all file paths first
        with parser_registry.create_parser(sga_path) as parser:
            file_entries = parser.parse()
            drive_count = parser.get_drive_count()
            merging = (
                self.should_merge
                if isinstance(self.should_merge, bool)
                else self.should_merge(drive_count)
            )

        self.logger.info(f"Calculating checksums for {len(file_entries)} files...")
        num_workers = self.num_workers or 1
        # Calculate checksums in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self._calculate_checksum_isolated, sga_path, file_path)
                for file_path in file_entries
            ]

            processed = 0
            for future in as_completed(futures):
                result: Result[FileEntry, str] = future.result()

                if result.output:
                    manifest[result.input.full_path(not self.merging)] = result.output
                else:
                    for i, error in enumerate(result.errors):
                        self.logger.error(
                            f"Could not checksum {result.input.full_path(not self.merging)}: {error}"
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
            with open(manifest_path, "r", encoding="utf8") as f:
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
            with open(manifest_path, "w", encoding="utf8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save manifest: {e}")

    def extract(
        self,
        sga_path: str,
        output_dir: str,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs: Any,
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
        self.stats = ExtractionStats()
        # Load previous manifest
        previous_manifest = self._load_previous_manifest(output_dir)

        if previous_manifest is None:
            self.logger.info("No previous manifest found - performing full extraction")
            # Force full extraction (avoid recursion)
            stats = self._optimized.extract(
                sga_path,
                output_dir,
                on_progress,
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

        stats = self._optimized.extract(
            sga_path,
            output_dir,
            on_progress,
            file_filter=changed_files,  # Only extract changed files!
        )

        stats.skipped_files = len(previous_manifest) - len(changed_files)
        stats.total_files = len(previous_manifest)

        # Save updated manifest
        self._save_manifest(output_dir, current_manifest)

        return stats


class SerialStrategy(_ExtractStrategy):
    def extract(
        self,
        sga_path: str,
        output_dir: str,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs: Any,
    ) -> ExtractionStats:

        self._start("Serial", sga_path)

        # Get file list
        files = self._collect(sga_path)
        self._create_dirs(files, output_dir)

        self.stats.total_files = len(files)
        processed = 0
        with self._disable_gc():
            with SgaReader(sga_path) as sga:
                with self._timer() as timer:
                    for processed, file in enumerate(files):
                        try:
                            # Read and decompress (fs handles this)
                            data = sga.read_file(file)

                            # Write to disk

                            self._write(
                                output_dir,
                                file.full_path(not self.merging),
                                data,
                                file.modified,
                            )

                            self.stats.extracted_files += 1
                            self.stats.extracted_bytes += len(data)

                        except Exception as e:
                            self.stats.failed_files += 1
                            if self.stats.failed_files <= 3:
                                self.logger.error(
                                    f"Failed {file.full_path(not self.merging)}: {e}"
                                )

                            # if on_progress and processed[0] % 500 == 0: # TODO fix the mod operation
                            if on_progress and processed % 50 == 0:
                                on_progress(processed, self.stats.total_files)

                    self.stats.timings.executing_batches = timer()

            # Final progress
            if on_progress:
                on_progress(processed, self.stats.total_files)

        # Summary
        self._print_summary()
        # self.logger.info(f"  Time:       {t_extract:.2f}s")
        # self.logger.info(
        #     f"  Throughput: {self.stats.extracted_bytes / t_extract / 1024 / 1024:.0f} MB/s"
        # )
        return self.stats


class ExtractionMethod(IntEnum):
    """
    Extraction Strategy to use;

    Names and enum values do not correlate to performance.
    Generally; use UltraFast or Native for best performance.
    """

    SERIAL = 5
    ULTRA_FAST = 0  # Rolled into optimized via config parameters
    OPTIMIZED = 1
    STREAMING = 2
    NATIVE = 3
    DELTA = 4  # untested;


class AdvancedParallelUnpacker:
    """Advanced parallel unpacker with comprehensive optimizations."""

    def __init__(self, config: UnpackerConfig):
        """Initialize advanced parallel unpacker.

        Args:
            num_workers: Number of worker threads (None = system thread count)
            logger: Logger instance
            enable_adaptive_threading: Adjust workers by file size
            enable_batching: Batch tiny files together
            enable_delta: Enable delta extraction (default: False)
            chunk_size: Read chunk size in bytes
        """
        self._config = config
        self._cache = DirectoryCacher()
        # define strategies as local variables to reuse them IF needed
        _optimized = OptimizedStrategy(self._config, self._cache)
        _streaming = StreamingStrategy(self._config, self._cache)
        _native = NativeStrategy(self._config, self._cache)
        _delta = DeltaStrategy(self._config, self._cache, _optimized)
        _serial = SerialStrategy(self._config, self._cache)
        self._strategies = {
            ExtractionMethod.OPTIMIZED: _optimized,
            ExtractionMethod.STREAMING: _streaming,
            ExtractionMethod.ULTRA_FAST: _optimized,  # use optimized; same thing
            ExtractionMethod.NATIVE: _native,
            ExtractionMethod.DELTA: _delta,
            ExtractionMethod.SERIAL: _serial,
        }
        self._default_extractor = _native

    def extract(
        self,
        sga_path: str,
        output_dir: str,
        on_progress: Callable[[int, int], None] | None = None,
        method: ExtractionMethod = ExtractionMethod.STREAMING,  # Based on personal testing
    ) -> ExtractionStats:
        extractor = self._strategies.get(method, self._default_extractor)
        stats = extractor.extract(sga_path, output_dir, on_progress)
        return stats
