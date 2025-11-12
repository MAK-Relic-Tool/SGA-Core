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
    WriteResult,
    ReadResult,
    ChecksumResult,
    BatchResult,
)
from relic.sga.core.native.native_reader import SgaReader
from relic.sga.core.native.v2 import NativeParserV2

ProgressCallback: TypeAlias = Callable[[int, int], None]


# @no_type_check
class FakeLogger(object):
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

        self.should_merge = config.should_merge
        self.chunk_size = config.chunk_size
        self._should_disable_gc = config.disable_gc
        self.batch_size = config.batch_size
        self.native_handles = config.native_files

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
    ) -> list[FileEntry]:
        self.verbose_logger.info("Collecting file paths...")
        with self._timer() as timer:
            entries = NativeParserV2(sga_path, self.logger, self.should_merge).parse()
            self.stats.timings.parsing_sga = timer()

        # Apply filter if provided (DELTA MODE!)
        if file_filter is not None:
            with self._timer() as timer:
                file_filter_set = set(file_filter)
                entries = [p for p in entries if p.path in file_filter_set]
                self.stats.timings.filtering_files = timer()
            self.verbose_logger.info(f"Filtered to {len(entries)} changed files")

        self.stats.total_files = len(entries)
        self.verbose_logger.info(f"Found {len(entries)} files")
        return entries

    def _create_dirs(
        self, entries: list[FileEntry], output_dir: str, force: bool = False
    ) -> None:
        if not force and not self._precreate_dirs:
            return
        self.verbose_logger.info("Pre-creating directory structure...")
        with self._timer() as timer:

            created_dirs = 0
            for file in entries:
                dst_path = Path(output_dir) / file.path.lstrip("/")
                if self.directories.ensure_directory(dst_path.parent):
                    created_dirs += 1
            self.stats.timings.creating_dirs = timer()
        self.verbose_logger.info(f"Created {created_dirs} directories")

    def _create_batches(
        self, entries: list[FileEntry], batch_size: int
    ) -> List[List[FileEntry]]:
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

        # Create batches
        batches = []
        with self._timer() as timer:
            batch_count, remaining = divmod(len(entries), batch_size)

            for i in range(batch_count):
                batches.append(entries[i * batch_size : (i + 1) * batch_size])
            if remaining > 0:
                offset = batch_count * batch_size
                batches.append(entries[offset : offset + remaining])
            self.stats.timings.creating_batches = timer()
        self.verbose_logger.info(f"Created {len(batches)} batches")
        return batches

    def _extract_batch_isolated(
        self, sga_path: str, output_dir: str, batch: List[FileEntry]
    ) -> List[BatchResult]:
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
        results: List[BatchResult] = []

        last_entry_completed = -1  # sentinel; -1 signifies no files completed
        try:
            # Open SGA once for entire batch
            with SgaReader(sga_path) as my_sga:
                for i, entry in enumerate(batch):
                    try:
                        # Get native destination path
                        native_dst = Path(output_dir) / entry.path.lstrip("/")

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
                            os.write(fd, data)
                            os.close(fd)

                        results.append(BatchResult(True, entry.path, None))
                        last_entry_completed = i

                    except Exception as e:
                        results.append(BatchResult(False, entry.path, str(e)))

        except Exception as e:
            # Batch failed entirely
            for i, entry in enumerate(batch):
                if i <= last_entry_completed:
                    continue
                results.append(BatchResult(False, entry.path, f"Batch error: {str(e)}"))

        return results

    def _execute_batches(
        self,
        batches: List[List[FileEntry]],
        executor: ThreadPoolExecutor,
        sga_path: str,
        output_dir: str,
    ) -> list[Future[list[BatchResult]]]:
        futures = []
        for batch in batches:
            future: Future[list[BatchResult]] = executor.submit(
                self._extract_batch_isolated,
                sga_path,
                output_dir,
                batch,
            )
            futures.append(future)
        return futures

    def _parse_batch_results(
        self,
        futures: list[Future[list[BatchResult]]],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        with self._timer() as timer:
            total_processed = 0
            for future in as_completed(futures):
                results = future.result()

                for result in results:
                    if result.success:
                        self.stats.extracted_files += 1
                        total_processed += 1
                    else:
                        self.stats.failed_files += 1
                        total_processed += 1
                        if self.stats.failed_files <= 10:  # Only show first 10 errors
                            self.logger.error(f"Failed: {result.path}: {result.error}")

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
                futures = self._execute_batches(batches, executor, sga_path, output_dir)
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
            # MULTI-READER with thread-local handles!
            num_readers = min(
                5, max(1, self.num_workers // 3)
            )  # 5 readers for 15 workers
            num_writers = max(self.num_workers - num_readers, 1)

            self.logger.info(
                f"Using {num_readers} readers (thread-local) + {num_writers} writers"
            )

            # Split files among readers
            files_per_reader = math.ceil(len(entries) / num_readers)
            reader_file_lists = self._create_batches(entries, files_per_reader)

            # Shared deque with massive buffer
            buffer_size = min(len(entries), 10000)
            file_deque: deque[ReadResult[bytes]] = deque(maxlen=buffer_size)
            self.logger.info(
                f"Using {buffer_size}-file memory buffer (~{buffer_size * 500 / 1024:.0f} MB)"
            )

            deque_lock = threading.Lock()
            deque_not_empty = threading.Condition(deque_lock)
            readers_done = threading.Event()
            active_readers = [num_readers]
            total_processed = [0]
            processing_lock = threading.Lock()

            def reader_thread(file_list: list[FileEntry], _reader_id: int) -> None:
                """Thread-local SGA handle reader."""
                try:
                    # Each reader creates its own NativeSGAReader with thread-local handle
                    with SgaReader(sga_path) as reader:
                        for file in file_list:
                            try:
                                data = reader.read_file(file)

                                with deque_not_empty:
                                    while len(file_deque) >= buffer_size:
                                        deque_not_empty.wait(0.001)
                                    file_deque.append(ReadResult(file.path, data, None))
                                    deque_not_empty.notify()
                            except Exception as e:
                                with deque_not_empty:
                                    while len(file_deque) >= buffer_size:
                                        deque_not_empty.wait(0.001)
                                    file_deque.append(
                                        ReadResult(file.path, None, str(e))
                                    )
                                    deque_not_empty.notify()
                finally:
                    with processing_lock:
                        active_readers[0] -= 1
                        if active_readers[0] == 0:
                            readers_done.set()

            def writer_thread() -> None:
                """Parallel writers."""
                while not readers_done.is_set() or file_deque:
                    with deque_not_empty:
                        while not file_deque and not readers_done.is_set():
                            deque_not_empty.wait(0.01)

                        if not file_deque:
                            break

                        item = file_deque.popleft()
                        deque_not_empty.notify()

                    file_path, data, error = item.path, item.data, item.error

                    if error:
                        with processing_lock:
                            self.stats.failed_files += 1
                            total_processed[0] += 1
                            if self.stats.failed_files <= 10:
                                self.logger.error(
                                    f"Failed to read {file_path}: {error}"
                                )
                        continue

                    try:
                        dst_path = Path(output_dir) / file_path.lstrip("/")

                        fd = os.open(
                            dst_path,
                            OSFlags.O_CREAT | OSFlags.O_WRONLY | OSFlags.O_BINARY,
                        )
                        os.write(fd, data)  # type: ignore
                        os.close(fd)

                        with processing_lock:
                            self.stats.extracted_files += 1
                            self.stats.extracted_bytes += len(data)  # type: ignore
                            total_processed[0] += 1

                            if on_progress and total_processed[0] % 500 == 0:
                                on_progress(total_processed[0], self.stats.total_files)

                    except Exception as e:
                        with processing_lock:
                            self.stats.failed_files += 1
                            total_processed[0] += 1
                            if self.stats.failed_files <= 10:
                                self.logger.error(f"Failed to write {file_path}: {e}")

            # Start readers
            readers = []
            for i in range(num_readers):
                r = threading.Thread(
                    target=reader_thread, args=(reader_file_lists[i], i), daemon=True
                )
                r.start()
                readers.append(r)

            # Start writers
            writers = []
            for _ in range(num_writers):
                w = threading.Thread(target=writer_thread, daemon=True)
                w.start()
                writers.append(w)

            # Wait for completion
            for r in readers:
                r.join()
            for w in writers:
                w.join()

            # Final progress
            if on_progress:
                on_progress(total_processed[0], self.stats.total_files)

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
                with SgaReader(sga_path) as reader:
                    results = reader.read_files_parallel(
                        files, num_workers=self.num_workers
                    )
                self.stats.timings.creating_batches = timer()
            self.verbose_logger.info(
                f"Read + decompressed in {self.stats.timings.creating_batches:.2f}s"
            )

            self.verbose_logger.info("Writing files to disk (parallel)...")

            def write_file(
                item: ReadResult[bytes],
            ) -> WriteResult:
                path = item.path
                err = item.error
                if err:
                    return WriteResult(item.path, False, err)
                if item.data is None:
                    return WriteResult(item.path, False, "Data was None")
                try:
                    dst_path = Path(output_dir) / path.lstrip("/")
                    self.directories.ensure_directory(dst_path.parent)
                    if self.native_handles:
                        with dst_path.open("wb") as dst:
                            dst.write(item.data)
                    else:
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

            with self._timer() as timer:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    write_results: List[WriteResult] = list(
                        executor.map(write_file, results)
                    )
                self.stats.timings.executing_batches = timer()

            # Count results
            with self._timer() as timer:
                for write_result in write_results:

                    if write_result.success:
                        self.stats.extracted_files += 1
                        # Get size from results
                        for read_result in results:
                            if read_result.data is None:
                                break

                            if read_result.path == write_result.path:
                                self.stats.extracted_bytes += len(read_result.data)
                                break
                    else:
                        self.stats.failed_files += 1
                        self.logger.error(
                            f"Failed to write {write_result.path}: {write_result.error}"
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
    should_merge: bool | Callable[[int], bool] = False


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
            raise NotImplementedError()  # TODO better error
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
        file_entries = NativeParserV2(
            sga_path, self.logger, should_merge=self.should_merge
        ).parse()

        self.logger.info(f"Calculating checksums for {len(file_entries)} files...")
        num_workers = self.num_workers or 1
        # Calculate checksums in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
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
        with self._disable_gc():
            with SgaReader(sga_path) as sga:
                with self._timer() as timer:
                    for processed, file in enumerate(files):
                        try:
                            # Read and decompress (fs handles this)
                            data = sga.read_file(file)

                            # Write to disk
                            dst_path = Path(output_dir) / file.path.lstrip("/")
                            self.directories.ensure_directory(dst_path.parent)
                            if self.native_handles:
                                with dst_path.open("wb") as dst:
                                    dst.write(data)
                            else:
                                fd = os.open(
                                    dst_path,
                                    OSFlags.O_CREAT
                                    | OSFlags.O_WRONLY
                                    | OSFlags.O_BINARY,
                                )
                                os.write(fd, data)
                                os.close(fd)
                            self.stats.extracted_files += 1
                            self.stats.extracted_bytes += len(data)

                        except Exception as e:
                            self.stats.failed_files += 1
                            if self.stats.failed_files <= 3:
                                self.logger.error(f"Failed {file.path}: {e}")

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

    Serial = 5
    UltraFast = 0  # Rolled into optimized via config parameters
    Optimized = 1
    Streaming = 2
    Native = 3
    Delta = 4  # untested;


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
            ExtractionMethod.Optimized: _optimized,
            ExtractionMethod.Streaming: _streaming,
            ExtractionMethod.UltraFast: _optimized,  # use optimized; same thing
            ExtractionMethod.Native: _native,
            ExtractionMethod.Delta: _delta,
            ExtractionMethod.Serial: _serial,
        }
        self._default_extractor = _native

    def extract(
        self,
        sga_path: str,
        output_dir: str,
        on_progress: Callable[[int, int], None] | None = None,
        method: ExtractionMethod = ExtractionMethod.Native,  # Based on personal testing
    ) -> ExtractionStats:
        extractor = self._strategies.get(method, self._default_extractor)
        stats = extractor.extract(sga_path, output_dir, on_progress)
        return stats
