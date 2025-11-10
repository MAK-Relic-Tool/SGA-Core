import datetime
import json
import logging
import multiprocessing
import sys
import tempfile
from contextlib import contextmanager
from typing import Generator, Callable, Any

from relic.sga.core.native.parallel_advanced import (
    UnpackerConfig,
    AdvancedParallelUnpacker,
    ExtractionMethod,
)

logger = logging.getLogger()
logger.setLevel(logging.ERROR)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.ERROR)
logger.addHandler(handler)


@contextmanager
def _timer() -> Generator[Callable[[], float], Any, None]:
    import time as time_module

    t0 = time_module.perf_counter()

    def delta() -> float:
        return time_module.perf_counter() - t0

    yield delta


if __name__ == "__main__":
    path = sys.argv[1]
    cfg = UnpackerConfig(
        num_workers=max(1, multiprocessing.cpu_count() - 1),
        logger=logger,
        disable_gc=True,
        native_files=False,
    )
    METHODS = [
        ExtractionMethod.Native,
        ExtractionMethod.Optimized,
        ExtractionMethod.UltraFast,
    ]
    unpacker = AdvancedParallelUnpacker(cfg)

    timings = {}
    RUNS = 10
    for method in METHODS:
        m_timings = timings[method] = []
        for runs in range(RUNS):
            with tempfile.TemporaryDirectory() as tmpdir:
                print(f"Extracting <{method.name}> - run {runs + 1}")
                with _timer() as timer:
                    unpacker.extract(path, tmpdir, method=method)
                    m_timings.append(timer())
    with open(
        "./prof/" + datetime.datetime.now().isoformat().replace(":", "_") + ".json", "w"
    ) as h:
        json.dump(timings, h, indent=4)

        for method in METHODS:
            m_timings = timings[method]
            print(method.name, ":", sum(m_timings) / RUNS)
            print("\t", ", ".join([str(v) for v in m_timings]))
        print()
