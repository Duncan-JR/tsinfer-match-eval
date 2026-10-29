"""Opt-in process windows and task-local, aggregate worker diagnostics.

Process CPU counters are sampled only by the coordinating thread. Worker
records use thread CPU and elapsed service time and must never be added to
process wall time. Benchmark processes self-sample private memory; the launcher
aggregates atomic snapshots without multiplying shared mapped pages by workers.
"""

import contextlib
import dataclasses
import json
import logging
import os
import pathlib
import resource
import threading
import time

import psutil

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Measurement:
    """One aggregate for a named stage and accounting scope."""

    stage: str
    scope: str
    call_count: int = 0
    parent_stage: str | None = None
    inclusive: bool = False
    start_timestamp: float | None = None
    end_timestamp: float | None = None
    wall_s: float | None = None
    user_s: float | None = None
    system_s: float | None = None
    service_s: float | None = None
    thread_cpu_s: float | None = None


@dataclasses.dataclass
class HaplotypeTiming:
    """Scalar service-time boundaries captured around one matching call."""

    worker_pid: int
    match_start_s: float
    match_end_s: float
    match_wall_s: float
    match_thread_cpu_s: float
    domain_sites: int
    focal_sites: int
    gap_sites: int
    gap_calls: int


@dataclasses.dataclass
class Collector:
    """Trial or haplotype-local records, merged only by the coordinator.

    Disabling ``stages`` keeps workload counters while omitting regional timers.
    This permits separate total-only and diagnostic experiments.
    """

    stages: bool = True
    records: dict = dataclasses.field(default_factory=dict)
    workload: dict = dataclasses.field(default_factory=dict)

    diagnostics: dict = dataclasses.field(default_factory=dict)
    haplotype_timings: list[dict] = dataclasses.field(default_factory=list)

    def count(self, **counts):
        for name, value in counts.items():
            self.workload[name] = self.workload.get(name, 0) + int(value)

    def add(self, measurement):
        key = (measurement.stage, measurement.scope)
        if key not in self.records:
            self.records[key] = measurement
        else:
            target = self.records[key]
            target.call_count += measurement.call_count
            if measurement.start_timestamp is not None:
                target.start_timestamp = min(
                    target.start_timestamp, measurement.start_timestamp
                )
                target.end_timestamp = max(
                    target.end_timestamp, measurement.end_timestamp
                )
            for name in ("wall_s", "user_s", "system_s", "service_s", "thread_cpu_s"):
                value = getattr(measurement, name)
                if value is not None:
                    setattr(target, name, getattr(target, name) + value)

    def elapsed(self, stage, duration, *, parent_stage=None, inclusive=False):
        """Record coordinator elapsed time or latency without sampling process CPU."""
        if self.stages:
            self.add(
                Measurement(
                    stage,
                    "coordinator",
                    call_count=1,
                    wall_s=duration,
                    parent_stage=parent_stage,
                    inclusive=inclusive,
                )
            )

    def merge(self, other):
        for measurement in other.records.values():
            self.add(measurement)
        self.count(**other.workload)


@contextlib.contextmanager
def _measure(collector, stage, scope, parent_stage, inclusive):
    start = time.perf_counter()
    if scope == "process":
        cpu = resource.getrusage(resource.RUSAGE_SELF)
    elif scope == "thread":
        cpu = time.thread_time()
    try:
        yield
    finally:
        end = time.perf_counter()
        elapsed = end - start
        record = Measurement(
            stage,
            scope,
            call_count=1,
            parent_stage=parent_stage,
            inclusive=inclusive,
            start_timestamp=start,
            end_timestamp=end,
        )
        if scope == "process":
            end_cpu = resource.getrusage(resource.RUSAGE_SELF)
            record.wall_s = elapsed
            record.user_s = end_cpu.ru_utime - cpu.ru_utime
            record.system_s = end_cpu.ru_stime - cpu.ru_stime
        elif scope == "thread":
            record.service_s = elapsed
            record.thread_cpu_s = time.thread_time() - cpu
        else:
            record.wall_s = elapsed
        collector.add(record)


def measure(collector, stage, scope="process", *, parent_stage=None, inclusive=False):
    """Return a no-op for ordinary runs; never sample process CPU in tasks."""
    if collector is None or not collector.stages:
        return contextlib.nullcontext()
    return _measure(collector, stage, scope, parent_stage, inclusive)


@dataclasses.dataclass
class Window:
    """One directly measured total, with cross-process monotonic timestamps."""

    start_timestamp: float | None = None
    end_timestamp: float | None = None
    start_rss_bytes: int | None = None
    end_rss_bytes: int | None = None
    wall_s: float | None = None
    user_s: float | None = None
    system_s: float | None = None

    def __enter__(self):
        self.start_rss_bytes = psutil.Process().memory_info().rss
        self._cpu = _tree_cpu()
        self.start_timestamp = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.end_timestamp = time.perf_counter()
        cpu = _tree_cpu()
        self.end_rss_bytes = psutil.Process().memory_info().rss
        self.wall_s = self.end_timestamp - self.start_timestamp
        self.user_s = cpu.user - self._cpu.user
        self.system_s = cpu.system - self._cpu.system


@dataclasses.dataclass
class CpuTime:
    """Cumulative process-tree CPU seconds at a measurement boundary."""

    user: float
    system: float


def _tree_cpu():
    """Include reaped children and live worker CPU at each window boundary."""
    own = resource.getrusage(resource.RUSAGE_SELF)
    reaped = resource.getrusage(resource.RUSAGE_CHILDREN)
    user = own.ru_utime + reaped.ru_utime
    system = own.ru_stime + reaped.ru_stime
    for child in psutil.Process().children(recursive=True):
        try:
            cpu = child.cpu_times()
            user += cpu.user
            system += cpu.system
        except psutil.NoSuchProcess:
            pass
    return CpuTime(user, system)


@contextlib.contextmanager
def sample_memory():
    """Benchmark-only self sampling; macOS forbids inspecting another task's USS.

    The launcher aggregates atomic per-process snapshots. An inherited benchmark
    environment enables this in its trial and matching workers only.
    """
    folder = os.environ.get("MATCH_EVAL_MEMORY_DIR")
    if folder is None:
        yield
        return
    directory = pathlib.Path(folder)
    interval = float(os.environ["MATCH_EVAL_MEMORY_INTERVAL_S"])
    process = psutil.Process()
    path = directory / f"memory_{process.pid}.json"
    temporary = path.with_suffix(".tmp")
    stop = threading.Event()

    history = directory / f"memory_{process.pid}.jsonl"

    def publish(active):
        info = process.memory_full_info()
        pss = getattr(info, "pss", None)
        if not active and pss is not None:
            pss = 0
        record = {
            "timestamp": time.perf_counter(),
            "private_bytes": info.uss if active else 0,
            "pss_bytes": pss,
            "active": active,
        }
        temporary.write_text(json.dumps(record))
        temporary.replace(path)
        with history.open("a") as output:
            output.write(json.dumps(record) + "\n")

    def sample():
        while not stop.is_set():
            publish(True)
            stop.wait(interval)
        publish(False)

    thread = threading.Thread(target=sample)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
