"""Opt-in process windows and task-local, aggregate worker diagnostics.

Process CPU counters are sampled only by the coordinating thread. Worker
records use thread CPU and elapsed service time and must never be added to
process wall time. RSS sampling between window boundaries belongs to the
launcher, outside the trial's GIL.
"""

import contextlib
import dataclasses
import logging
import resource
import time

import psutil

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Measurement:
    """One aggregate for a named stage and accounting scope."""

    stage: str
    scope: str
    call_count: int = 0
    wall_s: float | None = None
    user_s: float | None = None
    system_s: float | None = None
    service_s: float | None = None
    thread_cpu_s: float | None = None


@dataclasses.dataclass
class Collector:
    """Trial or haplotype-local records, merged only by the coordinator.

    Disabling ``stages`` keeps workload counters while omitting regional timers.
    This permits separate total-only and diagnostic experiments.
    """

    stages: bool = True
    records: dict = dataclasses.field(default_factory=dict)
    workload: dict = dataclasses.field(default_factory=dict)
    haplotype_times: list[dict] = dataclasses.field(default_factory=list)

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
            for name in ("wall_s", "user_s", "system_s", "service_s", "thread_cpu_s"):
                value = getattr(measurement, name)
                if value is not None:
                    setattr(target, name, getattr(target, name) + value)

    def merge(self, other):
        for measurement in other.records.values():
            self.add(measurement)
        self.count(**other.workload)


@contextlib.contextmanager
def _measure(collector, stage, scope):
    start = time.perf_counter()
    if scope == "process":
        cpu = resource.getrusage(resource.RUSAGE_SELF)
    else:
        cpu = time.thread_time()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        record = Measurement(stage, scope, call_count=1)
        if scope == "process":
            end_cpu = resource.getrusage(resource.RUSAGE_SELF)
            record.wall_s = elapsed
            record.user_s = end_cpu.ru_utime - cpu.ru_utime
            record.system_s = end_cpu.ru_stime - cpu.ru_stime
        else:
            record.service_s = elapsed
            record.thread_cpu_s = time.thread_time() - cpu
        collector.add(record)


def measure(collector, stage, scope="process"):
    """Return a no-op for ordinary runs; never sample process CPU in tasks."""
    if collector is None or not collector.stages:
        return contextlib.nullcontext()
    return _measure(collector, stage, scope)


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
        self._cpu = resource.getrusage(resource.RUSAGE_SELF)
        self.start_timestamp = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.end_timestamp = time.perf_counter()
        cpu = resource.getrusage(resource.RUSAGE_SELF)
        self.end_rss_bytes = psutil.Process().memory_info().rss
        self.wall_s = self.end_timestamp - self.start_timestamp
        self.user_s = cpu.ru_utime - self._cpu.ru_utime
        self.system_s = cpu.ru_stime - self._cpu.ru_stime
