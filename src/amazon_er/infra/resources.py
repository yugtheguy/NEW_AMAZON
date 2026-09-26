"""Lightweight CPU, RAM, and optional nvidia-smi resource monitoring."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable

import psutil


@dataclass(frozen=True)
class ResourceSample:
    timestamp: str
    stage: str
    country: str | None
    source: str | None
    shard: str | None
    processed_rows: int | None
    rows_per_second: float | None
    rss_bytes: int
    available_ram_bytes: int
    ram_fraction: float
    gpu_available: bool
    gpu_utilization_percent: float | None
    gpu_memory_used_bytes: int | None
    gpu_memory_total_bytes: int | None
    elapsed_seconds: float
    eta_seconds: float | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def query_gpu() -> dict[str, object]:
    command = [
        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=True)
        line = next(line for line in result.stdout.splitlines() if line.strip())
        utilization, used_mib, total_mib = [float(item.strip()) for item in line.split(",")[:3]]
        mib = 1024 * 1024
        return {"available": True, "utilization": utilization, "used": int(used_mib * mib), "total": int(total_mib * mib)}
    except (OSError, subprocess.SubprocessError, StopIteration, ValueError):
        return {"available": False, "utilization": None, "used": None, "total": None}


class ResourceMonitor:
    def __init__(self, stage: str, *, country: str | None = None, source: str | None = None, shard: str | None = None) -> None:
        self.stage, self.country, self.source, self.shard = stage, country, source, shard
        self.started = time.monotonic()
        self._initial_rows = 0

    def sample(self, processed_rows: int | None = None, total_rows: int | None = None) -> ResourceSample:
        elapsed = max(0.0, time.monotonic() - self.started)
        rate = processed_rows / elapsed if processed_rows is not None and elapsed > 0 else None
        eta = ((total_rows - processed_rows) / rate) if rate and total_rows is not None and processed_rows is not None else None
        memory = psutil.virtual_memory()
        gpu = query_gpu()
        return ResourceSample(
            timestamp=datetime.now(timezone.utc).isoformat(), stage=self.stage, country=self.country,
            source=self.source, shard=self.shard, processed_rows=processed_rows, rows_per_second=rate,
            rss_bytes=psutil.Process().memory_info().rss, available_ram_bytes=int(memory.available),
            ram_fraction=float(memory.percent) / 100.0, gpu_available=bool(gpu["available"]),
            gpu_utilization_percent=gpu["utilization"], gpu_memory_used_bytes=gpu["used"],
            gpu_memory_total_bytes=gpu["total"], elapsed_seconds=elapsed, eta_seconds=eta,
        )

    def memory_state(self, warning_fraction: float, critical_fraction: float) -> str:
        fraction = self.sample().ram_fraction
        if fraction >= critical_fraction:
            return "critical"
        if fraction >= warning_fraction:
            return "warning"
        return "normal"

    def start_heartbeat(self, interval_seconds: float, sink: Callable[[ResourceSample], None]) -> tuple[threading.Event, threading.Thread]:
        stop = threading.Event()
        def run() -> None:
            while not stop.wait(interval_seconds):
                sink(self.sample())
        thread = threading.Thread(target=run, name=f"resource-{self.stage}", daemon=True)
        thread.start()
        return stop, thread
