from __future__ import annotations

import asyncio
import csv
import io
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

PROMETHEUS_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+(?P<value>[-+0-9.eE]+)$"
)


def parse_prometheus(text: str) -> dict[str, float]:
    """Aggregate numeric Prometheus samples by metric name.

    Labels are intentionally summed so tensor-parallel workers become one model-level sample.
    """
    result: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_LINE.match(line)
        if not match:
            continue
        value = float(match.group("value"))
        name = match.group("name")
        result[name] = result.get(name, 0.0) + value
    return result


async def scrape_prometheus(endpoint: str | None) -> dict[str, float]:
    if not endpoint:
        return {}
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(endpoint)
            response.raise_for_status()
            return parse_prometheus(response.text)
    except Exception:
        return {}


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    interesting = (
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:request_success_total",
        "vllm:request_prompt_tokens_sum",
        "vllm:request_generation_tokens_sum",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:time_to_first_token_seconds_count",
        "vllm:time_per_output_token_seconds_sum",
        "vllm:time_per_output_token_seconds_count",
        "vllm:e2e_request_latency_seconds_sum",
        "vllm:e2e_request_latency_seconds_count",
    )
    return {
        key: value - before.get(key, 0.0)
        for key, value in after.items()
        if key in interesting and value >= before.get(key, 0.0)
    }


async def nvidia_snapshot() -> list[dict[str, Any]]:
    fields = [
        "index",
        "name",
        "uuid",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5)
        if process.returncode != 0:
            return []
    except (FileNotFoundError, TimeoutError):
        return []
    rows = csv.reader(io.StringIO(stdout.decode()))
    return [{field: _number(value.strip()) for field, value in zip(fields, row)} for row in rows]


def _number(value: str) -> Any:
    try:
        return float(value)
    except ValueError:
        return value


@dataclass(slots=True)
class HardwareMonitor:
    interval_seconds: float = 0.5
    samples: list[dict[str, Any]] = field(default_factory=list)
    _stop: asyncio.Event = field(default_factory=asyncio.Event)

    async def run(self) -> None:
        while not self._stop.is_set():
            self.samples.append({"timestamp": time.time(), "gpus": await nvidia_snapshot()})
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()

    def summary(self) -> dict[str, Any]:
        by_gpu: dict[str, dict[str, list[float]]] = {}
        for sample in self.samples:
            for gpu in sample["gpus"]:
                key = str(gpu.get("uuid") or gpu.get("index"))
                target = by_gpu.setdefault(key, {})
                for metric in (
                    "utilization.gpu",
                    "utilization.memory",
                    "memory.used",
                    "temperature.gpu",
                    "power.draw",
                ):
                    if isinstance(gpu.get(metric), float):
                        target.setdefault(metric, []).append(gpu[metric])
        return {
            gpu: {
                metric: {"mean": statistics.fmean(values), "max": max(values)}
                for metric, values in metrics.items()
            }
            for gpu, metrics in by_gpu.items()
        }
