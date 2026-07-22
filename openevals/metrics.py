from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Iterable

from .models import RequestResult


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    materialized = list(values)
    if not materialized:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(materialized),
        "mean": statistics.fmean(materialized),
        "min": min(materialized),
        "p50": percentile(materialized, 0.50),
        "p90": percentile(materialized, 0.90),
        "p95": percentile(materialized, 0.95),
        "p99": percentile(materialized, 0.99),
        "max": max(materialized),
    }


def summarize(
    requests: list[RequestResult], wall_time_seconds: float | None = None
) -> dict[str, Any]:
    successful = [item for item in requests if item.success]
    failed = [item for item in requests if not item.success]
    grouped: dict[int, list[RequestResult]] = defaultdict(list)
    for item in requests:
        grouped[item.concurrency].append(item)

    def stats(items: list[RequestResult]) -> dict[str, Any]:
        good = [item for item in items if item.success]
        value = {
            "requests": len(items),
            "successful": len(good),
            "failed": len(items) - len(good),
            "latency_ms": describe(item.latency_ms for item in good),
            "ttft_ms": describe(item.ttft_ms for item in good if item.ttft_ms is not None),
            "generation_ms": describe(
                item.generation_ms for item in good if item.generation_ms is not None
            ),
            "prefill_tps": describe(
                item.prefill_tps for item in good if item.prefill_tps is not None
            ),
            "output_tps": describe(item.output_tps for item in good if item.output_tps is not None),
            "total_tps": describe(item.total_tps for item in good if item.total_tps is not None),
            "inter_token_ms": describe(value for item in good for value in item.inter_token_ms),
            "prompt_tokens": sum(item.prompt_tokens or 0 for item in good),
            "output_tokens": sum(item.output_tokens or 0 for item in good),
            "checks_passed": sum(item.checks_passed for item in good),
            "checks_total": sum(item.checks_total for item in good),
        }
        value["check_pass_rate"] = (
            value["checks_passed"] / value["checks_total"] if value["checks_total"] else None
        )
        return value

    result = stats(requests)
    result["success_rate"] = len(successful) / len(requests) if requests else 0.0
    result["check_pass_rate"] = (
        result["checks_passed"] / result["checks_total"] if result["checks_total"] else None
    )
    result["by_concurrency"] = {str(key): stats(value) for key, value in sorted(grouped.items())}
    result["errors"] = _error_counts(failed)
    if wall_time_seconds and wall_time_seconds > 0:
        result["wall_time_seconds"] = wall_time_seconds
        result["request_throughput_rps"] = len(successful) / wall_time_seconds
        result["output_throughput_tps"] = (
            sum(x.output_tokens or 0 for x in successful) / wall_time_seconds
        )
    return result


def _error_counts(failed: list[RequestResult]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in failed:
        key = item.error or (f"HTTP {item.status_code}" if item.status_code else "Unknown error")
        counts[key] += 1
    return dict(counts)
