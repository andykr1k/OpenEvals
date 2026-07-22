from __future__ import annotations

import sys
from typing import Any, TextIO

from .adapters.openai_compatible import StreamKind
from .models import RunResult, WorkloadCase


class TerminalReporter:
    """Render a small live sample and the final benchmark analytics."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout
        self._active_sample: tuple[int, str] | None = None
        self._active_kind: StreamKind | None = None

    def stream_chunk(
        self,
        request_index: int,
        case: WorkloadCase,
        concurrency: int,
        kind: StreamKind,
        text: str,
    ) -> None:
        # One live request per concurrency stage stays readable while the other
        # requests run concurrently in the background.
        if request_index != 0:
            return
        sample = (concurrency, case.id)
        if sample != self._active_sample:
            self._finish_stream()
            print(
                f"\nLive response [{case.id}, concurrency {concurrency}]",
                file=self.stream,
            )
            self._active_sample = sample
            self._active_kind = None
        if kind != self._active_kind:
            if self._active_kind is not None:
                print(file=self.stream)
            print("Reasoning: " if kind == "reasoning" else "Response: ", end="", file=self.stream)
            self._active_kind = kind
        print(text, end="", file=self.stream, flush=True)

    def print_result(self, result: RunResult) -> None:
        self._finish_stream()
        summary = result.summary
        print(
            f"\n{result.status.capitalize()}: {result.model} / {result.suite_name}",
            file=self.stream,
        )
        print(
            "Requests: "
            f"{summary.get('successful', 0)}/{summary.get('requests', 0)} successful"
            f"  |  wall time: {_number(summary.get('wall_time_seconds'), 2, 's')}"
            f"  |  throughput: {_number(summary.get('request_throughput_rps'), 2, ' req/s')}"
            f"  |  output: {_number(summary.get('output_throughput_tps'), 2, ' tok/s')}",
            file=self.stream,
        )

        rows = [
            ("Latency (ms)", summary.get("latency_ms", {})),
            ("TTFT (ms)", summary.get("ttft_ms", {})),
            ("Generation (ms)", summary.get("generation_ms", {})),
            ("Output (tok/s)", summary.get("output_tps", {})),
            ("Inter-token (ms)", summary.get("inter_token_ms", {})),
        ]
        print(
            "\nMetric               mean       p50       p95       p99       max", file=self.stream
        )
        print(
            "----------------  --------  --------  --------  --------  --------", file=self.stream
        )
        for label, stats in rows:
            values = [stats.get(key) for key in ("mean", "p50", "p95", "p99", "max")]
            print(
                f"{label:<16}" + "".join(f"  {_cell(value):>8}" for value in values),
                file=self.stream,
            )

        by_concurrency = summary.get("by_concurrency", {})
        if len(by_concurrency) > 1:
            print(
                "\nConcurrency  success    p50 latency    p95 TTFT    mean output",
                file=self.stream,
            )
            print(
                "-----------  ---------  -------------  ----------  -----------",
                file=self.stream,
            )
            for concurrency, stage in by_concurrency.items():
                success = f"{stage.get('successful', 0)}/{stage.get('requests', 0)}"
                latency = _cell(stage.get("latency_ms", {}).get("p50"))
                ttft = _cell(stage.get("ttft_ms", {}).get("p95"))
                output = _cell(stage.get("output_tps", {}).get("mean"))
                print(
                    f"{concurrency:>11}  {success:>9}  {latency:>10} ms"
                    f"  {ttft:>7} ms  {output:>8} tok/s",
                    file=self.stream,
                )

        checks_total = summary.get("checks_total", 0)
        if checks_total:
            print(
                f"\nChecks: {summary.get('checks_passed', 0)}/{checks_total} passed",
                file=self.stream,
            )
        errors = summary.get("errors", {})
        if errors:
            print("\nErrors:", file=self.stream)
            for message, count in errors.items():
                print(f"  {count}x {message}", file=self.stream)
        if result.error:
            print(f"\nRun error: {result.error}", file=self.stream)

    def _finish_stream(self) -> None:
        if self._active_sample is not None:
            print(file=self.stream)
        self._active_sample = None
        self._active_kind = None


def _cell(value: Any) -> str:
    return "-" if value is None else f"{value:.2f}"


def _number(value: Any, digits: int, suffix: str) -> str:
    return "-" if value is None else f"{value:.{digits}f}{suffix}"
