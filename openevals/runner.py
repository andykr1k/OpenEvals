from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable

from .adapters import OpenAICompatibleClient
from .adapters.openai_compatible import StreamKind
from .evaluators import evaluate_checks
from .metrics import summarize
from .models import BenchmarkConfig, RequestResult, RunResult, WorkloadCase, utc_now
from .telemetry import HardwareMonitor, metric_delta, scrape_prometheus

ProgressCallback = Callable[[RunResult, RequestResult], Awaitable[None] | None]
StreamCallback = Callable[[int, WorkloadCase, int, StreamKind, str], Awaitable[None] | None]


class BenchmarkRunner:
    async def run(
        self,
        config: BenchmarkConfig,
        on_result: ProgressCallback | None = None,
        on_stream: StreamCallback | None = None,
    ) -> RunResult:
        self._validate(config)
        run = RunResult.create(config)
        client = OpenAICompatibleClient(
            config.endpoint, config.model, config.api_key, config.timeout_seconds
        )
        monitor = HardwareMonitor()
        monitor_task = asyncio.create_task(monitor.run())
        measured_started: float | None = None
        metrics_before: dict[str, float] = {}
        try:
            await self._warmup(client, config)
            metrics_before = await scrape_prometheus(config.metrics_endpoint)
            measured_started = time.perf_counter()
            for concurrency in config.concurrency:
                jobs = [case for _ in range(config.repetitions) for case in config.suite.cases]
                await self._run_stage(
                    client,
                    run,
                    jobs,
                    concurrency,
                    config.request_rate,
                    on_result,
                    on_stream,
                )
            run.status = "completed"
        except asyncio.CancelledError:
            run.status = "cancelled"
            raise
        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
        finally:
            monitor.stop()
            await monitor_task
            await client.aclose()
            elapsed = (
                time.perf_counter() - measured_started if measured_started is not None else None
            )
            metrics_after = await scrape_prometheus(config.metrics_endpoint)
            run.summary = summarize(run.requests, elapsed)
            run.summary["gpu"] = monitor.summary()
            run.summary["vllm_metrics_delta"] = metric_delta(metrics_before, metrics_after)
            run.finished_at = utc_now()
        return run

    async def _warmup(self, client: OpenAICompatibleClient, config: BenchmarkConfig) -> None:
        if not config.suite.cases:
            return
        for index in range(config.warmup_requests):
            await client.request(config.suite.cases[index % len(config.suite.cases)], 1)

    async def _run_stage(
        self,
        client: OpenAICompatibleClient,
        run: RunResult,
        cases: list[WorkloadCase],
        concurrency: int,
        request_rate: float | None,
        on_result: ProgressCallback | None,
        on_stream: StreamCallback | None,
    ) -> None:
        semaphore = asyncio.Semaphore(concurrency)

        async def execute(index: int, case: WorkloadCase) -> None:
            if request_rate:
                await asyncio.sleep(index / request_rate)

            async def relay_stream(kind: StreamKind, text: str) -> None:
                if on_stream:
                    callback_result = on_stream(index, case, concurrency, kind, text)
                    if inspect.isawaitable(callback_result):
                        await callback_result

            async with semaphore:
                result = await client.request(
                    case, concurrency, relay_stream if on_stream else None
                )
            if result.success and case.checks:
                result.check_results = evaluate_checks(result.response_text, case.checks)
                result.checks_total = len(result.check_results)
                result.checks_passed = sum(item["passed"] for item in result.check_results)
            run.requests.append(result)
            if on_result:
                callback_result = on_result(run, result)
                if inspect.isawaitable(callback_result):
                    await callback_result

        async with asyncio.TaskGroup() as group:
            for index, case in enumerate(cases):
                group.create_task(execute(index, case))

    @staticmethod
    def _validate(config: BenchmarkConfig) -> None:
        if not config.suite.cases:
            raise ValueError("The benchmark suite has no cases")
        if config.repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        if not config.concurrency or any(value < 1 for value in config.concurrency):
            raise ValueError("concurrency values must be positive")
        if config.request_rate is not None and config.request_rate <= 0:
            raise ValueError("request_rate must be positive")
