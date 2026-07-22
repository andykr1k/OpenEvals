from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

from .models import BenchmarkConfig, Suite, WorkloadCase
from .runner import BenchmarkRunner
from .suites import built_in_suites, standard_suite
from .terminal import TerminalReporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="open-evals", description="Benchmark local LLM/VLM servers"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Send requests and print analytics")
    run.add_argument("config", type=Path, nargs="?", help="Optional benchmark JSON")
    run.add_argument("--endpoint", default="http://127.0.0.1:8000")
    run.add_argument("--model", help="Served model name (required without a config)")
    run.add_argument("--prompt", default="Say hello in one short sentence.")
    run.add_argument("--requests", type=int, default=3, help="Requests per concurrency stage")
    run.add_argument("--concurrency", type=int, nargs="+", default=[1])
    run.add_argument("--max-tokens", type=int, default=64)
    run.add_argument("--warmups", type=int, default=1)
    run.add_argument("--timeout", type=float, default=300.0)
    run.add_argument("--metrics-endpoint")
    output = run.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="Print full result JSON")
    output.add_argument("--no-stream", action="store_true", help="Hide live response samples")

    init = subparsers.add_parser("init", help="Write an editable starter configuration")
    init.add_argument("path", type=Path, nargs="?", default=Path("benchmark.json"))
    init.add_argument("--endpoint", default="http://127.0.0.1:8000")
    init.add_argument("--model", default="Qwen/Qwen3-8B-AWQ")

    subparsers.add_parser("suites", help="List built-in benchmark suites")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        if args.config:
            config = BenchmarkConfig.load(args.config)
        else:
            if not args.model:
                raise SystemExit("open-evals run: --model is required without a config file")
            config = BenchmarkConfig(
                endpoint=args.endpoint,
                model=args.model,
                api_key=os.environ.get("OPEN_EVALS_API_KEY", "EMPTY"),
                suite=Suite(
                    name="terminal",
                    cases=[
                        WorkloadCase(
                            id="prompt",
                            messages=[{"role": "user", "content": args.prompt}],
                            max_tokens=args.max_tokens,
                        )
                    ],
                ),
                concurrency=args.concurrency,
                repetitions=args.requests,
                warmup_requests=args.warmups,
                timeout_seconds=args.timeout,
                metrics_endpoint=args.metrics_endpoint,
            )
        reporter = TerminalReporter()
        stream_callback = None if args.json or args.no_stream else reporter.stream_chunk
        result = asyncio.run(BenchmarkRunner().run(config, on_stream=stream_callback))
        if args.json:
            print(json.dumps(asdict(result), indent=2))
        else:
            reporter.print_result(result)
        if result.status != "completed":
            raise SystemExit(1)
    elif args.command == "init":
        config = BenchmarkConfig(
            endpoint=args.endpoint,
            model=args.model,
            suite=standard_suite(),
            concurrency=[1, 2, 4, 8],
            repetitions=2,
            metrics_endpoint=f"{args.endpoint.rstrip('/')}/metrics",
        )
        args.path.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.path}")
    elif args.command == "suites":
        for name, suite in built_in_suites().items():
            print(f"{name:16} {len(suite.cases):3} cases  {suite.description}")


if __name__ == "__main__":
    main()
