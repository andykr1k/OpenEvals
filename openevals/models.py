from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class WorkloadCase:
    """One request shape in a benchmark suite.

    Messages use the OpenAI chat schema, including multimodal content arrays.
    """

    id: str
    messages: list[dict[str, Any]]
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] | None = None
    expected_prompt_tokens: int | None = None
    tags: list[str] = field(default_factory=list)
    extra_body: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkloadCase:
        value = dict(value)
        if "prompt" in value and "messages" not in value:
            value["messages"] = [{"role": "user", "content": value.pop("prompt")}]
        return cls(**value)


@dataclass(slots=True)
class Suite:
    name: str
    description: str = ""
    cases: list[WorkloadCase] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Suite:
        return cls(
            name=value["name"],
            description=value.get("description", ""),
            cases=[WorkloadCase.from_dict(item) for item in value.get("cases", [])],
        )

    @classmethod
    def load(cls, path: str | Path) -> Suite:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass(slots=True)
class BenchmarkConfig:
    endpoint: str
    model: str
    suite: Suite
    api_key: str = "EMPTY"
    concurrency: list[int] = field(default_factory=lambda: [1])
    repetitions: int = 1
    warmup_requests: int = 1
    timeout_seconds: float = 300.0
    request_rate: float | None = None
    metrics_endpoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkConfig:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
        value["suite"] = Suite.from_dict(value["suite"])
        return cls(**value)


@dataclass(slots=True)
class RequestResult:
    case_id: str
    concurrency: int
    started_at: str
    success: bool
    latency_ms: float
    ttft_ms: float | None = None
    generation_ms: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    prefill_tps: float | None = None
    output_tps: float | None = None
    total_tps: float | None = None
    inter_token_ms: list[float] = field(default_factory=list)
    status_code: int | None = None
    error: str | None = None
    finish_reason: str | None = None
    response_text: str = ""
    response_reasoning: str = ""
    server_metrics: dict[str, float] = field(default_factory=dict)
    checks_passed: int = 0
    checks_total: int = 0
    check_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class RunResult:
    id: str
    created_at: str
    finished_at: str | None
    status: Literal["running", "completed", "failed", "cancelled"]
    endpoint: str
    model: str
    suite_name: str
    config: dict[str, Any]
    requests: list[RequestResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def create(cls, config: BenchmarkConfig) -> RunResult:
        safe_config = asdict(config)
        safe_config["api_key"] = "***" if config.api_key else ""
        return cls(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            finished_at=None,
            status="running",
            endpoint=config.endpoint,
            model=config.model,
            suite_name=config.suite.name,
            config=safe_config,
        )
