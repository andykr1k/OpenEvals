from __future__ import annotations

import json
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ..models import RequestResult, WorkloadCase, utc_now

StreamKind = Literal["content", "reasoning"]
StreamCallback = Callable[[StreamKind, str], Awaitable[None] | None]


@dataclass(slots=True)
class StreamState:
    text: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    token_times: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None


class OpenAICompatibleClient:
    """Streaming chat client for vLLM and other OpenAI-compatible servers."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_seconds: float = 300.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._client: Any = None

    @property
    def chat_url(self) -> str:
        if self.endpoint.endswith("/v1"):
            return f"{self.endpoint}/chat/completions"
        return f"{self.endpoint}/v1/chat/completions"

    async def request(
        self,
        case: WorkloadCase,
        concurrency: int,
        on_stream: StreamCallback | None = None,
    ) -> RequestResult:
        started_wall = utc_now()
        started = time.perf_counter()
        state = StreamState()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": case.messages,
            "max_tokens": case.max_tokens,
            "temperature": case.temperature,
            "top_p": case.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
            **case.extra_body,
        }
        if case.stop:
            payload["stop"] = case.stop
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            client = self._get_client()
            async with client.stream(
                "POST", self.chat_url, json=payload, headers=headers
            ) as response:
                status_code = response.status_code
                if response.is_error:
                    body = (await response.aread()).decode(errors="replace")[:2000]
                    return self._failure(
                        case, concurrency, started_wall, started, status_code, body
                    )
                async for line in response.aiter_lines():
                    for kind, text in self._consume_sse_line(line, state):
                        if on_stream:
                            callback_result = on_stream(kind, text)
                            if inspect.isawaitable(callback_result):
                                await callback_result
        except Exception as exc:  # the result must record transport failures, not abort the run
            return self._failure(case, concurrency, started_wall, started, None, str(exc))

        ended = time.perf_counter()
        latency = ended - started
        ttft = state.token_times[0] - started if state.token_times else None
        generation_seconds = ended - state.token_times[0] if state.token_times else None
        # vLLM provides exact counts in the final usage chunk. Chunk count is a useful
        # compatibility fallback for servers that omit usage (normally one token/chunk).
        output_tokens = (
            state.output_tokens if state.output_tokens is not None else len(state.token_times)
        )
        prompt_tokens = state.prompt_tokens or case.expected_prompt_tokens
        inter_token = [
            (current - previous) * 1000
            for previous, current in zip(state.token_times, state.token_times[1:])
        ]
        output_tps = None
        if (
            output_tokens is not None
            and output_tokens > 1
            and generation_seconds
            and generation_seconds > 0
        ):
            output_tps = (output_tokens - 1) / generation_seconds
        elif output_tokens == 1 and latency > 0:
            output_tps = 1 / latency
        total_tps = None
        if prompt_tokens is not None and output_tokens is not None and latency > 0:
            total_tps = (prompt_tokens + output_tokens) / latency
        prefill_tps = None
        if prompt_tokens is not None and ttft is not None and ttft > 0:
            prefill_tps = prompt_tokens / ttft

        return RequestResult(
            case_id=case.id,
            concurrency=concurrency,
            started_at=started_wall,
            success=True,
            latency_ms=latency * 1000,
            ttft_ms=ttft * 1000 if ttft is not None else None,
            generation_ms=generation_seconds * 1000 if generation_seconds is not None else None,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            prefill_tps=prefill_tps,
            output_tps=output_tps,
            total_tps=total_tps,
            inter_token_ms=inter_token,
            status_code=200,
            finish_reason=state.finish_reason,
            response_text="".join(state.text),
            response_reasoning="".join(state.reasoning),
        )

    def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            timeout = httpx.Timeout(self.timeout_seconds, connect=min(self.timeout_seconds, 30.0))
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _consume_sse_line(line: str, state: StreamState) -> list[tuple[StreamKind, str]]:
        emitted_chunks: list[tuple[StreamKind, str]] = []
        if not line.startswith("data:"):
            return emitted_chunks
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            return emitted_chunks
        try:
            chunk = json.loads(raw)
        except json.JSONDecodeError:
            return emitted_chunks
        usage = chunk.get("usage")
        if usage:
            state.prompt_tokens = usage.get("prompt_tokens", state.prompt_tokens)
            state.output_tokens = usage.get("completion_tokens", state.output_tokens)
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            emitted = False
            content = delta.get("content")
            if content:
                state.text.append(content)
                emitted_chunks.append(("content", content))
                emitted = True
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                state.reasoning.append(reasoning)
                emitted_chunks.append(("reasoning", reasoning))
                emitted = True
            if emitted:
                state.token_times.append(time.perf_counter())
            if choice.get("finish_reason"):
                state.finish_reason = choice["finish_reason"]
        return emitted_chunks

    @staticmethod
    def _failure(
        case: WorkloadCase,
        concurrency: int,
        started_wall: str,
        started: float,
        status_code: int | None,
        error: str,
    ) -> RequestResult:
        return RequestResult(
            case_id=case.id,
            concurrency=concurrency,
            started_at=started_wall,
            success=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            status_code=status_code,
            error=error,
        )
