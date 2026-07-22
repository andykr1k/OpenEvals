from __future__ import annotations

from typing import Any

from .models import Suite, WorkloadCase


_SEED_TEXT = (
    "Local inference systems balance model quality, memory capacity, batching, and latency. "
    "A useful benchmark fixes the workload, records every request, and reports distributions "
    "instead of a single average. "
)


def sized_prompt(approximate_tokens: int) -> str:
    # English prose is commonly near four characters/token. The server-reported usage remains canonical.
    target_chars = approximate_tokens * 4
    repeated = (_SEED_TEXT * (target_chars // len(_SEED_TEXT) + 1))[:target_chars]
    return f"Read the following passage and summarize its main point concisely.\n\n{repeated}"


def standard_suite() -> Suite:
    cases: list[WorkloadCase] = []
    for prompt_tokens in (128, 512, 2048, 8192):
        for output_tokens in (32, 128):
            cases.append(
                WorkloadCase(
                    id=f"context-{prompt_tokens}-output-{output_tokens}",
                    messages=[{"role": "user", "content": sized_prompt(prompt_tokens)}],
                    max_tokens=output_tokens,
                    tags=["context-sweep", f"prompt-{prompt_tokens}", f"output-{output_tokens}"],
                )
            )
    cases.extend(
        [
            WorkloadCase(
                id="reasoning-short",
                messages=[
                    {
                        "role": "user",
                        "content": "A warehouse has 17 boxes with 24 parts each and ships 135 parts. How many remain? Give only the number.",
                    }
                ],
                max_tokens=32,
                tags=["reasoning"],
                checks=[{"type": "exact", "value": "273"}],
            ),
            WorkloadCase(
                id="structured-json",
                messages=[
                    {
                        "role": "user",
                        "content": "Return JSON with keys title, priority, and three string tags for: GPU server thermal alert.",
                    }
                ],
                max_tokens=96,
                tags=["structured-output"],
                extra_body={"response_format": {"type": "json_object"}},
                checks=[{"type": "json_keys", "keys": ["title", "priority", "tags"]}],
            ),
            WorkloadCase(
                id="multi-turn",
                messages=[
                    {"role": "system", "content": "You are a concise systems engineer."},
                    {"role": "user", "content": "What is continuous batching?"},
                    {
                        "role": "assistant",
                        "content": "It dynamically combines active inference requests at each decoding step.",
                    },
                    {"role": "user", "content": "Name its main latency benefit in one sentence."},
                ],
                max_tokens=64,
                tags=["multi-turn"],
            ),
        ]
    )
    return Suite(
        name="standard-text",
        description="Context/output sweeps plus reasoning, JSON, and multi-turn chat.",
        cases=cases,
    )


def smoke_suite() -> Suite:
    return Suite(
        name="smoke",
        description="Two cheap requests to validate server connectivity and streaming usage.",
        cases=[
            WorkloadCase(
                id="short-generation",
                messages=[
                    {
                        "role": "user",
                        "content": "In one sentence, explain why benchmarking matters.",
                    }
                ],
                max_tokens=32,
            ),
            WorkloadCase(
                id="medium-generation",
                messages=[{"role": "user", "content": sized_prompt(256)}],
                max_tokens=64,
            ),
        ],
    )


def vision_suite(image_url: str) -> Suite:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "Describe this image, list visible objects, and mention any readable text.",
        },
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    return Suite(
        name="standard-vision",
        description="Image understanding at two output lengths.",
        cases=[
            WorkloadCase(
                id="vision-concise",
                messages=[{"role": "user", "content": content}],
                max_tokens=64,
                tags=["vision", "concise"],
            ),
            WorkloadCase(
                id="vision-detailed",
                messages=[{"role": "user", "content": content}],
                max_tokens=256,
                tags=["vision", "detailed"],
            ),
        ],
    )


def built_in_suites() -> dict[str, Suite]:
    return {"smoke": smoke_suite(), "standard-text": standard_suite()}
