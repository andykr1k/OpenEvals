# Open Evals

Open Evals is a terminal-first benchmark tool for LLMs and vision-language models
(VLMs) served through an OpenAI-compatible API. It runs repeatable workloads against
local inference servers, streams a readable response sample, and reports latency,
throughput, token, correctness, and optional hardware metrics.

It is intentionally small: no web UI, database, background service, or hosted account is
required. Your prompts and model responses are sent only to the endpoint you configure.

> [!NOTE]
> Open Evals is early-stage software. Benchmark configuration and result fields may evolve
> before a stable 1.0 release.

## Why Open Evals?

- **Local-first:** point it at a model running on your own machine or network.
- **LLM and VLM workloads:** send ordinary chat messages or OpenAI-style typed image content.
- **Real streaming measurements:** capture time to first token, generation time, chunk gaps,
  and end-to-end latency.
- **Concurrency sweeps:** compare latency and aggregate throughput as load increases.
- **Repeatable quality checks:** validate exact text, substrings, regexes, JSON, JSON keys, and
  maximum response length.
- **Useful telemetry:** include vLLM Prometheus counter deltas and NVIDIA GPU samples when
  those sources are available.
- **Human and machine output:** watch one live response per stage or emit the complete run as
  JSON for further analysis.

## Requirements

- Python 3.11 or newer
- An OpenAI-compatible Chat Completions endpoint
- [`uv`](https://docs.astral.sh/uv/) for the documented development workflow

The benchmark client does not install or start a model server for you. It is designed around
vLLM and should work with other servers that implement streaming `POST
/v1/chat/completions`. See [Compatibility and measurement notes](#compatibility-and-measurement-notes).

## Quick start

Install the locked development environment from a checkout:

```bash
uv sync --locked
```

Start an OpenAI-compatible model server in another terminal. For example, with vLLM:

```bash
uvx --from vllm vllm serve Qwen/Qwen3-8B-AWQ \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 16384 \
  --max-num-seqs 16
```

Run a three-request smoke benchmark:

```bash
uv run open-evals run --model Qwen/Qwen3-8B-AWQ
```

Or sweep multiple concurrency levels:

```bash
uv run open-evals run \
  --model Qwen/Qwen3-8B-AWQ \
  --prompt "Explain why the sky is blue in two sentences." \
  --requests 5 \
  --concurrency 1 2 4 8
```

If the endpoint requires authentication, set the key without putting it in shell history or a
committed configuration:

```bash
export OPEN_EVALS_API_KEY="your-api-key"
```

The environment variable is used by command-line runs. Configuration files may also contain
an `api_key`; it is redacted from serialized run configuration in JSON results.

## Installation options

For development, use `uv sync --locked` and prefix commands with `uv run`. To install the CLI
from a local checkout instead:

```bash
uv tool install .
open-evals --help
```

To remove that tool installation, run `uv tool uninstall open-evals`.

## CLI

### Run a quick benchmark

```text
open-evals run [CONFIG] [OPTIONS]
```

Without `CONFIG`, `--model` is required. Useful options include:

| Option | Default | Purpose |
|---|---:|---|
| `--endpoint` | `http://127.0.0.1:8000` | API server base URL, with or without `/v1` |
| `--model` | required | Model name expected by the server |
| `--prompt` | short greeting | Text prompt for the generated one-case suite |
| `--requests` | `3` | Repetitions per concurrency stage |
| `--concurrency` | `1` | One or more concurrency levels |
| `--max-tokens` | `64` | Maximum generated tokens per request |
| `--warmups` | `1` | Unmeasured warmup requests |
| `--timeout` | `300` | Request timeout in seconds |
| `--metrics-endpoint` | unset | Optional Prometheus endpoint, such as vLLM `/metrics` |
| `--no-stream` | off | Hide the live response sample |
| `--json` | off | Print the complete machine-readable result |

When `CONFIG` is supplied, workload settings come from that file. `--json` and `--no-stream`
still control output.

Examples:

```bash
# Analytics only
uv run open-evals run examples/smoke.json --no-stream

# Complete JSON result; live output is disabled automatically
uv run open-evals run examples/smoke.json --json > result.json
```

### Create a configuration

Generate an editable standard text benchmark:

```bash
uv run open-evals init benchmark.json --model Qwen/Qwen3-8B-AWQ
uv run open-evals run benchmark.json
```

The generated suite covers prompt-size and output-size sweeps plus reasoning, structured JSON,
and multi-turn chat. See [`examples/smoke.json`](examples/smoke.json) and
[`examples/vision.json`](examples/vision.json) for smaller text and VLM examples.

### Inspect built-in suites

```bash
uv run open-evals suites
```

This lists the suites shipped in the package. The `init` command currently writes the standard
text suite; JSON configurations remain the public interface for customized workloads.

## Configuration

Benchmark configurations are JSON documents:

```json
{
  "endpoint": "http://127.0.0.1:8000",
  "model": "my-served-model-name",
  "api_key": "EMPTY",
  "concurrency": [1, 2, 4, 8],
  "repetitions": 3,
  "warmup_requests": 2,
  "timeout_seconds": 300,
  "request_rate": null,
  "metrics_endpoint": "http://127.0.0.1:8000/metrics",
  "metadata": {"tensor_parallel": 1, "dtype": "bfloat16"},
  "suite": {
    "name": "my-suite",
    "description": "Structured-output smoke test",
    "cases": [
      {
        "id": "json-alert",
        "prompt": "Return a JSON alert with title and severity.",
        "max_tokens": 96,
        "temperature": 0,
        "extra_body": {"response_format": {"type": "json_object"}},
        "checks": [{"type": "json_keys", "keys": ["title", "severity"]}],
        "tags": ["json"]
      }
    ]
  }
}
```

### Top-level fields

| Field | Type | Meaning |
|---|---|---|
| `endpoint` | string | Server base URL |
| `model` | string | Served model name sent in every request |
| `suite` | object | Suite metadata and workload cases |
| `api_key` | string | Bearer token; defaults to `EMPTY` |
| `concurrency` | integer array | Sequential concurrency stages |
| `repetitions` | integer | Repetitions of every case at each stage |
| `warmup_requests` | integer | Unmeasured requests before all stages |
| `timeout_seconds` | number | Overall HTTP request timeout |
| `request_rate` | number or null | Optional request starts per second |
| `metrics_endpoint` | string or null | Optional Prometheus scrape URL |
| `metadata` | object | User-defined context copied into the result configuration |

Each case accepts `messages`, `max_tokens`, `temperature`, `top_p`, `stop`, `tags`, `checks`,
and `extra_body`. As a convenience, a string `prompt` is converted to one user message.
`extra_body` is merged into the outgoing request body, so server-specific options can be used
without changes to Open Evals.

### Deterministic checks

```json
{"type": "exact", "value": "expected", "case_sensitive": true}
{"type": "contains", "value": "needle", "case_sensitive": false}
{"type": "regex", "pattern": "^answer: .+$"}
{"type": "json"}
{"type": "json_keys", "keys": ["one", "two"]}
{"type": "max_length", "characters": 500}
```

Checks are evaluated only for successful HTTP requests. They are reported separately from
transport success, making it possible to distinguish server errors from incorrect responses.

### Vision-language requests

Typed message content is passed through unchanged:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image."},
        {
          "type": "image_url",
          "image_url": {"url": "data:image/jpeg;base64,..."}
        }
      ]
    }
  ]
}
```

Use a URL reachable by the model server or a data URI. Be mindful that embedding images as data
URIs makes configuration and JSON result files large.

## Metrics

| Metric | Definition |
|---|---|
| TTFT | Request start to the first non-empty streamed content or reasoning chunk |
| Prefill tokens/s | Prompt tokens divided by TTFT |
| Generation time | First emitted chunk to the end of the stream |
| Output tokens/s | `(completion_tokens - 1) / generation_time` |
| Inter-token latency | Wall-clock gap between emitted stream chunks |
| End-to-end latency | Request start through completion of the response stream |
| Output throughput | Successful completion tokens divided by measured wall time |
| Request throughput | Successful requests divided by measured wall time |

Open Evals reports mean, minimum, p50, p90, p95, p99, and maximum distributions where relevant.
Warmups are excluded. If `request_rate` is omitted, requests begin as quickly as each concurrency
limit permits; otherwise request starts are evenly spaced at the configured rate.

### Compatibility and measurement notes

- The server must support streamed Chat Completions using server-sent events (`data:` lines).
- Exact prompt and completion counts come from the final streamed `usage` object when supplied.
  If completion usage is absent, emitted chunk count is used as a compatibility fallback; this
  is an estimate because one chunk can contain more than one token.
- For the same reason, inter-token latency is technically inter-chunk latency for servers that
  batch multiple tokens into a chunk.
- Reasoning is captured from either `reasoning_content` or `reasoning` delta fields.
- Prometheus scrape failures and missing `nvidia-smi` are treated as unavailable telemetry, not
  benchmark failures.
- Compare runs using the same model, quantization, prompt set, server configuration, hardware,
  and warmup policy. Percentiles from very small samples are descriptive, not statistically
  robust.

## Development

```bash
uv sync --locked
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
uv lock --check
```

Unit tests do not require a GPU or a running model server. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.

## Security

Do not commit API keys, private prompts, model responses, or benchmark configurations containing
sensitive data. Please report vulnerabilities through the process in [`SECURITY.md`](SECURITY.md),
not through a public issue.

## License

Open Evals is available under the [MIT License](LICENSE).
