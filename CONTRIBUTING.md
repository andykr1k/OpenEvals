# Contributing to Open Evals

Thanks for helping make local model evaluation more useful and reproducible. Bug reports,
documentation improvements, benchmark cases, compatibility fixes, and focused features are all
welcome.

## Before you start

- Search existing issues before opening a new one.
- For a substantial feature or behavior change, open an issue first so the design can be agreed
  on before significant work begins.
- Never include private prompts, API keys, proprietary model output, or other sensitive benchmark
  data in an issue or pull request.
- Keep changes focused. Separate unrelated fixes into separate pull requests.

By participating, you agree to follow the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Development setup

Open Evals requires Python 3.11 or newer and uses
[`uv`](https://docs.astral.sh/uv/) to manage its locked environment.

```bash
git clone <your-fork-url>
cd OpenEvals
uv sync --locked
```

Run the checks used by CI:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
uv lock --check
```

The unit test suite must remain runnable without a GPU, model download, network connection, or
live inference server. Mock protocol and telemetry behavior in tests where practical.

## Making a change

1. Create a branch from `main`.
2. Add or update tests for behavior changes.
3. Update the README or examples when the CLI, configuration, or output schema changes.
4. Run the complete check suite.
5. Open a pull request explaining the problem, approach, and verification performed.

Use clear commit messages and avoid committing generated files such as virtual environments,
build artifacts, caches, benchmark databases, or secret-bearing result files.

## Design principles

Contributions should preserve these project goals:

- terminal-first and useful without a hosted service;
- compatible with local OpenAI-style inference APIs;
- transparent metric definitions and honest compatibility fallbacks;
- repeatable workloads with machine-readable results;
- minimal required dependencies;
- graceful behavior when optional GPU and Prometheus telemetry are unavailable.

## Reporting benchmark or compatibility bugs

Include enough non-sensitive detail to reproduce the problem:

- Open Evals and Python versions;
- operating system;
- inference server and version;
- sanitized command or configuration;
- expected and actual behavior;
- a minimal sanitized SSE response when the issue concerns streaming compatibility.

Do not publicly report security vulnerabilities. Follow [`SECURITY.md`](SECURITY.md) instead.

## License

By submitting a contribution, you agree that it may be distributed under the project's
[MIT License](LICENSE).
