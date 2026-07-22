import unittest
from unittest.mock import patch

from openevals.models import BenchmarkConfig, RequestResult, Suite, WorkloadCase, utc_now
from openevals.runner import BenchmarkRunner


class FakeClient:
    calls = []

    def __init__(self, *_args, **_kwargs):
        FakeClient.calls = []

    async def request(self, case, concurrency, on_stream=None):
        FakeClient.calls.append((case.id, concurrency))
        if on_stream:
            await on_stream("content", "ok")
        return RequestResult(
            case_id=case.id,
            concurrency=concurrency,
            started_at=utc_now(),
            success=True,
            latency_ms=10,
            ttft_ms=2,
            generation_ms=8,
            prompt_tokens=4,
            output_tokens=2,
            prefill_tps=2000,
            output_tps=125,
            response_text="ok",
        )

    async def aclose(self):
        return None


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_sweep_repetition_warmup_and_checks(self):
        suite = Suite(
            "runner-test",
            cases=[
                WorkloadCase(
                    "one",
                    [{"role": "user", "content": "one"}],
                    checks=[{"type": "exact", "value": "ok"}],
                ),
                WorkloadCase("two", [{"role": "user", "content": "two"}]),
            ],
        )
        config = BenchmarkConfig(
            endpoint="http://localhost:8000",
            model="test",
            suite=suite,
            concurrency=[1, 2],
            repetitions=2,
            warmup_requests=1,
        )
        with patch("openevals.runner.OpenAICompatibleClient", FakeClient):
            streamed = []

            async def on_stream(index, case, concurrency, kind, text):
                streamed.append((index, case.id, concurrency, kind, text))

            result = await BenchmarkRunner().run(config, on_stream=on_stream)
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(FakeClient.calls), 9)  # one warmup + eight measured requests
        self.assertEqual(len(result.requests), 8)
        self.assertEqual(set(result.summary["by_concurrency"]), {"1", "2"})
        checked = [item for item in result.requests if item.case_id == "one"]
        self.assertTrue(all(item.checks_passed == 1 for item in checked))
        self.assertEqual(len(streamed), 8)
        self.assertEqual(streamed[0][3:], ("content", "ok"))

    async def test_invalid_config_is_rejected(self):
        config = BenchmarkConfig(
            endpoint="http://localhost:8000",
            model="test",
            suite=Suite("empty"),
        )
        with self.assertRaisesRegex(ValueError, "no cases"):
            await BenchmarkRunner().run(config)


if __name__ == "__main__":
    unittest.main()
