import unittest

from openevals.metrics import describe, percentile, summarize
from openevals.models import RequestResult


def result(**overrides):
    values = {
        "case_id": "case",
        "concurrency": 1,
        "started_at": "2026-01-01T00:00:00+00:00",
        "success": True,
        "latency_ms": 100.0,
        "ttft_ms": 20.0,
        "generation_ms": 80.0,
        "prompt_tokens": 10,
        "output_tokens": 8,
        "prefill_tps": 500.0,
        "output_tps": 100.0,
        "total_tps": 180.0,
        "inter_token_ms": [10.0, 11.0],
    }
    values.update(overrides)
    return RequestResult(**values)


class MetricTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([0, 10], 0.5), 5)
        self.assertEqual(percentile([1], 0.99), 1)
        self.assertIsNone(percentile([], 0.5))

    def test_describe_has_tail_percentiles(self):
        stats = describe(range(1, 101))
        self.assertEqual(stats["count"], 100)
        self.assertEqual(stats["mean"], 50.5)
        self.assertAlmostEqual(stats["p99"], 99.01)

    def test_summary_groups_concurrency_and_throughput(self):
        requests = [result(), result(concurrency=2, output_tokens=12, latency_ms=200)]
        summary = summarize(requests, wall_time_seconds=2)
        self.assertEqual(summary["successful"], 2)
        self.assertEqual(summary["output_tokens"], 20)
        self.assertEqual(summary["output_throughput_tps"], 10)
        self.assertEqual(set(summary["by_concurrency"]), {"1", "2"})

    def test_failure_error_is_preserved_without_status(self):
        summary = summarize([result(success=False, error="connection refused", status_code=None)])
        self.assertEqual(summary["errors"], {"connection refused": 1})


if __name__ == "__main__":
    unittest.main()
