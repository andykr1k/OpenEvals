import unittest

from openevals.telemetry import metric_delta, parse_prometheus


class TelemetryTests(unittest.TestCase):
    def test_prometheus_workers_are_aggregated(self):
        text = """
# HELP vllm:prompt_tokens_total prompt tokens
vllm:prompt_tokens_total{model_name="m",worker="0"} 12
vllm:prompt_tokens_total{model_name="m",worker="1"} 8
vllm:num_requests_running{model_name="m"} 2
"""
        values = parse_prometheus(text)
        self.assertEqual(values["vllm:prompt_tokens_total"], 20)
        self.assertEqual(values["vllm:num_requests_running"], 2)

    def test_delta_filters_to_run_counters(self):
        delta = metric_delta(
            {"vllm:prompt_tokens_total": 10, "vllm:num_requests_running": 1},
            {"vllm:prompt_tokens_total": 35, "vllm:num_requests_running": 0},
        )
        self.assertEqual(delta, {"vllm:prompt_tokens_total": 25})


if __name__ == "__main__":
    unittest.main()
