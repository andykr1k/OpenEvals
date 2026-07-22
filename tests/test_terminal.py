import io
import unittest

from openevals.models import RequestResult, RunResult, WorkloadCase
from openevals.terminal import TerminalReporter


class TerminalReporterTests(unittest.TestCase):
    def test_only_first_request_in_stage_is_streamed(self):
        output = io.StringIO()
        reporter = TerminalReporter(output)
        case = WorkloadCase("hello", [{"role": "user", "content": "Hi"}])

        reporter.stream_chunk(1, case, 2, "content", "hidden")
        reporter.stream_chunk(0, case, 2, "reasoning", "think")
        reporter.stream_chunk(0, case, 2, "content", "answer")

        value = output.getvalue()
        self.assertNotIn("hidden", value)
        self.assertIn("Reasoning: think", value)
        self.assertIn("Response: answer", value)

    def test_result_renders_analytics(self):
        output = io.StringIO()
        reporter = TerminalReporter(output)
        result = RunResult(
            id="id",
            created_at="now",
            finished_at="now",
            status="completed",
            endpoint="http://localhost:8000",
            model="model",
            suite_name="suite",
            config={},
            requests=[
                RequestResult(
                    case_id="one",
                    concurrency=1,
                    started_at="now",
                    success=True,
                    latency_ms=10,
                )
            ],
            summary={
                "requests": 1,
                "successful": 1,
                "latency_ms": {"mean": 10, "p50": 10, "p95": 10, "p99": 10, "max": 10},
            },
        )

        reporter.print_result(result)

        value = output.getvalue()
        self.assertIn("Completed: model / suite", value)
        self.assertIn("1/1 successful", value)
        self.assertIn("Latency (ms)", value)


if __name__ == "__main__":
    unittest.main()
