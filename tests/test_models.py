import unittest

from openevals.models import BenchmarkConfig, RunResult, Suite, WorkloadCase


class ModelTests(unittest.TestCase):
    def test_prompt_shorthand_becomes_chat_message(self):
        case = WorkloadCase.from_dict({"id": "one", "prompt": "hello"})
        self.assertEqual(case.messages, [{"role": "user", "content": "hello"}])

    def test_api_key_is_redacted_from_in_memory_result(self):
        config = BenchmarkConfig(
            endpoint="http://localhost:8000",
            model="model",
            api_key="secret",
            suite=Suite("test", cases=[WorkloadCase("one", [{"role": "user", "content": "hi"}])]),
        )
        run = RunResult.create(config)
        self.assertEqual(run.config["api_key"], "***")


if __name__ == "__main__":
    unittest.main()
