import unittest

from openevals.evaluators import evaluate_checks


class EvaluatorTests(unittest.TestCase):
    def test_text_checks(self):
        checks = [
            {"type": "contains", "value": "WORLD"},
            {"type": "regex", "pattern": r"hello\s+world"},
            {"type": "max_length", "characters": 20},
        ]
        self.assertTrue(all(item["passed"] for item in evaluate_checks("hello world", checks)))

    def test_json_keys(self):
        results = evaluate_checks(
            '{"title":"alert","tags":[]}', [{"type": "json_keys", "keys": ["title", "tags"]}]
        )
        self.assertTrue(results[0]["passed"])

    def test_invalid_json_fails_without_raising(self):
        results = evaluate_checks("not json", [{"type": "json"}])
        self.assertFalse(results[0]["passed"])
        self.assertTrue(results[0]["detail"])


if __name__ == "__main__":
    unittest.main()
