import json
import unittest

from openevals.adapters.openai_compatible import OpenAICompatibleClient, StreamState


class AdapterParserTests(unittest.TestCase):
    def test_sse_captures_reasoning_content_and_usage(self):
        state = StreamState()
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "Think"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "Answer"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}},
        ]
        emitted = []
        for chunk in chunks:
            emitted.extend(
                OpenAICompatibleClient._consume_sse_line(f"data: {json.dumps(chunk)}", state)
            )
        self.assertEqual(state.reasoning, ["Think"])
        self.assertEqual(state.text, ["Answer"])
        self.assertEqual(len(state.token_times), 2)
        self.assertEqual(state.prompt_tokens, 9)
        self.assertEqual(state.output_tokens, 2)
        self.assertEqual(state.finish_reason, "stop")
        self.assertEqual(emitted, [("reasoning", "Think"), ("content", "Answer")])

    def test_non_data_and_done_lines_are_ignored(self):
        state = StreamState()
        OpenAICompatibleClient._consume_sse_line("event: message", state)
        OpenAICompatibleClient._consume_sse_line("data: [DONE]", state)
        self.assertEqual(state.token_times, [])


if __name__ == "__main__":
    unittest.main()
