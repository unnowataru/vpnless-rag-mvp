from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.api_input_normalizer import build_dify_options_payload
from core.api_input_normalizer import build_openai_options_payload
from core.api_input_normalizer import extract_dify_question
from core.api_input_normalizer import extract_openai_question


class ApiInputNormalizerTests(unittest.TestCase):
    def test_extract_openai_question_from_latest_user_message(self) -> None:
        question = extract_openai_question(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ack"},
                {"role": "user", "content": [{"type": "text", "text": "final question"}]},
            ]
        )
        self.assertEqual(question, "final question")

    def test_extract_openai_question_raises_without_user_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "messages"):
            extract_openai_question([{"role": "assistant", "content": "only assistant"}])

    def test_build_openai_options_payload_maps_model(self) -> None:
        payload = {
            "model": "cost",
            "top_k": 3,
            "allow_unscoped": True,
        }
        options = build_openai_options_payload(payload, {"cost": "dummy.model"})
        self.assertEqual(options["answer_profile"], "cost")
        self.assertNotIn("bedrock_model", options)
        self.assertEqual(options["top_k"], 3)

    def test_build_openai_options_payload_maps_non_profile_model_to_bedrock(self) -> None:
        payload = {"model": "custom.model"}
        options = build_openai_options_payload(payload, {"cost": "dummy.model"})
        self.assertEqual(options["bedrock_model"], "custom.model")
        self.assertNotIn("answer_profile", options)

    def test_extract_dify_question_uses_payload_or_inputs(self) -> None:
        question = extract_dify_question({"inputs": {"query": "from inputs"}})
        self.assertEqual(question, "from inputs")
        with self.assertRaisesRegex(ValueError, "question"):
            extract_dify_question({"inputs": {}})

    def test_build_dify_options_payload_prefers_payload_then_inputs(self) -> None:
        payload = {"top_k": 5}
        inputs = {"answer_profile": "cost", "request_id": "abc"}
        options = build_dify_options_payload(payload, inputs)
        self.assertEqual(options["top_k"], 5)
        self.assertEqual(options["answer_profile"], "cost")
        self.assertEqual(options["request_id"], "abc")


if __name__ == "__main__":
    unittest.main()
