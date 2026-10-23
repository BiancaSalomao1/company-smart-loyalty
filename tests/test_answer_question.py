"""Testes do orquestrador de respostas sem chamadas externas."""

import os
import unittest
from unittest.mock import patch

from src.knowledge.answer_question import (
    answer_question,
    build_evidence_context,
)


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": "Resposta fundamentada [E1]."})()


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


class AnswerQuestionTests(unittest.TestCase):

    def setUp(self):
        self.documents = [
            {
                "source": "fidelidade.md",
                "section": "Benefícios",
                "text": "O programa menciona embarque prioritário.",
            }
        ]

    def test_build_evidence_context_adds_traceable_identifiers(self):
        context = build_evidence_context(
            [{**self.documents[0], "score": 1.0}]
        )

        self.assertIn("[E1]", context)
        self.assertIn("Documento: fidelidade.md", context)
        self.assertIn("Seção: Benefícios", context)

    def test_question_with_no_evidence_does_not_call_api(self):
        client = FakeClient()

        with patch(
            "src.knowledge.answer_question.load_documents",
            return_value=self.documents,
        ):
            result = answer_question("benefício inexistente", client=client)

        self.assertEqual(result["sources"], [])
        self.assertIn("Não encontrei evidências", result["answer"])
        self.assertEqual(client.responses.calls, [])

    def test_question_sends_recovered_evidence_to_injected_client(self):
        client = FakeClient()

        with patch(
            "src.knowledge.answer_question.load_documents",
            return_value=self.documents,
        ):
            result = answer_question("embarque prioritário", client=client)

        self.assertEqual(result["answer"], "Resposta fundamentada [E1].")
        self.assertEqual(result["sources"][0]["id"], "E1")
        request = client.responses.calls[0]
        self.assertIn("[E1]", request["input"])
        self.assertIn("embarque prioritário", request["input"])

    def test_question_rejects_blank_input(self):
        with self.assertRaises(ValueError):
            answer_question("   ", client=FakeClient())

    def test_api_key_is_required_only_when_evidence_was_found(self):
        old_value = os.environ.pop("OPENAI_API_KEY", None)
        self.addCleanup(self._restore_api_key, old_value)

        with patch(
            "src.knowledge.answer_question.load_documents",
            return_value=self.documents,
        ):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                answer_question("embarque prioritário")

    def test_api_call_failure_raises_runtime_error(self):
        class FailingResponses:
            def create(self, **kwargs):
                raise ConnectionError("Timeout na API")

        class FailingClient:
            def __init__(self):
                self.responses = FailingResponses()

        with patch(
            "src.knowledge.answer_question.load_documents",
            return_value=self.documents,
        ):
            with self.assertRaisesRegex(RuntimeError, "Falha na comunicação com o provedor de IA"):
                answer_question("embarque prioritário", client=FailingClient())

    def test_empty_api_response_returns_fallback_message(self):
        class EmptyResponses:
            def create(self, **kwargs):
                return type("Response", (), {"output_text": ""})()

        class EmptyClient:
            def __init__(self):
                self.responses = EmptyResponses()

        with patch(
            "src.knowledge.answer_question.load_documents",
            return_value=self.documents,
        ):
            result = answer_question("embarque prioritário", client=EmptyClient())
            self.assertIn("não retornou conteúdo", result["answer"])
            self.assertEqual(len(result["sources"]), 1)

    @staticmethod
    def _restore_api_key(value):
        if value is not None:
            os.environ["OPENAI_API_KEY"] = value


if __name__ == "__main__":
    unittest.main()
