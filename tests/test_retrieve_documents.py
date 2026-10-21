import tempfile
import unittest
from pathlib import Path

from src.knowledge.retrieve_documents import (
    load_documents,
    search_documents,
    split_document,
    tokenize,
)


class RetrieveDocumentsTests(unittest.TestCase):

    def test_tokenize_ignores_accents_and_stopwords(self):
        self.assertEqual(
            tokenize("Benefícios de fidelidade"),
            ["beneficios", "fidelidade"],
        )

    def test_split_document_preserves_section(self):
        chunks = split_document(
            "# Benefícios\n\nEmbarque prioritário.",
            "exemplo.md",
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["section"], "Benefícios")
        self.assertIn("Embarque prioritário", chunks[0]["text"])

    def test_search_returns_relevant_evidence(self):
        documents = [
            {
                "source": "fidelidade.md",
                "section": "Benefícios",
                "text": "O programa oferece embarque prioritário.",
            },
            {
                "source": "fidelidade.md",
                "section": "Pontos",
                "text": "Os pontos podem ser usados em produtos.",
            },
        ]

        results = search_documents("embarque prioritário", documents)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["section"], "Benefícios")
        self.assertGreater(results[0]["score"], 0)

    def test_empty_documents_are_ignored(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "empty.md").write_text("   ", encoding="utf-8")
            (root / "README.md").write_text(
                "# Instruções", encoding="utf-8"
            )

            self.assertEqual(load_documents(root), [])

    def test_empty_query_returns_no_results(self):
        self.assertEqual(
            search_documents("de e para", []),
            [],
        )

    def test_invalid_top_k_is_rejected(self):
        with self.assertRaises(ValueError):
            search_documents("fidelidade", [], top_k=0)


        if __name__ == "__main__":
            unittest.main()



    def test_split_document_keeps_short_benefits_list_together(self):
        chunks = split_document(
            "# Benefícios\n\n"
            "- Embarque prioritário.\n"
            "- Bagagem gratuita.\n"
            "- Acesso ao lounge.\n",
            "beneficios.md",
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["section"], "Benefícios")
        self.assertIn("Embarque prioritário", chunks[0]["text"])
        self.assertIn("Bagagem gratuita", chunks[0]["text"])
        self.assertIn("Acesso ao lounge", chunks[0]["text"])
