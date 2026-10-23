import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ingestion.update_anac_dataset import update_dataset, validate_csv


VALID_CSV = '''atualizado em 2026-01-01
"Nome Fantasia";"Data Abertura";"Grupo Problema";"Problema";"Avaliação Reclamação"
"Company Linhas Aéreas";"2025-02-10";"Atendimento";"Demora";"Não Resolvida"
'''.encode()


class FakeResponse:
    content = VALID_CSV
    headers = {"Content-Type": "text/csv"}

    def raise_for_status(self):
        return None


class UpdateAnacDatasetTests(unittest.TestCase):
    def test_validation_rejects_wrong_year_without_writing(self):
        with self.assertRaisesRegex(ValueError, "ano solicitado"):
            validate_csv(VALID_CSV, 2024)

    @patch("src.ingestion.update_anac_dataset.requests.get", return_value=FakeResponse())
    def test_update_writes_only_validated_dataset(self, _request):
        with tempfile.TemporaryDirectory() as temporary_dir:
            result = update_dataset(2025, output_dir=Path(temporary_dir))
            self.assertEqual(result["rows_validated"], 1)
            self.assertTrue((Path(temporary_dir) / "dadosconsumidor2025.csv").exists())
            self.assertTrue((Path(temporary_dir) / "dadosconsumidor2025_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
