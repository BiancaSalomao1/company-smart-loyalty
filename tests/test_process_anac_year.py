import tempfile
import unittest
from pathlib import Path

from src.analysis.process_anac_year import process_year


CSV = '''atualizado em 2025-12-31
"Nome Fantasia";"Data Abertura";"Grupo Problema";"Problema";"Respondida";"Avaliação Reclamação";"Nota do Consumidor";"Tempo Resposta (em dias)"
"Company Linhas Aéreas";"2025-02-10";"Atendimento";"Demora";"S";"Não Resolvida";"2";"4"
"Company Linhas Aéreas";"2025-03-10";"Atendimento";"Demora";"S";"Resolvida";"4";"2"
'''


class ProcessAnacYearTests(unittest.TestCase):
    def test_process_creates_year_partitioned_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text(CSV, encoding="utf-8")
            result = process_year(2025, "Company Linhas Aéreas", source, root)
            self.assertEqual(result["records"], 2)
            self.assertTrue((root / "company_indicadores_2025.csv").exists())
            self.assertTrue((root / "company_recomendacoes_2025.csv").exists())


if __name__ == "__main__":
    unittest.main()
