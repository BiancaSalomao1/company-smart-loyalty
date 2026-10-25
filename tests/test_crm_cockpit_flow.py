"""Teste de integração da jornada do Cockpit de CRM com AppTest do Streamlit."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class CRMCockpitFlowTests(unittest.TestCase):
    """Valida o fluxo do analista de CRM com estado de sessão e Human-in-the-Loop."""

    def test_crm_cockpit_full_journey(self):
        project_root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(project_root / "src" / "dashboard" / "app.py")
        app.run(timeout=30)

        # 1. Verifica se não houve nenhuma exceção não tratada
        self.assertEqual(len(app.exception), 0, f"Exceções encontradas: {app.exception}")

        # 2. Verifica se a aba do Cockpit de CRM e seções existem
        headers = [h.value for h in app.header]
        self.assertIn("6. Clientes sintéticos: valor, risco e ação", headers)

        # 3. Verifica se as métricas e subheaders estão presentes
        subheaders = [s.value for s in app.subheader]
        self.assertTrue(any("Matriz de Priorização" in s for s in subheaders))
        self.assertTrue(any("Ficha 360°" in s for s in subheaders))
        self.assertTrue(any("Agente Cognitivo" in s for s in subheaders))
        self.assertTrue(any("Fila de Ações Aprovadas" in s for s in subheaders))

        # 4. Verifica se os botões interativos foram renderizados
        button_labels = [b.label for b in app.button]
        self.assertTrue(any("Gerar Próxima Melhor Ação" in label for label in button_labels))


if __name__ == "__main__":
    unittest.main()
