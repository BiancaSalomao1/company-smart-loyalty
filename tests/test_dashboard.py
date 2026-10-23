import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class DashboardTests(unittest.TestCase):

    def test_dashboard_renders_customer_intelligence_section(self):
        project_root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(project_root / "src" / "dashboard" / "app.py")
        app.run(timeout=20)

        self.assertEqual(len(app.exception), 0)
        headers = [header.value for header in app.header]
        self.assertIn("6. Clientes sintéticos: valor, risco e ação", headers)
        self.assertIn("8. Diagnóstico com base documental", headers)
        self.assertIn("9. Influenciadores e parceiros de relacionamento", headers)


if __name__ == "__main__":
    unittest.main()
