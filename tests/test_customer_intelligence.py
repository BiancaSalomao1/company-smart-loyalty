import unittest

from src.analysis.build_customer_intelligence import build_intelligence
from src.synthetic.generate_passengers import generate_passengers


class CustomerIntelligenceTests(unittest.TestCase):
    def test_generation_is_reproducible(self):
        first = generate_passengers(size=20, seed=7)
        second = generate_passengers(size=20, seed=7)
        self.assertTrue(first.equals(second))
        self.assertTrue((first["data_type"] == "synthetic").all())

    def test_profiles_include_explainable_scores_and_actions(self):
        profiles, metrics = build_intelligence(generate_passengers(size=100, seed=9))
        self.assertEqual(len(profiles), 100)
        self.assertTrue(profiles["churn_risk_score"].between(0, 1).all())
        self.assertGreaterEqual(profiles["segment"].nunique(), 2)
        self.assertTrue(profiles["recommended_action"].notna().all())
        self.assertTrue(profiles["risk_explanation"].notna().all())
        self.assertTrue(profiles["draft_message"].notna().all())
        self.assertEqual(metrics["data_type"], "synthetic")
        self.assertEqual(metrics["test_records"], 20)


if __name__ == "__main__":
    unittest.main()
