"""Testes do módulo de ranqueamento de criadores e parceiros de relacionamento."""

import tempfile
import unittest
from pathlib import Path

from src.analysis.influencer_ranking import (
    calculate_influencer_ranking,
    save_influencer_dataset,
)


class InfluencerRankingTests(unittest.TestCase):
    def test_ranking_computes_normalized_scores_and_positions(self):
        df = calculate_influencer_ranking()
        self.assertGreater(len(df), 0)
        self.assertIn("fit_score", df.columns)
        self.assertIn("ranking", df.columns)
        self.assertTrue(df["fit_score"].between(0, 1).all())
        self.assertEqual(df.iloc[0]["ranking"], 1)

    def test_save_dataset_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "influenciadores_test.csv"
            df = save_influencer_dataset(output_path=target)
            self.assertTrue(target.exists())
            self.assertEqual(len(df), 5)


if __name__ == "__main__":
    unittest.main()
