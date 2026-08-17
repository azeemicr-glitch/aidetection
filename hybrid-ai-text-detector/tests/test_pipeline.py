import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    ALL_ENGINEERED_FEATURES,
    analyse_text,
    count_words,
    extract_engineered_features,
    load_bundle,
)

MODEL_PATH = PROJECT_ROOT / "deployment_best_hybrid.joblib"

VALID_TEXT = """
The development of modern information systems has changed the way organisations
collect, process and communicate knowledge. A reliable analytical system should
not be judged only by its ability to produce a prediction, because the quality
of the underlying data and the assumptions used during evaluation are equally
important. Researchers therefore need to document preprocessing, preserve
independent test data, compare alternative models and report limitations when
performance changes across different datasets. This approach supports
reproducibility and makes the final conclusions easier to evaluate critically.
"""


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle(MODEL_PATH)

    def test_bundle_loads(self):
        self.assertEqual(self.bundle["feature_condition"], "Hybrid")
        self.assertEqual(self.bundle["model_name"], "Linear SVM")
        self.assertEqual(int(self.bundle["min_words"]), 50)

    def test_feature_order_matches_bundle(self):
        self.assertEqual(
            list(self.bundle["all_engineered_features"]),
            ALL_ENGINEERED_FEATURES,
        )

    def test_word_counter(self):
        self.assertGreaterEqual(count_words(VALID_TEXT), 50)

    def test_feature_extraction(self):
        features = extract_engineered_features(VALID_TEXT)
        self.assertEqual(set(features.keys()), set(ALL_ENGINEERED_FEATURES))
        self.assertGreater(features["word_count"], 0)
        self.assertGreater(features["sentence_count"], 0)

    def test_valid_prediction(self):
        result = analyse_text(VALID_TEXT, self.bundle)
        self.assertIn(result["prediction"], (0, 1))
        self.assertIn(result["prediction_name"], ("Human-written", "AI-generated"))
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        total_probability = sum(result["probabilities"].values())
        self.assertAlmostEqual(total_probability, 1.0, places=6)

    def test_short_text_rejected(self):
        with self.assertRaises(ValueError):
            analyse_text("This is far too short for a valid model analysis.", self.bundle)


if __name__ == "__main__":
    unittest.main()
