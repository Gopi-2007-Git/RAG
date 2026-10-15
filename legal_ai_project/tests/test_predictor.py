"""
Unit tests for the Multi-Case Legal Outcome Analysis and AI Prediction Engine.
Validates weighted outcome synthesis, court hierarchy authority, anti-hallucination,
uncertainty handling, and explainable report generation.
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from pipeline.config import PREDICTION_DATASET_CSV, EXCLUDED_DIR
from pipeline.predictor import LegalCasePredictor, OUTCOME_DISPLAY_NAMES


class TestLegalCasePredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = LegalCasePredictor()

    def test_feature_extraction(self):
        """Test extraction of factual themes and statutory provisions from case description."""
        text = (
            "The claimant suffered 50% permanent disability in a head-on lorry collision. "
            "A compensation claim under Section 166 of the Motor Vehicles Act was filed against the driver and insurer."
        )
        features = self.predictor.extract_features_from_text(text)

        self.assertIn("permanent_disability", features["facts"])
        self.assertIn("negligence_dispute", features["facts"])
        self.assertIn("insurance_liability", features["facts"])
        self.assertIn("compensation_quantum", features["facts"])
        self.assertTrue(any("Sec 166" in s for s in features["statutes"]))

    def test_court_authority_hierarchy(self):
        """Verify judicial authority weights (Supreme Court > High Court > Tribunal)."""
        sc_weight = self.predictor.get_court_authority("Supreme Court of India")
        hc_weight = self.predictor.get_court_authority("Karnataka High Court")
        tribunal_weight = self.predictor.get_court_authority("Motor Accidents Claims Tribunal")

        self.assertGreater(sc_weight, hc_weight)
        self.assertGreater(hc_weight, tribunal_weight)

    def test_weighted_synthesis_not_simple_majority(self):
        """Verify that a high-similarity case with strong court authority overrides weaker cases."""
        new_features = {
            "facts": {"permanent_disability", "compensation_quantum"},
            "statutes": {"MVA 1988 - Sec 166 (Compensation Claim)"}
        }

        # Mock retrieved precedents:
        # Case 1: L2 dist 0.70 (Supreme Court), appeal_allowed
        # Case 2: L2 dist 1.50 (Tribunal), appeal_dismissed
        # Case 3: L2 dist 1.55 (Tribunal), appeal_dismissed
        mock_precedents = [
            {
                "case_id": "case_sc",
                "case_title": "SC Precedent vs Insurer",
                "court": "Supreme Court of India",
                "judgment_date": "2020-01-01",
                "distance": 0.70,
                "decision": "appeal_allowed",
                "decision_evidence": "Appeal is allowed with compensation.",
                "legal_sections": {"Motor Vehicles Act, 1988": ["Section 166"]}
            },
            {
                "case_id": "case_trib_1",
                "case_title": "Tribunal Precedent 1",
                "court": "Motor Accidents Claims Tribunal",
                "judgment_date": "2010-01-01",
                "distance": 1.50,
                "decision": "appeal_dismissed",
                "decision_evidence": "Dismissed.",
                "legal_sections": {}
            },
            {
                "case_id": "case_trib_2",
                "case_title": "Tribunal Precedent 2",
                "court": "Motor Accidents Claims Tribunal",
                "judgment_date": "2011-01-01",
                "distance": 1.55,
                "decision": "appeal_dismissed",
                "decision_evidence": "Dismissed.",
                "legal_sections": {}
            }
        ]

        synthesis = self.predictor.synthesize_outcome(new_features, mock_precedents)

        # In simple majority, dismissed would win (2 vs 1).
        # In weighted synthesis, SC with distance 0.70 has higher weighted score.
        self.assertEqual(synthesis["predicted_outcome"], "appeal_allowed")
        self.assertGreater(synthesis["outcome_scores"]["appeal_allowed"], synthesis["outcome_scores"]["appeal_dismissed"])

    def test_unable_to_predict_on_distant_evidence(self):
        """Verify that if nearest precedent is too distant (> 1.85 L2), returns unable_to_predict."""
        new_features = {"facts": set(), "statutes": set()}
        distant_precedents = [
            {
                "case_id": "case_distant",
                "case_title": "Distant Case",
                "court": "High Court",
                "distance": 2.10,
                "decision": "appeal_allowed",
                "decision_evidence": "Allowed."
            }
        ]

        synthesis = self.predictor.synthesize_outcome(new_features, distant_precedents)
        self.assertEqual(synthesis["predicted_outcome"], "unable_to_predict")
        self.assertEqual(synthesis["evidence_strength"], "Insufficient")

    def test_unable_to_predict_on_empty_precedents(self):
        """Verify that empty precedent list returns unable_to_predict."""
        new_features = {"facts": set(), "statutes": set()}
        synthesis = self.predictor.synthesize_outcome(new_features, [])
        self.assertEqual(synthesis["predicted_outcome"], "unable_to_predict")

    def test_end_to_end_prediction_and_report(self):
        """Test full prediction workflow on a realistic motor accident query."""
        query = (
            "A passenger sustained severe injuries in a bus collision and claims "
            "compensation under Section 166 of the Motor Vehicles Act against the state transport corporation."
        )
        res = self.predictor.predict(query, top_k=3)

        self.assertIn("predicted_outcome", res)
        self.assertIn("evidence_strength", res)
        self.assertIn("retrieved_precedents", res)
        self.assertIn("legal_disclaimer", res)
        self.assertIn("limitations", res)

        # Disclaimer must be present
        self.assertIn("DISCLAIMER:", res["legal_disclaimer"])
        self.assertIn("does NOT provide legal advice", res["legal_disclaimer"])

        # Precedents must be from prediction dataset
        self.assertGreater(len(res["retrieved_precedents"]), 0)

        # Formatting report test
        report_text = self.predictor.format_prediction_report(res)
        self.assertIn("AI-POWERED LEGAL OUTCOME ANALYSIS", report_text)
        self.assertIn("PREDICTED OUTCOME", report_text)
        self.assertIn("MANDATORY LEGAL DISCLAIMER", report_text)

    def test_excluded_cases_never_in_prediction(self):
        """Verify that excluded cases cannot appear in prediction evidence."""
        query = "Motor accident case"
        res = self.predictor.predict(query, top_k=5)

        excluded_ids = [f.stem for f in EXCLUDED_DIR.glob("*.json")]
        for p in res["retrieved_precedents"]:
            self.assertNotIn(p["case_id"], excluded_ids, f"Excluded case {p['case_id']} leaked into prediction!")


if __name__ == "__main__":
    unittest.main()
