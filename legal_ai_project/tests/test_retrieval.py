"""
Unit tests for the Legal Knowledge Base and Semantic Case Retrieval Engine.
Validates strict prediction-dataset filtering, FAISS IndexFlatL2 configuration,
1-to-1 vector mapping, and semantic retrieval accuracy.
"""

import json
import unittest
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from pipeline.config import (
    DATABASE_DIR,
    PREDICTION_DATASET_CSV,
    EXCLUDED_DIR
)
from database.vector_store import LegalVectorKnowledgeBase, INDEX_FILE, CASE_MAPPING_FILE
from pipeline.search_cases import LegalCaseRetriever


class TestLegalKnowledgeBaseAndRetrieval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure knowledge base is built
        kb = LegalVectorKnowledgeBase()
        kb.build_or_rebuild(force_reembed=False)
        cls.retriever = LegalCaseRetriever()

    def test_strict_validated_cases_only(self):
        """Verify that the knowledge base contains ONLY usable prediction cases and ZERO excluded cases."""
        df_pred = pd.read_csv(PREDICTION_DATASET_CSV)
        usable_ids = set(df_pred[df_pred["processing_status"] == "usable"]["case_id"].tolist())

        with open(CASE_MAPPING_FILE, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

        indexed_ids = [c["case_id"] for c in mapping_data["cases"]]

        # Every indexed case must be in the prediction dataset
        for cid in indexed_ids:
            self.assertIn(cid, usable_ids, f"Case {cid} is in knowledge base but not in prediction dataset!")

        self.assertEqual(len(indexed_ids), len(usable_ids))

        # Ensure excluded cases are strictly absent
        excluded_files = list(EXCLUDED_DIR.glob("*.json"))
        excluded_ids = [f.stem for f in excluded_files]
        for ex_id in excluded_ids:
            self.assertNotIn(ex_id, indexed_ids, f"Excluded case {ex_id} was improperly indexed into FAISS!")

    def test_faiss_index_dimension_and_count(self):
        """Verify that the FAISS index is IndexFlatL2 with dimension 384 and exact count match."""
        index = faiss.read_index(str(INDEX_FILE))
        self.assertEqual(index.d, 384, f"Expected 384-dimensional embeddings, got {index.d}")

        with open(CASE_MAPPING_FILE, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

        self.assertEqual(index.ntotal, len(mapping_data["cases"]))
        self.assertEqual(index.ntotal, mapping_data["total_cases"])

    def test_search_returns_ranked_l2_results(self):
        """Verify that retrieval returns results sorted by non-negative L2 distance."""
        query = "Claimant suffered permanent disability in motor accident claiming compensation"
        results = self.retriever.search(query, top_k=3)

        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)

        prev_distance = -1.0
        for r in results:
            self.assertIn("rank", r)
            self.assertIn("case_id", r)
            self.assertIn("case_title", r)
            self.assertIn("court", r)
            self.assertIn("judgment_date", r)
            self.assertIn("decision", r)
            self.assertIn("distance", r)

            # L2 distance must be non-negative
            self.assertGreaterEqual(r["distance"], 0.0)

            # Results must be sorted in ascending order of L2 distance
            self.assertGreaterEqual(r["distance"], prev_distance)
            prev_distance = r["distance"]

    def test_decision_and_outcome_integrity(self):
        """Verify that each indexed case retains its exact extracted outcome without mixing."""
        with open(CASE_MAPPING_FILE, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

        df_pred = pd.read_csv(PREDICTION_DATASET_CSV).set_index("case_id")

        for case_info in mapping_data["cases"]:
            cid = case_info["case_id"]
            expected_decision = df_pred.loc[cid, "decision"]
            self.assertEqual(
                case_info["decision"],
                expected_decision,
                f"Outcome mismatch for {cid}: expected {expected_decision}, got {case_info['decision']}"
            )

    def test_format_results_as_text(self):
        """Verify that the text formatting generates complete evidence blocks."""
        query = "Motor accident compensation appeal"
        results = self.retriever.search(query, top_k=2)
        formatted_text = self.retriever.format_results_as_text(results, query)

        self.assertIn("SEMANTIC SIMILAR-CASE RETRIEVAL", formatted_text)
        self.assertIn("Rank #1", formatted_text)
        self.assertIn("Historical Outcome:", formatted_text)
        self.assertIn("L2 Distance:", formatted_text)


if __name__ == "__main__":
    unittest.main()
