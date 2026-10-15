"""
Unit tests for Dataset Builder module.
"""

import tempfile
import unittest
from pathlib import Path

from pipeline.dataset_builder import DatasetBuilder, MANIFEST_COLUMNS


class TestDatasetBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = DatasetBuilder()

    def test_summary_generation(self):
        mock_records = [
            {
                "case_id": "case_1",
                "case_title": "Case 1",
                "court": "Delhi High Court",
                "judgment_date": "2020-01-01",
                "case_number": "MAC 100",
                "source_url": "http://example.com/1",
                "document_type": "final_judgment",
                "decision": "appeal_allowed",
                "decision_evidence": "Appeal allowed",
                "text_length": 5000,
                "collection_date": "2023-01-01",
                "processing_status": "usable",
                "exclusion_reason": "",
                "data_quality_score": 0.85
            },
            {
                "case_id": "case_2",
                "case_title": "Case 2",
                "court": "Karnataka High Court",
                "judgment_date": "2021-01-01",
                "case_number": "MFA 200",
                "source_url": "http://example.com/2",
                "document_type": "interim_order",
                "decision": "interim_order",
                "decision_evidence": "Stay granted",
                "text_length": 2000,
                "collection_date": "2023-01-01",
                "processing_status": "excluded",
                "exclusion_reason": "interim_order",
                "data_quality_score": 0.40
            }
        ]

        summary = self.builder.generate_summary(mock_records)

        self.assertEqual(summary["total_candidate_cases"], 2)
        self.assertEqual(summary["usable_prediction_cases"], 1)
        self.assertEqual(summary["excluded_cases"], 1)
        self.assertEqual(summary["average_quality_score"], 0.625)
        self.assertEqual(summary["decision_distribution"]["appeal_allowed"], 1)
        self.assertEqual(summary["exclusion_reason_breakdown"]["interim_order"], 1)

    def test_export_csv_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test_manifest.csv"
            jsonl_path = Path(tmp_dir) / "test_manifest.jsonl"

            mock_records = [
                {
                    "case_id": "case_1",
                    "case_title": "Case 1",
                    "court": "Delhi High Court",
                    "judgment_date": "2020-01-01",
                    "case_number": "MAC 100",
                    "source_url": "http://example.com/1",
                    "document_type": "final_judgment",
                    "decision": "appeal_allowed",
                    "decision_evidence": "Appeal allowed",
                    "text_length": 5000,
                    "collection_date": "2023-01-01",
                    "processing_status": "usable",
                    "exclusion_reason": "",
                    "data_quality_score": 0.85
                }
            ]

            self.builder.export_csv_and_jsonl(mock_records, csv_path, jsonl_path)

            self.assertTrue(csv_path.exists())
            self.assertTrue(jsonl_path.exists())

            # Read back and verify CSV columns
            with open(csv_path, "r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
                for col in MANIFEST_COLUMNS:
                    self.assertIn(col, header)


if __name__ == "__main__":
    unittest.main()
