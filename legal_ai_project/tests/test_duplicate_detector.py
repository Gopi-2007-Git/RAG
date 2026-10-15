"""
Unit tests for duplicate detection module.
"""

import tempfile
import unittest
from pathlib import Path

from pipeline.duplicate_detector import DuplicateDetector


class TestDuplicateDetector(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry_file = Path(self.temp_dir.name) / "test_hash_registry.json"
        self.detector = DuplicateDetector(registry_path=self.registry_file, auto_scan=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_clean_case_id(self):
        self.assertEqual(self.detector.clean_case_id("12345"), "case_12345")
        self.assertEqual(self.detector.clean_case_id("case_12345.txt"), "case_12345")
        self.assertEqual(self.detector.clean_case_id("case_12345.json"), "case_12345")

    def test_normalize_text(self):
        raw = "HIGH COURT OF KARNATAKA\n\nJudgment in M.V.C. No. 100!"
        normalized = self.detector.normalize_text(raw)
        self.assertEqual(normalized, "high court of karnataka judgment in m v c no 100")

    def test_id_duplicate_detection(self):
        self.detector.register("case_1001")
        self.assertTrue(self.detector.is_duplicate_id("case_1001"))
        self.assertTrue(self.detector.is_duplicate_id("1001"))
        self.assertFalse(self.detector.is_duplicate_id("case_9999"))

    def test_content_hash_duplicate_detection(self):
        sample_judgment = (
            "In the High Court of Karnataka at Bengaluru. "
            "Appeal under Section 173 of the Motor Vehicles Act 1988 against award in MVC 500. "
            "The claimant sustained grievous injuries in a road accident involving bus and truck. "
        ) * 10

        self.detector.register("case_5001", sample_judgment)

        # Exact match
        is_dup, matched_case = self.detector.is_duplicate_content(sample_judgment)
        self.assertTrue(is_dup)
        self.assertEqual(matched_case, "case_5001")

        # Minor formatting/whitespace difference should still match
        altered_whitespace = sample_judgment.replace(" ", "  \n ")
        is_dup2, matched_case2 = self.detector.is_duplicate_content(altered_whitespace)
        self.assertTrue(is_dup2)
        self.assertEqual(matched_case2, "case_5001")

        # Completely different text
        different_text = "Criminal Revision Petition under section 397 of CrPC. Bail application." * 15
        is_dup3, _ = self.detector.is_duplicate_content(different_text)
        self.assertFalse(is_dup3)


if __name__ == "__main__":
    unittest.main()
