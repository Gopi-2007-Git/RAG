"""
Unit tests for Quality Validator module.
"""

import unittest
from pipeline.validator import DataQualityValidator
from pipeline.config import (
    DOC_TYPE_FINAL_JUDGMENT,
    DOC_TYPE_INTERIM_ORDER
)


class TestDataQualityValidator(unittest.TestCase):

    def setUp(self):
        self.validator = DataQualityValidator(min_length=500, min_words=80, min_score=0.60)

    def test_usable_case_validation(self):
        metadata = {
            "case_id": "case_101",
            "case_title": "National Insurance vs Ramesh Kumar",
            "court": "Delhi High Court",
            "judgment_date": "2020-05-15",
            "case_number": "MAC.APP. 123/2019",
            "bench": ["Prathiba M. Singh"],
            "legal_sections": {"Motor Vehicles Act, 1988": ["Section 166", "Section 173"]},
            "citations": ["2020 ACJ 500"],
            "source_url": "https://indiankanoon.org/doc/101/",
            "document_type": DOC_TYPE_FINAL_JUDGMENT,
            "decision": "appeal_dismissed",
            "decision_evidence": "Finding no error in the award, the appeal is dismissed."
        }
        raw_text = "Delhi High Court Judgment in Motor Accident Claim. " * 80
        cleaned_text = "Delhi High Court Judgment in Motor Accident Claim. " * 50

        result = self.validator.validate_case("case_101", metadata, raw_text, cleaned_text)

        self.assertEqual(result["processing_status"], "usable")
        self.assertIsNone(result["exclusion_reason"])
        self.assertGreaterEqual(result["data_quality_score"], 0.65)

    def test_interim_order_exclusion(self):
        metadata = {
            "case_id": "case_102",
            "case_title": "Oriental Insurance vs Suresh",
            "court": "Karnataka High Court",
            "judgment_date": "2021-08-10",
            "source_url": "https://indiankanoon.org/doc/102/",
            "document_type": DOC_TYPE_INTERIM_ORDER,
            "decision": "interim_order"
        }
        raw_text = "Interim stay order text. " * 80
        cleaned_text = "Interim stay order text. " * 50

        result = self.validator.validate_case("case_102", metadata, raw_text, cleaned_text)
        self.assertEqual(result["processing_status"], "excluded")
        self.assertEqual(result["exclusion_reason"], "interim_order")

    def test_insufficient_length_exclusion(self):
        metadata = {
            "case_id": "case_103",
            "case_title": "Ramesh vs Suresh",
            "court": "Madras High Court",
            "judgment_date": "2022-01-01",
            "source_url": "https://indiankanoon.org/doc/103/",
            "document_type": DOC_TYPE_FINAL_JUDGMENT,
            "decision": "appeal_allowed"
        }
        raw_text = "Short text"
        cleaned_text = "Short text"

        result = self.validator.validate_case("case_103", metadata, raw_text, cleaned_text)
        self.assertEqual(result["processing_status"], "excluded")
        self.assertEqual(result["exclusion_reason"], "insufficient_text_length")


if __name__ == "__main__":
    unittest.main()
