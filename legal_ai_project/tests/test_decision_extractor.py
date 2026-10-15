"""
Unit tests for Decision Extractor and Document Classifier module.
"""

import unittest
from pipeline.decision_extractor import DecisionExtractor
from pipeline.config import (
    DOC_TYPE_FINAL_JUDGMENT,
    DOC_TYPE_INTERIM_ORDER,
    DOC_TYPE_FURTHER_HEARING
)


class TestDecisionExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = DecisionExtractor()

    def test_classify_document_type(self):
        # Final judgment
        judgment_text = """The accident occurred on 12.01.2018.
The tribunal awarded compensation of Rs. 4,00,000.
Having considered the evidence and submissions, this court finds that the appeal is allowed.""" * 10
        self.assertEqual(self.extractor.classify_document_type(judgment_text), DOC_TYPE_FINAL_JUDGMENT)

        # Interim order
        interim_text = """Heard learned counsel for the parties.
Subject to appellant depositing 50% of the award amount within four weeks, interim stay is granted.
Re-notify on 15.11.2023."""
        self.assertEqual(self.extractor.classify_document_type(interim_text), DOC_TYPE_INTERIM_ORDER)

        # Further hearing
        hearing_text = """Counsel for the respondent seeks time to file counter affidavit.
List for further directions on 20.10.2024."""
        self.assertEqual(self.extractor.classify_document_type(hearing_text), DOC_TYPE_FURTHER_HEARING)

    def test_extract_compensation_enhanced(self):
        text = """The tribunal failed to take future prospects into consideration.
In view of the above calculation, the compensation awarded by the tribunal is enhanced to Rs. 8,50,000 with 7.5% interest.
The insurance company shall deposit the enhanced sum within six weeks."""
        decision, evidence = self.extractor.extract_decision_and_evidence(text, DOC_TYPE_FINAL_JUDGMENT)
        self.assertEqual(decision, "compensation_enhanced")
        self.assertIsNotNone(evidence)
        self.assertIn("enhanced", evidence.lower())

    def test_extract_appeal_dismissed(self):
        text = """The driver of the offending vehicle did not possess a valid driving licence.
Finding no error in the award passed by the tribunal, the appeal is devoid of merits and is dismissed."""
        decision, evidence = self.extractor.extract_decision_and_evidence(text, DOC_TYPE_FINAL_JUDGMENT)
        self.assertEqual(decision, "appeal_dismissed")
        self.assertIn("dismissed", evidence.lower())

    def test_extract_appeal_partly_allowed(self):
        text = """Accordingly, the finding on liability is confirmed but the quantum is modified.
The appeal is partly allowed to the extent indicated above."""
        decision, evidence = self.extractor.extract_decision_and_evidence(text, DOC_TYPE_FINAL_JUDGMENT)
        self.assertEqual(decision, "appeal_partly_allowed")

    def test_extract_order_set_aside(self):
        text = """The tribunal had no territorial jurisdiction to entertain the claim petition.
Consequently, the impugned judgment and award is set aside."""
        decision, evidence = self.extractor.extract_decision_and_evidence(text, DOC_TYPE_FINAL_JUDGMENT)
        self.assertEqual(decision, "order_set_aside")

    def test_unidentified_outcome(self):
        text = """The parties have exchanged their arguments on the question of maintainability.
No final order is passed today."""
        decision, _ = self.extractor.extract_decision_and_evidence(text, DOC_TYPE_FINAL_JUDGMENT)
        self.assertEqual(decision, "not_identified")


if __name__ == "__main__":
    unittest.main()
