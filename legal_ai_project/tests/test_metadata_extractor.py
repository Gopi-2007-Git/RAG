"""
Unit tests for Metadata Extraction module.
"""

import unittest
from pipeline.metadata_extractor import MetadataExtractor


class TestMetadataExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = MetadataExtractor()

    def test_extract_title_and_date(self):
        header = "National Insurance Co. Ltd. vs Smt. Laxmi And Ors. on 14 March, 2012\nDelhi High Court"
        title, raw_date, iso_date = self.extractor.extract_title_and_date_from_header(header)

        self.assertEqual(title, "National Insurance Co. Ltd. vs Smt. Laxmi And Ors.")
        self.assertEqual(raw_date, "14 March, 2012")
        self.assertEqual(iso_date, "2012-03-14")

    def test_court_identification(self):
        text_sc = "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION"
        self.assertEqual(self.extractor.extract_court(text_sc), "Supreme Court of India")

        text_hc = "Karnataka High Court\nGanesh vs Syed Munned Ahamed on 18 September, 1998"
        self.assertEqual(self.extractor.extract_court(text_hc), "Karnataka High Court")

        text_mact = "Before the Motor Accidents Claims Tribunal, Bangalore Rural District"
        self.assertEqual(self.extractor.extract_court(text_mact), "Motor Accidents Claims Tribunal")

    def test_case_number_extraction(self):
        text = """Delhi High Court
MAC.APP. 450/2015
National Insurance vs Ramesh
JUDGMENT"""
        case_num = self.extractor.extract_case_number(text)
        self.assertIsNotNone(case_num)
        self.assertIn("MAC.APP. 450/2015", case_num)

        text_mfa = """Miscellaneous First Appeal No. 1234 of 2018 (MV)
Judgment dated 10.05.2019"""
        case_num_mfa = self.extractor.extract_case_number(text_mfa)
        self.assertIsNotNone(case_num_mfa)

    def test_bench_extraction(self):
        text = """Bench: P. Vishwanatha Shetty, V. Gopala Gowda
JUDGMENT"""
        bench = self.extractor.extract_bench(text)
        self.assertIn("P. Vishwanatha Shetty", bench)
        self.assertIn("V. Gopala Gowda", bench)

    def test_legal_sections_and_statutes(self):
        text = """The claim petition was filed under Section 166 of the Motor Vehicles Act, 1988.
An appeal under Section 173 of the MVA is preferred.
The driver was prosecuted under Section 279 and Section 304A of the Indian Penal Code."""
        sections = self.extractor.extract_legal_sections(text)

        self.assertIn("Motor Vehicles Act, 1988", sections)
        self.assertIn("Indian Penal Code, 1860", sections)

    def test_equivalent_citations(self):
        text = "Equivalent citations: 2000 ACJ 1463, AIR 2000 SC 123\nJUDGMENT"
        citations = self.extractor.extract_citations(text)
        self.assertIn("2000 ACJ 1463", citations)


if __name__ == "__main__":
    unittest.main()
