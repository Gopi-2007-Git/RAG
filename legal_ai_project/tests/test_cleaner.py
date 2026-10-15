"""
Unit tests for Document Cleaner module.
"""

import unittest
from pipeline.cleaner import DocumentCleaner


class TestDocumentCleaner(unittest.TestCase):

    def setUp(self):
        self.cleaner = DocumentCleaner()

    def test_remove_header_and_promo_boilerplate(self):
        raw_text = """Ganesh vs Syed Munned on 18 September, 1998
Skip to main content
Indian Kanoon - Search engine for Indian Law
Search
Search Indian laws and judgments
Main Navigation
Free features
Premium
Prism AI
IKademy
Pricing
Login
Tools for analyzing structure and cite text of judgments
Unlock Advanced Research with
PRISM
AI
Integrated with over 4 crore judgments and laws
-
Know your Kanoon
-
Doc Gen Hub
-
Case Predict AI
Upgrade to Premium
Document Options
Get in PDF
Print it!
[Cites 40, Cited by 106]
Karnataka High Court
Ganesh vs Syed Munned Ahamed on 18 September, 1998
JUDGMENT
1. The appellant sustained injuries in a motor vehicle accident.
2. The tribunal awarded Rs. 50,000 compensation.
"""
        cleaned = self.cleaner.clean(raw_text)

        self.assertNotIn("Skip to main content", cleaned)
        self.assertNotIn("Prism AI", cleaned)
        self.assertNotIn("Unlock Advanced Research", cleaned)
        self.assertNotIn("Upgrade to Premium", cleaned)
        self.assertNotIn("Doc Gen Hub", cleaned)
        self.assertNotIn("[Cites 40, Cited by 106]", cleaned)

        # Core legal text must be preserved
        self.assertIn("Karnataka High Court", cleaned)
        self.assertIn("Ganesh vs Syed Munned", cleaned)
        self.assertIn("appellant sustained injuries in a motor vehicle accident", cleaned)
        self.assertIn("tribunal awarded Rs. 50,000 compensation", cleaned)

    def test_remove_footer_boilerplate(self):
        raw_text = """JUDGMENT
The appeal is allowed with costs.
Related AI tags, queries and research notes
Translation
Top AI Tags
law-regarding-motor-vehicles
Related user Queries
composite negligence
contributory negligence
About
Disclaimer
Privacy Policy
Terms
Case Removal
Blog
Share URL
Mobile View
Case Recast AI"""

        cleaned = self.cleaner.clean(raw_text)
        self.assertIn("The appeal is allowed with costs.", cleaned)
        self.assertNotIn("Related AI tags", cleaned)
        self.assertNotIn("Privacy Policy", cleaned)
        self.assertNotIn("Case Recast AI", cleaned)

    def test_unicode_normalization(self):
        raw_text = "The court\u2019s finding in \u201cCase X\u201d \u2014 valid &amp; affirmed\u00a0here."
        cleaned = self.cleaner.clean(raw_text)
        self.assertEqual(cleaned, "The court's finding in \"Case X\" - valid & affirmed here.")


if __name__ == "__main__":
    unittest.main()
