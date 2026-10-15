"""
Metadata Extraction Agent for Indian Legal Judgments.
Extracts structured metadata: Case ID, Title, Court, Judgment Date, Case Number,
Bench/Judges, Parties, Legal Acts/Sections (Motor Vehicles Act, IPC), Citations, and Source URL.
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dateutil import parser as date_parser

from pipeline.config import (
    DOC_ENDPOINT,
    RAW_DIR,
    METADATA_DIR,
    PIPELINE_LOG_FILE
)

logger = logging.getLogger(__name__)

# Precompiled Court Recognition Patterns
KNOWN_COURTS = [
    ("Supreme Court of India", r"supreme\s+court\s+of\s+india"),
    ("Delhi High Court", r"delhi\s+high\s+court|high\s+court\s+of\s+delhi"),
    ("Karnataka High Court", r"karnataka\s+high\s+court|high\s+court\s+of\s+karnataka"),
    ("Madras High Court", r"madras\s+high\s+court|high\s+court\s+of\s+judicature\s+at\s+madras"),
    ("Bombay High Court", r"bombay\s+high\s+court|high\s+court\s+of\s+bombay|high\s+court\s+of\s+judicature\s+at\s+bombay"),
    ("Allahabad High Court", r"allahabad\s+high\s+court|high\s+court\s+of\s+judicature\s+at\s+allahabad"),
    ("Punjab and Haryana High Court", r"punjab\s+and\s+haryana\s+high\s+court|high\s+court\s+of\s+punjab\s+and\s+haryana"),
    ("Gujarat High Court", r"gujarat\s+high\s+court|high\s+court\s+of\s+gujarat"),
    ("Kerala High Court", r"kerala\s+high\s+court|high\s+court\s+of\s+kerala"),
    ("Calcutta High Court", r"calcutta\s+high\s+court|high\s+court\s+at\s+calcutta"),
    ("Andhra Pradesh High Court", r"andhra\s+pradesh\s+high\s+court|high\s+court\s+of\s+andhra\s+pradesh"),
    ("Telangana High Court", r"telangana\s+high\s+court|high\s+court\s+for\s+the\s+state\s+of\s+telangana"),
    ("Rajasthan High Court", r"rajasthan\s+high\s+court|high\s+court\s+of\s+judicature\s+for\s+rajasthan"),
    ("Madhya Pradesh High Court", r"madhya\s+pradesh\s+high\s+court|high\s+court\s+of\s+madhya\s+pradesh"),
    ("Patna High Court", r"patna\s+high\s+court|high\s+court\s+of\s+judicature\s+at\s+patna"),
    ("Orissa High Court", r"orissa\s+high\s+court|high\s+court\s+of\s+orissa"),
    ("Gauhati High Court", r"gauhati\s+high\s+court|high\s+court\s+of\s+assam"),
    ("Himachal Pradesh High Court", r"himachal\s+pradesh\s+high\s+court|high\s+court\s+of\s+himachal\s+pradesh"),
    ("Jharkhand High Court", r"jharkhand\s+high\s+court|high\s+court\s+of\s+jharkhand"),
    ("Uttarakhand High Court", r"uttarakhand\s+high\s+court|high\s+court\s+of\s+uttarakhand"),
    ("Chhattisgarh High Court", r"chhattisgarh\s+high\s+court|high\s+court\s+of\s+chhattisgarh"),
    ("Motor Accidents Claims Tribunal", r"motor\s+accidents?\s+claims?\s+tribunal|m\.?a\.?c\.?t\.?"),
    ("National Consumer Disputes Redressal Commission", r"national\s+consumer\s+disputes\s+redressal\s+commission|ncdrc")
]

# Case Number Regexes
CASE_NUMBER_PATTERNS = [
    r"((?:Miscellaneous\s+First\s+Appeal|Civil\s+Miscellaneous\s+Appeal|First\s+Appeal\s+(?:From\s+Order)?|Motor\s+Accident\s+Claims?\s+Appeal|M\.?A\.?C\.?A\.?|MACA|M\.?F\.?A\.?|MFA|C\.?M\.?A\.?|CMA|F\.?A\.?O\.?|FAO|M\.?A\.?C\.?\s*App\.?)\s*(?:No\.?|Nos\.?)?\s*[\d\/\-\s]+(?:of|\/)\s*\d{4}(?:\s*\([A-Za-z0-9\s\-]+\))?)",
    r"((?:Civil|Criminal)\s+Appeal\s+(?:No\.?|Nos\.?)\s*[\d,\-\s]+(?:of|\/)\s*\d{4})",
    r"((?:Special\s+Leave\s+Petition|SLP)\s*(?:\([^\)]+\))?\s*(?:No\.?|Nos\.?)\s*[\d,\-\s]+(?:of|\/)\s*\d{4})",
    r"((?:Writ\s+Petition|W\.?P\.?)\s*(?:\([^\)]+\))?\s*(?:No\.?|Nos\.?)\s*[\d,\-\s]+(?:of|\/)\s*\d{4})",
    r"((?:M\.?V\.?C\.?|M\.?A\.?C\.?P\.?|Claim\s+Petition)\s*(?:No\.?|Nos\.?)?\s*[\d,\-\s]+(?:of|\/)\s*\d{4})"
]

# Legal Statutes & Sections Patterns (Focus: Motor Accident Domain)
STATUTE_PATTERNS = [
    ("Motor Vehicles Act, 1988", [
        (r"(?:Section|Sec\.|u/s|under\s+section)\s+(166|163-?A|173|140|147|149|170|168|165|134|185|187)\b", "Section {}"),
        (r"Motor\s+Vehicles?\s+Act(?:,\s*1988)?", "Motor Vehicles Act, 1988")
    ]),
    ("Motor Vehicles Act, 1939", [
        (r"(?:Section|Sec\.|u/s|under\s+section)\s+(110-?[A-F]|95|96)\b", "Section {}"),
        (r"Motor\s+Vehicles?\s+Act(?:,\s*1939)?", "Motor Vehicles Act, 1939")
    ]),
    ("Indian Penal Code, 1860", [
        (r"(?:Section|Sec\.|u/s|under\s+section)\s+(279|304-?A|337|338|427|304)\b", "Section {}"),
        (r"Indian\s+Penal\s+Code", "IPC")
    ]),
    ("Code of Civil Procedure, 1908", [
        (r"(?:Order\s+(?:XLI|41|XXI|21|VII|7)|Section\s+(?:96|100|151|115))\b", "{}"),
        (r"Code\s+of\s+Civil\s+Procedure|CPC", "CPC")
    ])
]

# Major Insurance Companies (common in Motor Accident Claims)
KNOWN_INSURERS = [
    "New India Assurance",
    "United India Insurance",
    "National Insurance",
    "Oriental Insurance",
    "ICICI Lombard",
    "Bajaj Allianz",
    "HDFC ERGO",
    "Reliance General Insurance",
    "SBI General Insurance",
    "Tata AIG",
    "IFFCO Tokio",
    "Royal Sundaram",
    "Shriram General Insurance",
    "Cholamandalam MS General Insurance",
    "Future Generali",
    "Universal Sompo"
]


class MetadataExtractor:
    """Extracts structured legal metadata from raw Indian Kanoon judgments."""

    def __init__(self):
        pass

    @staticmethod
    def _extract_doc_id(case_id: str) -> str:
        """Extract numeric doc ID from case_id."""
        clean = case_id.replace("case_", "").replace(".txt", "").replace(".json", "")
        return clean

    def extract_title_and_date_from_header(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract case title and judgment date from the initial lines of the text.
        Returns: (title, date_str, formatted_iso_date)
        """
        # Look at the first 30 lines
        lines = [line.strip() for line in text.split("\n")[:30] if line.strip()]

        for line in lines:
            # Pattern: "Party A vs/v/versus Party B on DD Month, YYYY"
            match = re.search(
                r"^(.+?)\s+(?:vs\.?|v\.?|versus)\s+(.+?)\s+on\s+([0-9]{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+[0-9]{4})",
                line,
                re.IGNORECASE
            )
            if match:
                title = f"{match.group(1).strip()} vs {match.group(2).strip()}"
                date_str = match.group(3).strip()
                iso_date = self._normalize_date(date_str)
                return title, date_str, iso_date

            # Pattern without "on Date": "Party A vs Party B"
            match_title_only = re.search(
                r"^([A-Z][A-Za-z0-9\s\.\,\&\'\-]+)\s+(?:vs\.?|v\.?|versus)\s+([A-Z][A-Za-z0-9\s\.\,\&\'\-]+)$",
                line,
                re.IGNORECASE
            )
            if match_title_only and len(line) < 180 and not line.startswith("Skip to"):
                title = f"{match_title_only.group(1).strip()} vs {match_title_only.group(2).strip()}"
                return title, None, None

        return None, None, None

    def extract_court(self, text: str) -> Optional[str]:
        """Detect the court name from judgment header and content."""
        header_text = "\n".join(text.split("\n")[:100]).lower()

        # Prioritize header occurrences
        for court_name, pattern in KNOWN_COURTS:
            if re.search(pattern, header_text):
                return court_name

        # Fallback to scanning first 20,000 characters
        sample_text = text[:20000].lower()
        for court_name, pattern in KNOWN_COURTS:
            if re.search(pattern, sample_text):
                return court_name

        return None

    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Parse human date string into standard ISO YYYY-MM-DD format."""
        if not date_str:
            return None
        # Clean ordinal suffixes like 1st, 2nd, 3rd, 18th
        clean_str = re.sub(r"(\d+)(?:st|nd|rd|th)", r"\1", date_str)
        try:
            parsed = date_parser.parse(clean_str, fuzzy=True)
            return parsed.strftime("%Y-%m-%d")
        except Exception:
            return None

    def extract_judgment_date(self, text: str, fallback_date: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract judgment date if not already found in header title line.
        Returns: (raw_date_string, iso_date_string)
        """
        if fallback_date:
            iso = self._normalize_date(fallback_date)
            return fallback_date, iso

        header_lines = text.split("\n")[:50]
        date_regexes = [
            r"(?:Decided\s+On|Dated|Date\s+of\s+Decision|Judgment\s+dated|Pronounced\s+on)[\s\:\-]+([0-9]{1,2}[\.\-\/\s]+[A-Za-z0-9]+[\.\-\/\s]+[0-9]{4})",
            r"(?:on|dated)\s+([0-9]{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+[0-9]{4})",
            r"([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4})"
        ]

        for line in header_lines:
            for regex in date_regexes:
                match = re.search(regex, line, re.IGNORECASE)
                if match:
                    raw_date = match.group(1).strip()
                    iso_date = self._normalize_date(raw_date)
                    if iso_date:
                        return raw_date, iso_date

        return None, None

    def extract_case_number(self, text: str) -> Optional[str]:
        """Extract formal case appeal or petition number."""
        header_text = "\n".join(text.split("\n")[:80])

        for pattern in CASE_NUMBER_PATTERNS:
            match = re.search(pattern, header_text, re.IGNORECASE | re.MULTILINE)
            if match:
                clean_num = re.sub(r"\s+", " ", match.group(1)).strip()
                return clean_num

        # Search deeper in first 10,000 characters
        sample_text = text[:10000]
        for pattern in CASE_NUMBER_PATTERNS:
            match = re.search(pattern, sample_text, re.IGNORECASE)
            if match:
                clean_num = re.sub(r"\s+", " ", match.group(1)).strip()
                return clean_num

        return None

    def extract_bench(self, text: str) -> List[str]:
        """Extract bench/judges names."""
        bench_list = []
        header_text = "\n".join(text.split("\n")[:60])

        # Look for "Bench: ...", "Author: ...", "CORAM: ..."
        bench_match = re.search(r"(?:Bench|CORAM|Author)[\s\:\-]+(.+?)(?=\n[A-Z0-9\s]{4,}|\n\n|JUDGMENT|ORDER)",
                                header_text, re.IGNORECASE | re.DOTALL)
        if bench_match:
            raw_bench = bench_match.group(1).strip()
            # Split by comma or 'and' or newlines
            raw_judges = re.split(r",|\band\b|\n", raw_bench)
            for j in raw_judges:
                clean_j = re.sub(r"^(?:Hon['’]ble|Mr\.|Ms\.|Mrs\.|Justice|Dr\.)\s*", "", j.strip(), flags=re.IGNORECASE)
                clean_j = clean_j.strip(" ,.-")
                if len(clean_j) > 2 and clean_j.lower() not in ("j.", "c.j.", "jj.", "judge"):
                    bench_list.append(clean_j)

        # Also search for "HON'BLE MR. JUSTICE X" in header lines
        if not bench_list:
            for line in header_text.split("\n"):
                match = re.search(r"(?:HON['’]BLE\s+(?:MR\.|MS\.|MRS\.)?\s*JUSTICE\s+([A-Z\.\s]+))", line, re.IGNORECASE)
                if match:
                    name = match.group(1).strip(" ,.-")
                    if len(name) > 3 and name not in bench_list:
                        bench_list.append(name)

        return bench_list

    def extract_parties(self, case_title: Optional[str], text: str) -> Dict[str, Any]:
        """Extract petitioner/appellant, respondent, and involved insurance company."""
        parties = {
            "appellant_petitioner": None,
            "respondent": None,
            "insurance_company": None
        }

        if case_title and " vs " in case_title:
            parts = case_title.split(" vs ", 1)
            parties["appellant_petitioner"] = parts[0].strip()
            parties["respondent"] = parts[1].strip()

        # Identify insurance company
        sample_text = text[:15000]
        for insurer in KNOWN_INSURERS:
            if re.search(rf"\b{re.escape(insurer)}\b", sample_text, re.IGNORECASE):
                parties["insurance_company"] = insurer
                break

        return parties

    def extract_legal_sections(self, text: str) -> Dict[str, List[str]]:
        """Extract cited statutes and sections."""
        cited = {}

        for statute_name, patterns in STATUTE_PATTERNS:
            found_sections = []
            for pattern, template in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for m in matches:
                    if m.groups():
                        sec = m.group(1).strip()
                        formatted = template.format(sec) if "{}" in template else template
                    else:
                        formatted = template
                    if formatted not in found_sections:
                        found_sections.append(formatted)
            if found_sections:
                cited[statute_name] = sorted(found_sections)

        return cited

    def extract_citations(self, text: str) -> List[str]:
        """Extract equivalent citations (e.g. 2000 ACJ 1463, AIR 2001 SC 100)."""
        citations = []
        header_text = "\n".join(text.split("\n")[:40])

        cite_match = re.search(r"Equivalent\s+citations\s*:\s*(.+?)(?=\n[A-Z]|\n\n)", header_text, re.IGNORECASE)
        if cite_match:
            raw_cites = cite_match.group(1).strip()
            for c in raw_cites.split(","):
                clean_c = c.strip()
                if clean_c and clean_c not in citations:
                    citations.append(clean_c)

        # Standard citation regex: e.g. "2018 (3) TAC 123", "AIR 1998 SC 250"
        std_matches = re.finditer(r"\b(\d{4}\s+(?:\(\d+\)\s+)?[A-Z\.]+\s+\d+)\b", header_text)
        for sm in std_matches:
            c_str = sm.group(1).strip()
            if c_str not in citations and len(c_str) < 35:
                citations.append(c_str)

        return citations

    def process_document(
        self,
        case_id: str,
        raw_text: str,
        source_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract full metadata dictionary from raw judgment text.
        """
        doc_id = self._extract_doc_id(case_id)
        if not source_url:
            source_url = f"{DOC_ENDPOINT}{doc_id}/" if doc_id.isdigit() else ""

        title, date_str, iso_date = self.extract_title_and_date_from_header(raw_text)
        if not title:
            # Fallback title from first valid non-boilerplate line
            first_lines = [l.strip() for l in raw_text.split("\n")[:10] if len(l.strip()) > 5 and "skip to" not in l.lower()]
            title = first_lines[0] if first_lines else f"Case {doc_id}"

        raw_date, iso_date = self.extract_judgment_date(raw_text, fallback_date=date_str)
        court = self.extract_court(raw_text)
        case_number = self.extract_case_number(raw_text)
        bench = self.extract_bench(raw_text)
        parties = self.extract_parties(title, raw_text)
        legal_sections = self.extract_legal_sections(raw_text)
        citations = self.extract_citations(raw_text)

        metadata = {
            "case_id": case_id,
            "doc_id": doc_id,
            "case_title": title,
            "court": court,
            "judgment_date": iso_date or raw_date,
            "raw_judgment_date": raw_date,
            "case_number": case_number,
            "parties": parties,
            "bench": bench,
            "legal_sections": legal_sections,
            "citations": citations,
            "source_url": source_url,
            "document_type": "unknown",
            "decision": "not_identified",
            "decision_evidence": None,
            "raw_text_length": len(raw_text),
            "collection_timestamp": datetime.now(timezone.utc).isoformat()
        }

        return metadata

    def extract_and_save(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Extract metadata for a specific case file and save to data/metadata/."""
        raw_file = RAW_DIR / f"{case_id}.txt"
        if not raw_file.exists():
            # Try finding without 'case_' prefix
            clean_name = case_id.replace("case_", "")
            alt_file = RAW_DIR / f"{clean_name}.txt"
            if alt_file.exists():
                raw_file = alt_file
            else:
                logger.error("Raw case file not found: %s", raw_file)
                return None

        try:
            raw_text = raw_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error("Failed to read %s: %s", raw_file, e)
            return None

        metadata = self.process_document(case_id, raw_text)

        # Preserve existing metadata fields if already present (e.g. source_url, collection_date)
        meta_file = METADATA_DIR / f"{case_id}.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    for k, v in existing.items():
                        if v is not None and metadata.get(k) is None:
                            metadata[k] = v
            except Exception:
                pass

        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        logger.info("Metadata extracted and saved: %s", meta_file.name)
        return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Extract structured metadata from legal judgments")
    parser.add_argument("case_file", nargs="?", help="Specific case ID/file to process (optional)")
    args = parser.parse_args()

    extractor = MetadataExtractor()

    if args.case_file:
        clean_id = args.case_file.replace(".txt", "").replace(".json", "")
        if not clean_id.startswith("case_") and clean_id.isdigit():
            clean_id = f"case_{clean_id}"
        extractor.extract_and_save(clean_id)
    else:
        # Process all raw files
        raw_files = list(RAW_DIR.glob("*.txt"))
        logger.info("Processing metadata for %d raw files...", len(raw_files))
        for rf in raw_files:
            extractor.extract_and_save(rf.stem)
        logger.info("Metadata extraction completed for all cases.")
