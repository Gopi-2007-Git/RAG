"""
Decision Extraction and Document Type Classification Agent.
Classifies legal document types (final judgment vs. interim order vs. procedural order)
and extracts domain-specific outcomes with verbatim evidentiary context.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import (
    PROCESSED_DIR,
    METADATA_DIR,
    DOC_TYPE_FINAL_JUDGMENT,
    DOC_TYPE_INTERIM_ORDER,
    DOC_TYPE_PROCEDURAL_ORDER,
    DOC_TYPE_FURTHER_HEARING,
    DOC_TYPE_UNKNOWN,
    USABLE_DECISION_OUTCOMES,
    PIPELINE_LOG_FILE
)

logger = logging.getLogger(__name__)

# Document Type Classification Rules
INTERIM_PATTERNS = [
    r"interim\s+(?:stay|relief|injunction|order)\s+(?:is\s+)?granted",
    r"ad[\-\s]interim\s+(?:stay|order|relief)",
    r"till\s+the\s+next\s+date\s+of\s+hearing",
    r"interim\s+stay\s+shall\s+continue",
    r"pending\s+final\s+disposal\s+of\s+the\s+appeal",
    r"pending\s+disposal\s+of\s+the\s+petition",
    r"stay\s+of\s+execution\s+of\s+the\s+impugned\s+award\s+till",
    r"conditional\s+stay\s+is\s+granted"
]

FURTHER_HEARING_PATTERNS = [
    r"list\s+(?:the\s+matter\s+)?for\s+further\s+(?:directions|consideration|hearing)",
    r"stand\s+over\s+to\s+[0-9]{1,2}",
    r"re[\-\s]notify\s+on\s+[0-9]{1,2}",
    r"post\s+the\s+matter\s+on\s+[0-9]{1,2}",
    r"adjourned\s+to\s+[0-9]{1,2}",
    r"matter\s+to\s+come\s+up\s+on\s+[0-9]{1,2}",
    r"call\s+on\s+[0-9]{1,2}"
]

PROCEDURAL_PATTERNS = [
    r"issue\s+notice\s+to\s+(?:the\s+)?respondents?",
    r"steps\s+be\s+taken\s+within\s+(?:two|three|four|one|\d+)\s+weeks",
    r"service\s+(?:is\s+)?complete",
    r"delay\s+in\s+filing\s+is\s+condoned",
    r"application\s+for\s+condonation\s+of\s+delay\s+is\s+allowed",
    r"exemption\s+from\s+filing\s+certified\s+copy\s+is\s+allowed"
]

# Operative Decision Outcome Expressions (Ordered by Specificity)
DECISION_PATTERNS = [
    # 1. Compensation Enhancement (Specific to Motor Accident cases)
    ("compensation_enhanced", [
        r"compensation\s+(?:awarded\s+(?:by\s+the\s+tribunal\s+)?)?(?:is\s+)?enhanced\s+(?:to|by|from)\s+Rs\.?\s*[\d,\-]+",
        r"award\s+(?:of\s+the\s+tribunal\s+)?is\s+modified\s+and\s+enhanced",
        r"enhanced\s+by\s+a\s+further\s+sum\s+of\s+Rs\.?\s*[\d,\-]+",
        r"claimants?\s+(?:is|are)\s+entitled\s+to\s+(?:an?\s+)?enhanced\s+compensation",
        r"compensation\s+is\s+accordingly\s+enhanced",
        r"enhanced\s+compensation\s+of\s+Rs\.?\s*[\d,\-]+",
        r"compensation\s+is\s+increased\s+to\s+Rs\.?\s*[\d,\-]+",
        r"appeal\s+is\s+partly\s+allowed\s+by\s+enhancing\s+the\s+compensation"
    ]),

    # 2. Compensation Reduction
    ("compensation_reduced", [
        r"compensation\s+(?:awarded\s+)?(?:is\s+)?reduced\s+(?:to|by|from)\s+Rs\.?\s*[\d,\-]+",
        r"award\s+(?:of\s+the\s+tribunal\s+)?is\s+scaled\s+down\s+to",
        r"compensation\s+is\s+decreased\s+to",
        r"reduced\s+the\s+compensation\s+from\s+Rs\.?"
    ]),

    # 3. Appeal Partly Allowed
    ("appeal_partly_allowed", [
        r"appeal[s]?\s+(?:is|are|stands?|hereby)?\s*partly\s+allowed",
        r"appeal[s]?\s+(?:is|are|stands?|hereby)?\s*allowed\s+in\s+part",
        r"appeal[s]?\s+stand[s]?\s+partly\s+allowed",
        r"appeal[s]?\s+succeeds?\s+in\s+part",
        r"mfa\s+is\s+partly\s+allowed",
        r"maca\s+is\s+partly\s+allowed"
    ]),

    # 4. Appeal Allowed
    ("appeal_allowed", [
        r"appeal[s]?\s+(?:is|are|stands?|hereby)?\s*allowed\b",
        r"appeal[s]?\s+succeeds?\s+and\s+(?:is|are)\s+allowed",
        r"appeal[s]?\s+is\s+accordingly\s+allowed",
        r"appeal[s]?\s+is\s+allowed\s+with\s+costs",
        r"appeal[s]?\s+stands?\s+allowed\s+with\s+no\s+order\s+as\s+to\s+costs",
        r"mfa\s+is\s+allowed",
        r"maca\s+is\s+allowed"
    ]),

    # 5. Appeal Dismissed
    ("appeal_dismissed", [
        r"appeal[s]?\s+(?:is|are|stands?|hereby)?\s*dismissed\b",
        r"appeal[s]?\s+is\s+devoid\s+of\s+merit[s]?\s+and\s+(?:is|stands?)\s+dismissed",
        r"appeal[s]?\s+fails?\s+and\s+(?:is|stands?)\s+dismissed",
        r"no\s+merit\s+in\s+(?:this|the)\s+appeal\s+and\s+(?:the\s+same\s+)?is\s+dismissed",
        r"appeal[s]?\s+is\s+accordingly\s+dismissed",
        r"mfa\s+is\s+dismissed",
        r"maca\s+is\s+dismissed"
    ]),

    # 6. Petition Partly Allowed
    ("petition_partly_allowed", [
        r"(?:writ\s+)?petition[s]?\s+(?:is|are|stands?|hereby)?\s*partly\s+allowed",
        r"(?:writ\s+)?petition[s]?\s+(?:is|are|stands?|hereby)?\s*allowed\s+in\s+part"
    ]),

    # 7. Petition Allowed
    ("petition_allowed", [
        r"(?:writ\s+)?petition[s]?\s+(?:is|are|stands?|hereby)?\s*allowed\b",
        r"(?:writ\s+)?petition[s]?\s+stands?\s+allowed",
        r"(?:writ\s+)?petition[s]?\s+succeeds?\s+and\s+is\s+allowed"
    ]),

    # 8. Petition Dismissed
    ("petition_dismissed", [
        r"(?:writ\s+)?petition[s]?\s+(?:is|are|stands?|hereby)?\s*dismissed\b",
        r"(?:writ\s+)?petition[s]?\s+stands?\s+dismissed",
        r"(?:writ\s+)?petition[s]?\s+is\s+devoid\s+of\s+merit[s]?\s+and\s+dismissed"
    ]),

    # 9. Claim Allowed / Partly Allowed / Dismissed
    ("claim_partly_allowed", [
        r"claim\s+(?:petition\s+)?(?:is|stands?)\s+partly\s+allowed",
        r"claim\s+(?:petition\s+)?is\s+allowed\s+in\s+part"
    ]),
    ("claim_allowed", [
        r"claim\s+(?:petition\s+)?(?:is|stands?)\s+allowed\b",
        r"claimant[s]?\s+is\s+held\s+entitled\s+to\s+compensation"
    ]),
    ("claim_dismissed", [
        r"claim\s+(?:petition\s+)?(?:is|stands?)\s+dismissed\b"
    ]),

    # 10. Order Set Aside / Modified / Confirmed
    ("order_set_aside", [
        r"(?:impugned\s+)?(?:judgment|order|award)\s+(?:and\s+decree\s+)?(?:is|stands?|hereby)?\s*set\s+aside",
        r"(?:impugned\s+)?(?:judgment|order|award)\s+is\s+quashed\s+and\s+set\s+aside"
    ]),
    ("order_modified", [
        r"(?:impugned\s+)?(?:judgment|order|award)\s+(?:is|stands?)\s+modified\s+accordingly",
        r"award\s+of\s+the\s+tribunal\s+is\s+modified"
    ]),
    ("order_confirmed", [
        r"(?:impugned\s+)?(?:judgment|order|award)\s+(?:is|stands?)\s+confirmed",
        r"(?:impugned\s+)?(?:judgment|order|award)\s+is\s+upheld\s+and\s+affirmed"
    ]),

    # 11. Remanded / Remitted
    ("remanded", [
        r"matter\s+is\s+remanded\s+back\s+to\s+(?:the\s+)?(?:tribunal|claims\s+tribunal|court)",
        r"remitted\s+back\s+for\s+(?:fresh|re-)consideration",
        r"matter\s+is\s+remitted\s+to\s+the\s+tribunal"
    ]),

    # 12. Application Allowed / Dismissed
    ("application_allowed", [
        r"application\s+(?:is|stands?|hereby)?\s*allowed\b",
        r"i\.?a\.?\s*(?:no\.?\s*\d+)?\s*(?:is|stands?)\s*allowed\b"
    ]),
    ("application_dismissed", [
        r"application\s+(?:is|stands?|hereby)?\s*dismissed\b",
        r"i\.?a\.?\s*(?:no\.?\s*\d+)?\s*(?:is|stands?)\s*dismissed\b"
    ])
]


class DecisionExtractor:
    """Classifies legal document types and extracts operative outcomes with evidence."""

    def __init__(self):
        pass

    def classify_document_type(self, text: str) -> str:
        """
        Classify document as final_judgment, interim_order, procedural_order, or further_hearing.
        """
        tail_text = text[-12000:].lower()
        full_sample = text[:10000].lower() + "\n" + tail_text

        # Check for interim order indicators
        for pattern in INTERIM_PATTERNS:
            if re.search(pattern, tail_text):
                return DOC_TYPE_INTERIM_ORDER

        # Check for further hearing / adjournment
        for pattern in FURTHER_HEARING_PATTERNS:
            if re.search(pattern, tail_text):
                return DOC_TYPE_FURTHER_HEARING

        # Check for procedural orders
        for pattern in PROCEDURAL_PATTERNS:
            if re.search(pattern, full_sample) and len(text) < 2500:
                return DOC_TYPE_PROCEDURAL_ORDER

        # Check for substantive judgment markers
        if any(w in full_sample for w in ["judgment", "appeal is", "petition is", "ordered accordingly", "decree", "award of the tribunal"]):
            return DOC_TYPE_FINAL_JUDGMENT

        return DOC_TYPE_FINAL_JUDGMENT if len(text) >= 2000 else DOC_TYPE_UNKNOWN

    def extract_decision_and_evidence(self, text: str, doc_type: str) -> Tuple[str, Optional[str]]:
        """
        Scan concluding portion of the judgment to extract outcome label and exact evidence snippet.
        Returns: (decision_label, evidence_snippet)
        """
        # If explicitly interim order or further hearing, return appropriate outcome
        if doc_type in (DOC_TYPE_INTERIM_ORDER, DOC_TYPE_FURTHER_HEARING):
            # Extract sentence snippet showing interim status
            tail_lines = [l.strip() for l in text[-4000:].split("\n") if l.strip()]
            for line in reversed(tail_lines):
                for pattern in INTERIM_PATTERNS + FURTHER_HEARING_PATTERNS:
                    if re.search(pattern, line, re.IGNORECASE):
                        return "interim_order", line[:300]
            return "interim_order", None

        # Inspect the last 15,000 characters for final operative decision
        tail_text = text[-15000:]
        tail_lines = [l.strip() for l in tail_text.split("\n") if l.strip()]

        # Scan lines in reverse from the end (most recent operative order first)
        for decision_label, patterns in DECISION_PATTERNS:
            for pattern in patterns:
                # Check lines from end
                for line in reversed(tail_lines):
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        evidence = line.strip()
                        # Expand context slightly if line is very short
                        if len(evidence) < 50:
                            line_idx = tail_lines.index(line)
                            context = tail_lines[max(0, line_idx - 1): min(len(tail_lines), line_idx + 2)]
                            evidence = " ".join(context)
                        return decision_label, evidence[:400]

        # Scan broader multi-line tail text as fallback
        for decision_label, patterns in DECISION_PATTERNS:
            for pattern in patterns:
                match = re.search(pattern, tail_text, re.IGNORECASE)
                if match:
                    start_pos = max(0, match.start() - 60)
                    end_pos = min(len(tail_text), match.end() + 100)
                    snippet = tail_text[start_pos:end_pos].replace("\n", " ").strip()
                    return decision_label, snippet[:400]

        return "not_identified", None

    def process_document(self, text: str) -> Dict[str, Any]:
        """Classify document and extract decision and evidence."""
        doc_type = self.classify_document_type(text)
        decision, evidence = self.extract_decision_and_evidence(text, doc_type)

        return {
            "document_type": doc_type,
            "decision": decision,
            "decision_evidence": evidence
        }

    def process_and_update(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Process cleaned text for a case and update metadata JSON."""
        processed_file = PROCESSED_DIR / f"{case_id}.txt"
        if not processed_file.exists():
            clean_name = case_id.replace("case_", "")
            alt_file = PROCESSED_DIR / f"{clean_name}.txt"
            if alt_file.exists():
                processed_file = alt_file
            else:
                logger.error("Processed file not found: %s", processed_file)
                return None

        try:
            cleaned_text = processed_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error("Failed to read %s: %s", processed_file, e)
            return None

        decision_info = self.process_document(cleaned_text)

        # Update metadata JSON
        meta_file = METADATA_DIR / f"{case_id}.json"
        metadata = {}
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                pass

        metadata.update(decision_info)
        metadata["cleaned_text_length"] = len(cleaned_text)

        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        logger.info("Decision updated for %s: DocType=%s, Decision=%s",
                    case_id, decision_info["document_type"], decision_info["decision"])
        return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Extract legal decision and classify document type")
    parser.add_argument("case_file", nargs="?", help="Specific case ID/file to process (optional)")
    args = parser.parse_args()

    extractor = DecisionExtractor()

    if args.case_file:
        clean_id = args.case_file.replace(".txt", "").replace(".json", "")
        if not clean_id.startswith("case_") and clean_id.isdigit():
            clean_id = f"case_{clean_id}"
        extractor.process_and_update(clean_id)
    else:
        processed_files = list(PROCESSED_DIR.glob("*.txt"))
        logger.info("Extracting decisions for %d processed judgment files...", len(processed_files))
        for pf in processed_files:
            extractor.process_and_update(pf.stem)
        logger.info("Decision extraction completed for all documents.")