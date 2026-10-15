"""
Quality Validation Agent for Legal Judgments.
Evaluates completeness, text length, document type, outcome reliability,
computes quality scores, and routes cases into Prediction Dataset vs. Excluded Dataset.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import (
    MIN_TEXT_LENGTH,
    MIN_WORD_COUNT,
    MIN_QUALITY_SCORE,
    DOC_TYPE_FINAL_JUDGMENT,
    DOC_TYPE_INTERIM_ORDER,
    DOC_TYPE_FURTHER_HEARING,
    DOC_TYPE_PROCEDURAL_ORDER,
    USABLE_DECISION_OUTCOMES,
    RAW_DIR,
    PROCESSED_DIR,
    METADATA_DIR,
    EXCLUDED_DIR,
    PIPELINE_LOG_FILE
)

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Validates judgment records and calculates multidimensional quality scores."""

    def __init__(
        self,
        min_length: int = MIN_TEXT_LENGTH,
        min_words: int = MIN_WORD_COUNT,
        min_score: float = MIN_QUALITY_SCORE
    ):
        self.min_length = min_length
        self.min_words = min_words
        self.min_score = min_score

    def calculate_quality_score(
        self,
        metadata: Dict[str, Any],
        raw_text: str,
        cleaned_text: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate composite quality score (0.00 to 1.00) based on weighted factors.
        Returns: (composite_score, score_breakdown)
        """
        breakdown = {
            "text_richness": 0.0,       # Max: 0.20
            "metadata_completeness": 0.0, # Max: 0.35
            "legal_citations": 0.0,     # Max: 0.15
            "outcome_certainty": 0.0     # Max: 0.30
        }

        # 1. Text richness (0.20)
        word_count = len(cleaned_text.split()) if cleaned_text else 0
        if word_count >= 1000:
            breakdown["text_richness"] = 0.20
        elif word_count >= 500:
            breakdown["text_richness"] = 0.16
        elif word_count >= self.min_words:
            breakdown["text_richness"] = 0.12
        elif word_count > 50:
            breakdown["text_richness"] = 0.05

        # 2. Metadata completeness (0.35)
        title = metadata.get("case_title")
        if title and not title.lower().startswith("case") and " vs " in title:
            breakdown["metadata_completeness"] += 0.10
        elif title:
            breakdown["metadata_completeness"] += 0.05

        if metadata.get("court"):
            breakdown["metadata_completeness"] += 0.10

        if metadata.get("judgment_date"):
            breakdown["metadata_completeness"] += 0.05

        if metadata.get("case_number"):
            breakdown["metadata_completeness"] += 0.05

        if metadata.get("bench") and len(metadata.get("bench", [])) > 0:
            breakdown["metadata_completeness"] += 0.05

        # 3. Legal Sections & Citations (0.15)
        sections = metadata.get("legal_sections", {})
        if sections and len(sections) > 0:
            breakdown["legal_citations"] += 0.10

        citations = metadata.get("citations", [])
        if citations and len(citations) > 0:
            breakdown["legal_citations"] += 0.05

        # 4. Outcome Certainty & Evidence (0.30)
        doc_type = metadata.get("document_type")
        decision = metadata.get("decision")
        evidence = metadata.get("decision_evidence")

        if doc_type == DOC_TYPE_FINAL_JUDGMENT and decision in USABLE_DECISION_OUTCOMES:
            breakdown["outcome_certainty"] += 0.20
            if evidence and len(evidence) > 20:
                breakdown["outcome_certainty"] += 0.10
        elif doc_type == DOC_TYPE_FINAL_JUDGMENT:
            breakdown["outcome_certainty"] += 0.05

        total_score = sum(breakdown.values())
        return round(total_score, 3), breakdown

    def validate_case(
        self,
        case_id: str,
        metadata: Dict[str, Any],
        raw_text: str,
        cleaned_text: str
    ) -> Dict[str, Any]:
        """
        Run full validation suite and determine usability and exclusion reason.
        """
        score, breakdown = self.calculate_quality_score(metadata, raw_text, cleaned_text)
        word_count = len(cleaned_text.split()) if cleaned_text else 0
        char_count = len(raw_text) if raw_text else 0

        # Check conditions
        doc_type = metadata.get("document_type", "unknown")
        decision = metadata.get("decision", "not_identified")
        court = metadata.get("court")
        title = metadata.get("case_title")
        source_url = metadata.get("source_url")

        is_usable = True
        exclusion_reason = None

        if char_count < self.min_length or word_count < self.min_words:
            is_usable = False
            exclusion_reason = "insufficient_text_length"
        elif not source_url or not source_url.startswith("http"):
            is_usable = False
            exclusion_reason = "missing_source_url"
        elif not title or (title.lower().startswith("case") and " vs " not in title):
            is_usable = False
            exclusion_reason = "missing_mandatory_metadata"
        elif doc_type in (DOC_TYPE_INTERIM_ORDER, DOC_TYPE_FURTHER_HEARING):
            is_usable = False
            exclusion_reason = "interim_order"
        elif doc_type == DOC_TYPE_PROCEDURAL_ORDER:
            is_usable = False
            exclusion_reason = "procedural_order"
        elif decision not in USABLE_DECISION_OUTCOMES:
            is_usable = False
            exclusion_reason = "decision_not_identified"
        elif score < self.min_score:
            is_usable = False
            exclusion_reason = "low_quality_score"

        status = "usable" if is_usable else "excluded"

        validation_result = {
            "case_id": case_id,
            "processing_status": status,
            "exclusion_reason": exclusion_reason,
            "data_quality_score": score,
            "quality_breakdown": breakdown,
            "text_length": char_count,
            "word_count": word_count
        }

        return validation_result

    def validate_and_save(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Validate a single case and save diagnostics/updates."""
        raw_file = RAW_DIR / f"{case_id}.txt"
        processed_file = PROCESSED_DIR / f"{case_id}.txt"
        meta_file = METADATA_DIR / f"{case_id}.json"

        if not meta_file.exists():
            logger.error("Metadata file not found for %s", case_id)
            return None

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error("Failed to read metadata for %s: %s", case_id, e)
            return None

        raw_text = raw_file.read_text(encoding="utf-8", errors="ignore") if raw_file.exists() else ""
        cleaned_text = processed_file.read_text(encoding="utf-8", errors="ignore") if processed_file.exists() else ""

        val_result = self.validate_case(case_id, metadata, raw_text, cleaned_text)

        # Update metadata file with validation fields
        metadata["data_quality_score"] = val_result["data_quality_score"]
        metadata["processing_status"] = val_result["processing_status"]
        metadata["exclusion_reason"] = val_result["exclusion_reason"]
        metadata["quality_breakdown"] = val_result["quality_breakdown"]

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        # If excluded, write detailed diagnostics record to data/excluded/
        if val_result["processing_status"] == "excluded":
            EXCLUDED_DIR.mkdir(parents=True, exist_ok=True)
            excluded_file = EXCLUDED_DIR / f"{case_id}.json"
            with open(excluded_file, "w", encoding="utf-8") as f:
                json.dump({
                    "case_id": case_id,
                    "case_title": metadata.get("case_title"),
                    "exclusion_reason": val_result["exclusion_reason"],
                    "document_type": metadata.get("document_type"),
                    "decision": metadata.get("decision"),
                    "data_quality_score": val_result["data_quality_score"],
                    "quality_breakdown": val_result["quality_breakdown"],
                    "source_url": metadata.get("source_url")
                }, f, indent=4, ensure_ascii=False)
            logger.info("Case %s -> EXCLUDED (Reason: %s, Score: %.2f)",
                        case_id, val_result["exclusion_reason"], val_result["data_quality_score"])
        else:
            logger.info("Case %s -> USABLE (Score: %.2f, Decision: %s)",
                        case_id, val_result["data_quality_score"], metadata.get("decision"))

        return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Validate legal judgment quality")
    parser.add_argument("case_file", nargs="?", help="Specific case ID to validate (optional)")
    args = parser.parse_args()

    validator = DataQualityValidator()

    if args.case_file:
        clean_id = args.case_file.replace(".txt", "").replace(".json", "")
        if not clean_id.startswith("case_") and clean_id.isdigit():
            clean_id = f"case_{clean_id}"
        validator.validate_and_save(clean_id)
    else:
        meta_files = list(METADATA_DIR.glob("*.json"))
        logger.info("Validating %d metadata files...", len(meta_files))
        for mf in meta_files:
            validator.validate_and_save(mf.stem)
        logger.info("Validation completed for all documents.")
