"""
Dataset Builder Module.
Compiles Candidate Dataset, Prediction Dataset, Unified Manifests (CSV & JSONL),
and statistical summary reports for downstream AI and retrieval tasks.
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from pipeline.config import (
    METADATA_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    DATASET_DIR,
    MANIFEST_CSV,
    MANIFEST_JSONL,
    PREDICTION_DATASET_CSV,
    PREDICTION_DATASET_JSONL,
    CANDIDATE_DATASET_CSV,
    CANDIDATE_DATASET_JSONL,
    SUMMARY_REPORT_JSON,
    PIPELINE_LOG_FILE
)

logger = logging.getLogger(__name__)

# Standard Manifest Schema
MANIFEST_COLUMNS = [
    "case_id",
    "case_title",
    "court",
    "judgment_date",
    "case_number",
    "source_url",
    "document_type",
    "decision",
    "decision_evidence",
    "text_length",
    "collection_date",
    "processing_status",
    "exclusion_reason",
    "data_quality_score"
]


class DatasetBuilder:
    """Compiles candidate and prediction datasets from validated judgment metadata."""

    def __init__(self):
        DATASET_DIR.mkdir(parents=True, exist_ok=True)

    def load_all_records(self) -> List[Dict[str, Any]]:
        """Load and normalize all metadata JSON files."""
        records = []
        if not METADATA_DIR.exists():
            return records

        for meta_file in sorted(METADATA_DIR.glob("*.json")):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                case_id = data.get("case_id", meta_file.stem)
                raw_file = RAW_DIR / f"{case_id}.txt"
                raw_len = len(raw_file.read_text(encoding="utf-8", errors="ignore")) if raw_file.exists() else data.get("raw_text_length", 0)

                record = {
                    "case_id": case_id,
                    "case_title": data.get("case_title", "Unknown"),
                    "court": data.get("court", "Unknown Court"),
                    "judgment_date": data.get("judgment_date", "Unknown Date"),
                    "case_number": data.get("case_number", "N/A"),
                    "source_url": data.get("source_url", ""),
                    "document_type": data.get("document_type", "unknown"),
                    "decision": data.get("decision", "not_identified"),
                    "decision_evidence": data.get("decision_evidence") or "",
                    "text_length": raw_len or data.get("raw_text_length", 0),
                    "collection_date": data.get("collection_timestamp", ""),
                    "processing_status": data.get("processing_status", "excluded"),
                    "exclusion_reason": data.get("exclusion_reason") or "",
                    "data_quality_score": float(data.get("data_quality_score", 0.0))
                }
                records.append(record)
            except Exception as e:
                logger.warning("Failed to load metadata file %s: %s", meta_file, e)

        return records

    def export_csv_and_jsonl(self, records: List[Dict[str, Any]], csv_path: Path, jsonl_path: Path) -> None:
        """Export records to both CSV and JSONL formats."""
        # CSV Export
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for r in records:
                writer.writerow(r)

        # JSONL Export
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def generate_summary(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistical metrics and dataset breakdown."""
        total = len(records)
        usable_records = [r for r in records if r["processing_status"] == "usable"]
        excluded_records = [r for r in records if r["processing_status"] == "excluded"]

        court_counts = Counter(r["court"] for r in records)
        decision_counts = Counter(r["decision"] for r in records)
        doc_type_counts = Counter(r["document_type"] for r in records)
        exclusion_counts = Counter(r["exclusion_reason"] for r in excluded_records if r["exclusion_reason"])

        scores = [r["data_quality_score"] for r in records]
        avg_score = round(sum(scores) / total, 3) if total > 0 else 0.0

        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_candidate_cases": total,
            "usable_prediction_cases": len(usable_records),
            "excluded_cases": len(excluded_records),
            "average_quality_score": avg_score,
            "decision_distribution": dict(decision_counts),
            "document_type_distribution": dict(doc_type_counts),
            "court_distribution": dict(court_counts),
            "exclusion_reason_breakdown": dict(exclusion_counts)
        }

        return summary

    def build(self) -> Dict[str, Any]:
        """Compile manifests and datasets."""
        logger.info("Compiling datasets and manifest files...")
        records = self.load_all_records()

        usable_records = [r for r in records if r["processing_status"] == "usable"]
        candidate_records = list(records)

        # Write Unified Manifest
        self.export_csv_and_jsonl(records, MANIFEST_CSV, MANIFEST_JSONL)
        logger.info("-> Exported Unified Manifest: %s (%d rows)", MANIFEST_CSV.name, len(records))

        # Write Prediction Dataset
        self.export_csv_and_jsonl(usable_records, PREDICTION_DATASET_CSV, PREDICTION_DATASET_JSONL)
        logger.info("-> Exported Prediction Dataset: %s (%d usable rows)", PREDICTION_DATASET_CSV.name, len(usable_records))

        # Write Candidate Dataset
        self.export_csv_and_jsonl(candidate_records, CANDIDATE_DATASET_CSV, CANDIDATE_DATASET_JSONL)
        logger.info("-> Exported Candidate Dataset: %s (%d rows)", CANDIDATE_DATASET_CSV.name, len(candidate_records))

        # Write Summary Report
        summary = self.generate_summary(records)
        with open(SUMMARY_REPORT_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        logger.info("-> Exported Summary Report: %s", SUMMARY_REPORT_JSON.name)

        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    builder = DatasetBuilder()
    summary = builder.build()
    print("\nDataset Summary Report:")
    print(json.dumps(summary, indent=2))
