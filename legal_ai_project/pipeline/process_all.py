"""
Master Pipeline Orchestrator for Indian Legal Judgment Dataset Collection.
Coordinates Discovery, Collection, Deduplication, Metadata Extraction,
Cleaning, Decision Extraction, Quality Validation, and Dataset Compilation.
"""

import argparse
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import (
    DEFAULT_QUERY,
    DEFAULT_MAX_DOCS,
    DEFAULT_MAX_PAGES,
    REQUEST_DELAY,
    RAW_DIR,
    METADATA_DIR,
    PROCESSED_DIR,
    DATASET_DIR,
    PIPELINE_LOG_FILE,
    ERROR_LOG_FILE,
    MANIFEST_CSV,
    PREDICTION_DATASET_CSV
)
from pipeline.collector import IndianKanoonCollector
from pipeline.duplicate_detector import DuplicateDetector
from pipeline.metadata_extractor import MetadataExtractor
from pipeline.cleaner import DocumentCleaner
from pipeline.decision_extractor import DecisionExtractor
from pipeline.validator import DataQualityValidator
from pipeline.dataset_builder import DatasetBuilder


def configure_logging(verbose: bool = False):
    """Configure multi-target logging for console, execution log, and error log."""
    PIPELINE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(PIPELINE_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def log_error(case_id: str, stage: str, exception: Exception):
    """Log isolated pipeline error with traceback to error log file."""
    error_msg = f"[{datetime.now().isoformat()}] Error in Case '{case_id}' during '{stage}': {str(exception)}\n"
    tb_str = traceback.format_exc()

    logging.getLogger(__name__).error("Error in %s during %s: %s", case_id, stage, exception)

    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(error_msg)
        f.write(tb_str)
        f.write("\n" + "-" * 60 + "\n")


class LegalDataPipeline:
    """End-to-end automated pipeline for legal judgment dataset building."""

    def __init__(
        self,
        query: str = DEFAULT_QUERY,
        max_docs: int = DEFAULT_MAX_DOCS,
        max_pages: int = DEFAULT_MAX_PAGES,
        delay: float = REQUEST_DELAY,
        skip_collector: bool = False,
        force: bool = False
    ):
        self.query = query
        self.max_docs = max_docs
        self.max_pages = max_pages
        self.delay = delay
        self.skip_collector = skip_collector
        self.force = force

        # Initialize sub-agents
        self.duplicate_detector = DuplicateDetector()
        self.collector = IndianKanoonCollector(
            query=self.query,
            max_docs=self.max_docs,
            max_pages=self.max_pages,
            delay=self.delay,
            duplicate_detector=self.duplicate_detector
        )
        self.metadata_extractor = MetadataExtractor()
        self.cleaner = DocumentCleaner()
        self.decision_extractor = DecisionExtractor()
        self.validator = DataQualityValidator()
        self.dataset_builder = DatasetBuilder()

    def run(self) -> Dict:
        """Execute all stages of the pipeline sequentially with fault isolation."""
        logger = logging.getLogger(__name__)

        print("\n" + "=" * 65)
        print("  LEGAL JUDGMENT DATASET COLLECTION PIPELINE")
        print("  Domain: Motor Accident Claims & Judgments (Indian Kanoon)")
        print("=" * 65)
        print(f"Search query        : {self.query}")
        print(f"Target case limit   : {self.max_docs}")
        print(f"Skip live crawler   : {self.skip_collector}")
        print(f"Force re-process    : {self.force}")
        print("=" * 65 + "\n")

        # -------------------------------------------------------------
        # STAGE 1: Automatic Discovery & Collection
        # -------------------------------------------------------------
        collector_stats = {"discovered_urls": 0, "collected": 0, "duplicates_skipped": 0}
        if not self.skip_collector:
            print("[STAGE 1/6] Starting dataset collection...")
            collector_stats = self.collector.collect()
        else:
            print("[STAGE 1/6] Live collection skipped (--skip-collector flag active).")

        # Discover all available raw cases
        raw_files = sorted(list(RAW_DIR.glob("*.txt")))
        total_raw_cases = len(raw_files)
        print(f"\n[INFO] Total raw judgment files in storage: {total_raw_cases}")

        if total_raw_cases == 0:
            print("[WARN] No judgment files available to process. Exiting.")
            return {}

        # -------------------------------------------------------------
        # STAGE 2: Metadata Extraction
        # -------------------------------------------------------------
        print("\n[STAGE 2/6] Extracting structured metadata...")
        meta_success = 0
        for idx, raw_file in enumerate(raw_files, start=1):
            case_id = raw_file.stem
            meta_json = METADATA_DIR / f"{case_id}.json"
            if meta_json.exists() and not self.force:
                meta_success += 1
                continue
            try:
                res = self.metadata_extractor.extract_and_save(case_id)
                if res:
                    meta_success += 1
            except Exception as e:
                log_error(case_id, "Metadata Extraction", e)

        print(f"   [OK] Metadata extraction completed: {meta_success}/{total_raw_cases}")

        # -------------------------------------------------------------
        # STAGE 3: Document Cleaning
        # -------------------------------------------------------------
        print("\n[STAGE 3/6] Cleaning judgments and removing website noise...")
        clean_success = 0
        for idx, raw_file in enumerate(raw_files, start=1):
            case_id = raw_file.stem
            proc_file = PROCESSED_DIR / f"{case_id}.txt"
            if proc_file.exists() and not self.force:
                clean_success += 1
                continue
            try:
                res = self.cleaner.clean_and_save(case_id)
                if res:
                    clean_success += 1
            except Exception as e:
                log_error(case_id, "Document Cleaning", e)

        print(f"   [OK] Cleaning completed: {clean_success}/{total_raw_cases}")

        # -------------------------------------------------------------
        # STAGE 4: Document Type & Decision Extraction
        # -------------------------------------------------------------
        print("\n[STAGE 4/6] Classifying document types and extracting operative decisions...")
        decision_success = 0
        for idx, raw_file in enumerate(raw_files, start=1):
            case_id = raw_file.stem
            try:
                res = self.decision_extractor.process_and_update(case_id)
                if res:
                    decision_success += 1
            except Exception as e:
                log_error(case_id, "Decision Extraction", e)

        print(f"   [OK] Decision extraction completed: {decision_success}/{total_raw_cases}")

        # -------------------------------------------------------------
        # STAGE 5: Quality Validation & Filtering
        # -------------------------------------------------------------
        print("\n[STAGE 5/6] Validating quality and segmenting candidate vs prediction cases...")
        usable_count = 0
        excluded_count = 0
        for idx, raw_file in enumerate(raw_files, start=1):
            case_id = raw_file.stem
            try:
                res = self.validator.validate_and_save(case_id)
                if res:
                    if res.get("processing_status") == "usable":
                        usable_count += 1
                    else:
                        excluded_count += 1
            except Exception as e:
                log_error(case_id, "Quality Validation", e)

        print(f"   [OK] Quality validation completed: {usable_count} usable, {excluded_count} excluded")

        # -------------------------------------------------------------
        # STAGE 6: Dataset Compilation & Manifest Generation
        # -------------------------------------------------------------
        print("\n[STAGE 6/6] Compiling dataset manifests and summary reports...")
        summary = self.dataset_builder.build()

        # -------------------------------------------------------------
        # Final Summary Console Display
        # -------------------------------------------------------------
        print("\n" + "=" * 65)
        print("PIPELINE EXECUTION COMPLETED")
        print("=" * 65)
        print(f"Search query                 : {self.query}")
        print(f"Judgment links discovered    : {collector_stats.get('discovered_urls', 0)}")
        print(f"New documents collected       : {collector_stats.get('collected', 0)}")
        print(f"Duplicates skipped           : {collector_stats.get('duplicates_skipped', 0)}")
        print(f"Metadata extraction completed: {meta_success}")
        print(f"Cleaning completed           : {clean_success}")
        print(f"Decision extraction completed: {decision_success}")
        print(f"Usable cases (Prediction)    : {summary.get('usable_prediction_cases', 0)}")
        print(f"Excluded cases               : {summary.get('excluded_cases', 0)}")
        print(f"Average quality score        : {summary.get('average_quality_score', 0.0):.2f}/1.00")
        print(f"Manifest CSV                 : {MANIFEST_CSV}")
        print(f"Prediction Dataset CSV       : {PREDICTION_DATASET_CSV}")
        print("=" * 65 + "\n")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Automated Legal Judgment Dataset Collection Pipeline (Indian Kanoon)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Search query or legal domain (default: 'motor accident doctypes:judgments')"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=DEFAULT_MAX_DOCS,
        help="Maximum target cases to discover and collect"
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="Maximum search result pages to scan"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY,
        help="Delay in seconds between HTTP requests"
    )
    parser.add_argument(
        "--skip-collector",
        action="store_true",
        help="Skip web collection and process existing local raw files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction and re-cleaning of all existing documents"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging"
    )

    args = parser.parse_args()
    configure_logging(verbose=args.verbose)

    pipeline = LegalDataPipeline(
        query=args.query,
        max_docs=args.max_docs,
        max_pages=args.pages,
        delay=args.delay,
        skip_collector=args.skip_collector,
        force=args.force
    )

    pipeline.run()


if __name__ == "__main__":
    main()