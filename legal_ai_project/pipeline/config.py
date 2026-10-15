"""
Configuration module for the Legal Judgment Dataset Collection Pipeline.
Centralizes paths, scraping parameters, quality thresholds, and domain rules.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"
PROCESSED_DIR = DATA_DIR / "processed"
EXCLUDED_DIR = DATA_DIR / "excluded"
DATASET_DIR = DATA_DIR / "dataset"
LOGS_DIR = PROJECT_ROOT / "logs"
DATABASE_DIR = PROJECT_ROOT / "database"

# Ensure essential directories exist
for path in [DATA_DIR, RAW_DIR, METADATA_DIR, PROCESSED_DIR, EXCLUDED_DIR, DATASET_DIR, LOGS_DIR, DATABASE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Manifest & Registry file paths
MANIFEST_CSV = DATASET_DIR / "manifest.csv"
MANIFEST_JSONL = DATASET_DIR / "manifest.jsonl"
PREDICTION_DATASET_CSV = DATASET_DIR / "prediction_dataset.csv"
PREDICTION_DATASET_JSONL = DATASET_DIR / "prediction_dataset.jsonl"
CANDIDATE_DATASET_CSV = DATASET_DIR / "candidate_dataset.csv"
CANDIDATE_DATASET_JSONL = DATASET_DIR / "candidate_dataset.jsonl"
SUMMARY_REPORT_JSON = DATASET_DIR / "summary_report.json"
HASH_REGISTRY_FILE = DATASET_DIR / "hash_registry.json"

# Logging Paths
PIPELINE_LOG_FILE = LOGS_DIR / "pipeline.log"
ERROR_LOG_FILE = LOGS_DIR / "pipeline_errors.log"

# Indian Kanoon Scraping Settings
BASE_URL = "https://indiankanoon.org"
SEARCH_ENDPOINT = f"{BASE_URL}/search/"
DOC_ENDPOINT = f"{BASE_URL}/doc/"

DEFAULT_DOMAIN = "motor accident"
DEFAULT_QUERY = "motor accident doctypes:judgments"
DEFAULT_MAX_DOCS = 20
DEFAULT_MAX_PAGES = 5

REQUEST_TIMEOUT = 30  # seconds
REQUEST_DELAY = 2.0   # seconds base sleep
DELAY_JITTER_MIN = 0.5
DELAY_JITTER_MAX = 1.5
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://indiankanoon.org/",
    "Connection": "keep-alive",
    "DNT": "1"
}

# Quality & Validation Thresholds
MIN_TEXT_LENGTH = 1000    # minimum characters in raw judgment
MIN_WORD_COUNT = 150      # minimum words in cleaned judgment
MIN_QUALITY_SCORE = 0.65  # minimum quality score for prediction dataset inclusion

# Document Type Classifications
DOC_TYPE_FINAL_JUDGMENT = "final_judgment"
DOC_TYPE_INTERIM_ORDER = "interim_order"
DOC_TYPE_PROCEDURAL_ORDER = "procedural_order"
DOC_TYPE_APPLICATION = "application_order"
DOC_TYPE_FURTHER_HEARING = "further_hearing"
DOC_TYPE_UNKNOWN = "unknown"

# Standard Decision Categories (Normalized)
USABLE_DECISION_OUTCOMES = {
    "appeal_allowed",
    "appeal_dismissed",
    "appeal_partly_allowed",
    "petition_allowed",
    "petition_dismissed",
    "petition_partly_allowed",
    "claim_allowed",
    "claim_dismissed",
    "claim_partly_allowed",
    "compensation_enhanced",
    "compensation_reduced",
    "order_set_aside",
    "order_modified",
    "order_confirmed",
    "application_allowed",
    "application_dismissed",
    "remanded"
}

# Excluded Decision / Document Statuses
EXCLUDED_DECISION_OUTCOMES = {
    "not_identified",
    "interim_order",
    "further_hearing",
    "procedural_direction",
    "adjourned"
}
