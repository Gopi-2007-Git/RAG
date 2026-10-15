"""
Duplicate Detection Module.
Handles both document ID matching and content-based cryptographic hashing (SHA-256)
to prevent redundant downloads and duplicate case processing.
"""

import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Set

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import (
    HASH_REGISTRY_FILE,
    RAW_DIR,
    METADATA_DIR
)

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """Detects duplicate legal judgments by Indian Kanoon Document ID and Text Hash."""

    def __init__(self, registry_path: Path = HASH_REGISTRY_FILE, auto_scan: bool = True):
        self.registry_path = Path(registry_path)
        self.doc_ids: Set[str] = set()
        self.hash_to_case_id: Dict[str, str] = {}
        self._load_registry()
        if auto_scan:
            self.scan_existing_files()

    def _load_registry(self) -> None:
        """Load persistent hash registry if it exists."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.hash_to_case_id = data.get("hashes", {})
                    self.doc_ids = set(data.get("doc_ids", []))
                logger.debug("Loaded %d hashes and %d doc IDs from registry",
                             len(self.hash_to_case_id), len(self.doc_ids))
            except Exception as e:
                logger.warning("Failed to load hash registry from %s: %s", self.registry_path, e)
                self.hash_to_case_id = {}
                self.doc_ids = set()

    def save_registry(self) -> None:
        """Save hash registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump({
                    "hashes": self.hash_to_case_id,
                    "doc_ids": sorted(list(self.doc_ids))
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save hash registry: %s", e)

    @staticmethod
    def clean_case_id(case_id: str) -> str:
        """Standardize case ID string (e.g., 'case_1395762.txt' -> 'case_1395762')."""
        clean = case_id.replace(".txt", "").replace(".json", "")
        if not clean.startswith("case_") and clean.isdigit():
            clean = f"case_{clean}"
        return clean

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text before hashing to ignore whitespace/case variations."""
        if not text:
            return ""
        # Lowercase, remove non-alphanumeric chars, normalize whitespace
        text_lower = text.lower()
        cleaned = re.sub(r"[^\w\s]", " ", text_lower)
        normalized = re.sub(r"\s+", " ", cleaned).strip()
        return normalized

    def compute_hash(self, text: str) -> str:
        """Compute SHA-256 hash of normalized text."""
        normalized = self.normalize_text(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def scan_existing_files(self) -> None:
        """Scan raw directory to index existing case IDs and text hashes."""
        if not RAW_DIR.exists():
            return

        for file_path in RAW_DIR.glob("*.txt"):
            case_id = self.clean_case_id(file_path.stem)
            self.doc_ids.add(case_id)
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if len(content) >= 500:
                    text_hash = self.compute_hash(content)
                    if text_hash not in self.hash_to_case_id:
                        self.hash_to_case_id[text_hash] = case_id
            except Exception as e:
                logger.warning("Error reading %s for duplicate indexing: %s", file_path, e)

        self.save_registry()

    def is_duplicate_id(self, case_id: str) -> bool:
        """Check if case ID has already been recorded."""
        clean_id = self.clean_case_id(case_id)
        if clean_id in self.doc_ids:
            return True
        # Also check raw directory directly as fallback
        raw_file = RAW_DIR / f"{clean_id}.txt"
        return raw_file.exists()

    def is_duplicate_content(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check if judgment text content matches an existing document hash."""
        if not text or len(text.strip()) < 500:
            return False, None
        text_hash = self.compute_hash(text)
        if text_hash in self.hash_to_case_id:
            existing_case = self.hash_to_case_id[text_hash]
            return True, existing_case
        return False, None

    def check_duplicate(self, case_id: str, text: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Check if document is duplicate by ID or Content.
        Returns:
            (is_duplicate, reason_or_existing_id)
        """
        clean_id = self.clean_case_id(case_id)

        if self.is_duplicate_id(clean_id):
            return True, f"Duplicate Case ID already exists ({clean_id})"

        if text is not None:
            is_dup_content, existing_case = self.is_duplicate_content(text)
            if is_dup_content:
                return True, f"Identical content hash matching existing case: {existing_case}"

        return False, None

    def register(self, case_id: str, text: Optional[str] = None) -> None:
        """Register a new case ID and content hash."""
        clean_id = self.clean_case_id(case_id)
        self.doc_ids.add(clean_id)
        if text and len(text) >= 500:
            text_hash = self.compute_hash(text)
            self.hash_to_case_id[text_hash] = clean_id
        self.save_registry()
